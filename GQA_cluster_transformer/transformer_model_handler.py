# transformer_model_handler.py
from __future__ import annotations
import os
import math
import numpy as np
from typing import Optional, List, Union, Any
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer ,LlamaTokenizer
import hashlib
import torch
import glob
import sys
import json

# Add your project root imports
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
CLUSTER_MATRIX_DIR = os.path.join(PROJECT_ROOT, "cluster_matrix")
if CLUSTER_MATRIX_DIR not in sys.path:
    sys.path.insert(0, CLUSTER_MATRIX_DIR)

from cluster_matrix_v1 import cluster_matrix
from cluster_matrix_v1 import cluster_zmq
from cluster_matrix_v1 import check_combined_result_values

class hugging_face_model_handler:

    """
    Load HuggingFace safetensors weights lazily.
    Handles multi-shard safetensors and PyTorch .bin files.
    """

    def __init__(
        self,
        model_path: str,
        cluster_zmq_object: Any,  # just placeholder for cluster integration
        percentages: List[float],
        CPU_GPU_select_list: List[bool],
        backend_select_list: List[str],
    ):
        self.model_path = os.path.abspath(model_path)
        self.model_name = os.path.basename(self.model_path)
        self.cluster_zmq_object = cluster_zmq_object
        self.IP_list = cluster_zmq_object.node_IP_list
        self.percentages = percentages
        self.CPU_GPU_select_list = CPU_GPU_select_list
        self.backend_select_list = backend_select_list
        self.split_system = 1
        self.split_dim = 1
        self.q_proj_shape = None
        # Add tokenizer initialization
        self.tokenizer = None
        self._load_tokenizer()

        #self.get_q_shape()

        self.k_proj_list = []
        self.q_proj_list = []
        self.v_proj_list = []
        self.o_proj_list = []
        self.k_proj_bias_list = []
        self.q_proj_bias_list = []
        self.v_proj_bias_list = []
        self.k_layernorm_list = []
        self.q_layernorm_list = []
        self.input_layernorm_list = []
        self.post_attention_layernorm_list = []
        self.operator_norm_list = []
        self.ffn_norm_list = []
        self.qkv_proj_list = []
        self.qkv_proj_bias_list = []
        self.mlp_up_list = [] 
        self.mlp_down_list = []
        self.mlp_gate_list = []
        self.mlp_up_bias_list = []
        self.mlp_down_bias_list = []
        self.mlp_gate_bias_list = []
        self.lm_head_list = []
        self.lm_head_bias_list = []

        self.model_key_words1 = {'bias','input_layernorm','post_attention_layernorm',
                                'ffn_norm','operator_norm','k_layernorm','q_layernorm','q_proj'
                                ,'k_proj','v_proj','o_proj','norm','feed_forward','attention','qkv','qkv_proj'
                                ,'mlp','down_proj','gate_proj','up_proj','feed_forward','w1','w2','w3'
                                ,'lm_head','head','output','classifier','fc1','fc2','fc3',
                                'embedding','embed','embeddings','token_embedding','word_embedding'}

        self.embedding_matrix = None
        self.final_norm_weight = None

        self.num_layers = 0
        self.use_cache = True
        self.attention_bias = False
        self.attention_dropout = False
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.hidden_act = 'silu'
        self.hidden_size = None
        self.initializer_range = 0
        self.intermediate_size = 0
        self.max_position_embeddings = 0
        self.mlp_bias = False
        self.model_type = 'llama'
        self.num_attention_heads = 0
        self.num_hidden_layers = 0
        self.num_key_value_heads = 0
        self.pretraining_tp = 0
        self.rms_norm_eps = 0
        self.rope_scaling = 0
        self.rope_theta = 0 
        self.tie_word_embeddings = False
        self.torch_dtype = 'bfloat16'
        self.use_cache = True
        self.vocab_size = 0

        self.lm_head = None

        self.read_llama_config()
        self.cache_dtype = self._resolve_cache_dtype()

        self.save_progress = 0
        self.load_progress = 0

    def _resolve_cache_dtype(self):
        force_fp32 = str(os.environ.get("CLUSTER_FORCE_FP32", "")).lower() in ("1", "true", "yes", "on")
        if force_fp32:
            return torch.float32

        dtype_str = os.environ.get("CLUSTER_MODEL_DTYPE", "") or str(getattr(self, "torch_dtype", "float16"))
        dtype_str = dtype_str.lower()
        if "bfloat16" in dtype_str or "bf16" in dtype_str:
            return torch.bfloat16
        if "float32" in dtype_str or "fp32" in dtype_str:
            return torch.float32
        if "float16" in dtype_str or "fp16" in dtype_str:
            return torch.float16
        return torch.float32

    def wrap_cluster_matrix(self, tensor_to_wrap,
                            transpose=False,
                            split_dim=1,
                            split_system=1,
                            saveOrload='save',
                            split_matrix=True,
                            key=''):

        if transpose:
            tensor_to_wrap = tensor_to_wrap.T

        cm = cluster_matrix(
            tensor_to_wrap,
            cluster_zmq_object=self.cluster_zmq_object,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=split_matrix,
            dim=split_dim,
            auto_set_up=[split_system, saveOrload],
            matrix_name=key
        )

        if saveOrload == "save":
            if hasattr(cm, "node_matrices") and cm.node_matrices:
                shard_shapes = [s.shape for s in cm.node_matrices]
            else:
                shard_shapes = [tensor_to_wrap.shape]
            return cm, shard_shapes

        return cm, None

    def get_splitting_dimension(self):
        """
        Determine which dimension to split by checking for (n, 1) or (1, n) patterns.
        Returns 0 or 1 (rows or columns).
        """
        shard_files = sorted(glob.glob(os.path.join(self.model_path, "*.safetensors")))

        if not shard_files:
            print(f"No .safetensors files found in {self.model_path}")
            return 1  # default: columns

        try:
            with safe_open(shard_files[0], framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)

                    if tensor.ndim != 2:
                        continue

                    rows, cols = tensor.shape
                    print(f"  Checking {key}: shape = ({rows}, {cols})")

                    if cols == 1 and rows > 1:
                        print("  ✅ (n,1) detected → split dim = 0")
                        return 0

                    if rows == 1 and cols > 1:
                        print("  ✅ (1,n) detected → split dim = 1")
                        return 1

                    if rows > 1 and cols > 1:
                        print("  ✅ regular 2D tensor → split dim = 1")
                        return 1

            print("  ⚠️ No clear pattern found → fallback split dim = 1")
            return 1

        except Exception as e:
            print(f"Error checking splitting dimensions: {e}")
            return 1

    def cache_model_tensors(self, saveOrload="save", split_system=1, split_dim=None):
        """Cache model tensors from safetensors files using cluster_matrix."""

        self.load_progress = 0
        self.save_progress = 0
        layers_seen = set()

        shard_files = sorted(glob.glob(os.path.join(self.model_path, "*.safetensors")))
        if not shard_files:
            print(f"No .safetensors files found in {self.model_path}")
            return

        if self.model_type != "llama":
            raise ValueError("Only llama models are supported")

        # ----------------- Split dim -----------------
        if split_dim is None:
            # Pick a stable split dimension based on an existing shard.
            self.split_dim = self.get_splitting_dimension()
            print(f"📏 Auto-selected splitting dimension: {self.split_dim}")
        else:
            self.split_dim = split_dim

        self.split_system = split_system

        # ----------------- Tensor loading -----------------
        for shard_path in shard_files:
            print(f"Loading shard: {shard_path}")
            with safe_open(shard_path, framework="pt") as f:

                for key in f.keys():
                    layer_idx = None
                    key_parts = key.split('.')
                    if 'layers' in key_parts:
                        li = key_parts.index('layers')
                        if li + 1 < len(key_parts) and key_parts[li + 1].isdigit():
                            layer_idx = int(key_parts[li + 1])
                    if layer_idx is not None and layer_idx not in layers_seen:
                        layers_seen.add(layer_idx)
                        if saveOrload == 'save':
                            self.save_progress += 1
                        else:
                            self.load_progress += 1

                    tensor = f.get_tensor(key)
                    tensor = tensor.to(self.cache_dtype)

                    if tensor.ndim == 1:
                        tid = self._handle_1d_tensor(
                            tensor,
                            key,
                        )
                        print(f"  ✅ 1D tensor '{key}' → {tid}")

                    elif tensor.ndim == 2:
                        tid = self._handle_2d_tensor(
                            tensor,
                            key,
                            saveOrload=saveOrload,
                            split_system=split_system,
                        )
                        print(f"  ✅ 2D tensor '{key}' → {tid}")
                        key_split = key.split('.')
                        #for key_words in key_split:
                            #if key_words == '0':
                                #torch.save(tensor,key)
                        

                    else:
                        cm, shard_shapes = self.wrap_cluster_matrix(
                            tensor,
                            split_dim=0,
                            split_system=split_system,
                            saveOrload=saveOrload,
                            key=key
                        )
                        self.other_list = getattr(self, "other_list", [])
                        self.other_list.append((key, tensor.shape, shard_shapes))
                        print(f"  ⚠️ Higher-rank tensor '{key}'")

        
        self.sort_llama_weights()
        self.sort_norms()

    def _handle_1d_tensor(self, tensor, key):
        """
        Handle 1D tensors (biases, layer norms, etc.) and append them directly to
        the appropriate list in the model handler.
        """
        key_lower = key.lower()
        key_str_split = set(key.split('.'))
        matching_words1 = key_str_split & self.model_key_words1

            # Exact matches
        if matching_words1 == {'input_layernorm'} and torch.is_tensor(tensor):
            self.input_layernorm_list.append([tensor,key])
            return 'input_layernorm'
        elif matching_words1 == {'post_attention_layernorm'} and torch.is_tensor(tensor):
            self.post_attention_layernorm_list.append([tensor,key])
            return 'post_attention_layernorm'

        # Final model norm (e.g., model.norm.weight)
        if (
            'norm' in key_lower
            and 'weight' in key_lower
            and 'input_layernorm' not in key_lower
            and 'post_attention_layernorm' not in key_lower
        ):
            self.final_norm_weight = tensor
            return 'final_norm'

        return 'other_norm'

    def _handle_2d_tensor(self, tensor, key, saveOrload, split_system):
        """Handle 2D weight matrices, wrap in cluster_matrix, and store shapes for later."""
        key_lower = key.lower()

        # ----------------- Embeddings -----------------
        if any(embed_keyword in key_lower for embed_keyword in 
                ['embed', 'embeddings', 'token_embedding', 'word_embedding']):
            if 'weight' in key_lower and 'bias' not in key_lower:
                self.embedding_matrix = tensor
                print(f"Found embedding matrix (kept local, not distributed): {key}")
                return 'embedding'

        # ----------------- Individual projections -----------------
        elif 'q_proj' in key_lower and 'bias' not in key_lower:
            # Save Q shape for later K/V expansion
            self.q_proj_shape = tensor.shape  # [out_dim, in_dim] or equivalent
            cm, shard_shapes = self.wrap_cluster_matrix(
                tensor,
                transpose=True,
                split_dim=1,
                split_system=split_system,
                saveOrload=saveOrload,
                key=key,
                split_matrix=True
            )
            if not hasattr(self, 'q_proj_list'):
                self.q_proj_list = []
            self.q_proj_list.append((key, cm, tensor.shape, shard_shapes))
            return 'q'

        elif 'k_proj' in key_lower and 'bias' not in key_lower:
            # No expansion here; keep raw K projection weights.
            cm, shard_shapes = self.wrap_cluster_matrix(
                tensor,
                transpose=True,
                split_dim=1,
                split_system=split_system,
                saveOrload=saveOrload,
                key=key,
                split_matrix=True
            )
            if not hasattr(self, 'k_proj_list'):
                self.k_proj_list = []
            self.k_proj_list.append((key, cm, tensor.shape, shard_shapes))
            return 'k'

        elif 'v_proj' in key_lower and 'bias' not in key_lower:
            # No expansion here; keep raw V projection weights.
            cm, shard_shapes = self.wrap_cluster_matrix(
                tensor,
                transpose=True,
                split_dim=1,
                split_system=split_system,
                saveOrload=saveOrload,
                key=key,
                split_matrix=True
            )
            if not hasattr(self, 'v_proj_list'):
                self.v_proj_list = []
            self.v_proj_list.append((key, cm, tensor.shape, shard_shapes))
            return 'v'

        elif 'o_proj' in key_lower and 'bias' not in key_lower:
            # --------- EXPAND V TO MATCH Q ---------
            #print('k_proj befor repeat = ',tensor.shape)
            #tensor = self.repeat_kv_tensors(tensor)
            #print('k_proj after repeat tensor.repeat(1, repeat_factor)',tensor.shape)
            cm, shard_shapes = self.wrap_cluster_matrix(
                tensor,
                transpose=True,
                split_dim=1,
                split_system=split_system,
                saveOrload=saveOrload,
                key=key,
                split_matrix=True
            )
            if not hasattr(self, 'o_proj_list'):
                self.o_proj_list = []
            self.o_proj_list.append((key, cm, tensor.shape, shard_shapes))
            return 'o'


        # ----------------- MLP/FFN projections -----------------
        elif 'up_proj' in key_lower or 'w3' in key_lower or 'fc3' in key_lower:
            cm, shard_shapes = self.wrap_cluster_matrix(
                tensor,
                key=key,
                transpose=True,
                split_dim=1,
                saveOrload=saveOrload,
                split_matrix=True
            )
            if not hasattr(self, 'mlp_up_list'):
                self.mlp_up_list = []
            self.mlp_up_list.append((key, cm, tensor.shape, shard_shapes))
            return 'up'

        elif 'down_proj' in key_lower or 'w2' in key_lower or 'fc2' in key_lower:
            cm, shard_shapes = self.wrap_cluster_matrix(
                tensor,
                key=key,
                transpose=True,
                split_dim=1,
                saveOrload=saveOrload,
                split_matrix=True
            )
            if not hasattr(self, 'mlp_down_list'):
                self.mlp_down_list = []
            self.mlp_down_list.append((key, cm, tensor.shape, shard_shapes))
            return 'down'

        elif 'gate_proj' in key_lower or 'w1' in key_lower or 'fc1' in key_lower:
            cm, shard_shapes = self.wrap_cluster_matrix(
                tensor,
                key=key,
                transpose=True,
                split_dim=1,
                saveOrload=saveOrload,
                split_matrix=True
            )
            if not hasattr(self, 'mlp_gate_list'):
                self.mlp_gate_list = []
            self.mlp_gate_list.append((key, cm, tensor.shape, shard_shapes))
            return 'gate'


        # ----------------- LM Head / Output projection -----------------
        elif (('lm_head' in key_lower and 'weight' in key_lower) or
            ('output' in key_lower and 'weight' in key_lower and 'projection' not in key_lower) or
            ('head' in key_lower and 'weight' in key_lower and 'attention' not in key_lower) or
            ('classifier' in key_lower and 'weight' in key_lower) or
            ('language_model' in key_lower and 'head' in key_lower and 'weight' in key_lower)):
            
            self.lm_head = tensor
            print('******************LM_HEAD_FOUND*****************')
            return 'lm_head'
            
        # ----------------- Default -----------------
        cm, shard_shapes = self.wrap_cluster_matrix(tensor,key=key,saveOrload=saveOrload)
        if not hasattr(self, 'other_list'):
            self.other_list = []
        self.other_list.append((key, tensor.shape, shard_shapes))
        return 'other'

    def print_model_info(self):
        """Print comprehensive model information including tensor counts and metadata"""
        print("\n" + "="*60)
        print("MODEL INFORMATION SUMMARY")
        print("="*60)
        print(f"Model Path: {self.model_path}")
        print(f"Model Name: {self.model_name}")
        
        # Print tensor list counts
        print("\n--- TENSOR COUNTS ---")
        print(f"embedding_matrix: {'Found' if self.embedding_matrix is not None else 'Not found'}")
        print(f"q_proj_list: {len(self.q_proj_list)}")
        print(f"k_proj_list: {len(self.k_proj_list)}")
        print(f"v_proj_list: {len(self.v_proj_list)}")
        print(f"qkv_proj_list: {len(self.qkv_proj_list)}")
        print(f"q_proj_bias_list: {len(self.q_proj_bias_list)}")
        print(f"k_proj_bias_list: {len(self.k_proj_bias_list)}")
        print(f"v_proj_bias_list: {len(self.v_proj_bias_list)}")
        print(f"qkv_proj_bias_list: {len(self.qkv_proj_bias_list)}")
        print(f"input_layernorm_list: {len(self.input_layernorm_list)}")
        print(f"post_attention_layernorm_list: {len(self.post_attention_layernorm_list)}")
        print(f"k_layernorm_list: {len(self.k_layernorm_list)}")
        print(f"q_layernorm_list: {len(self.q_layernorm_list)}")
        print(f"operator_norm_list: {len(self.operator_norm_list)}")
        print(f"ffn_norm_list: {len(self.ffn_norm_list)}")
        print(f"mlp_up_list: {len(self.mlp_up_list)}")
        print(f"mlp_down_list: {len(self.mlp_down_list)}")
        print(f"mlp_gate_list: {len(self.mlp_gate_list)}")
        print(f"mlp_up_bias_list: {len(self.mlp_up_bias_list)}")
        print(f"mlp_down_bias_list: {len(self.mlp_down_bias_list)}")
        print(f"mlp_gate_bias_list: {len(self.mlp_gate_bias_list)}")
        print(f"lm_head_list: {len(self.lm_head_list)}")
        print(f"lm_head_bias_list: {len(self.lm_head_bias_list)}")
        
        # Print sample keys
        print("\n--- SAMPLE KEYS (first 3 of each) ---")
        lists_to_show = [
            ("embedding_matrix", [f"Shape: {self.embedding_matrix.shape}"] if self.embedding_matrix is not None else []),
            ("q_proj_list", self.q_proj_list[:3]),
            ("k_proj_list", self.k_proj_list[:3]),
            ("v_proj_list", self.v_proj_list[:3]),
            ("qkv_proj_list", self.qkv_proj_list[:3]),
            ("mlp_up_list", self.mlp_up_list[:3]),
            ("mlp_down_list", self.mlp_down_list[:3]),
            ("mlp_gate_list", self.mlp_gate_list[:3]),
            ("lm_head_list", self.lm_head_list[:3]),
            ("input_layernorm_list", self.input_layernorm_list[:3]),
            ("post_attention_layernorm_list", self.post_attention_layernorm_list[:3]),
            ("operator_norm_list", self.operator_norm_list[:3]),
            ("ffn_norm_list", self.ffn_norm_list[:3]),
        ]
        
        for list_name, lst in lists_to_show:
            if lst:
                print(f"{list_name}: {lst}")
        
        # Print metadata if available
        if hasattr(self, 'num_layers') and self.num_layers is not None:
            print("\n--- MODEL METADATA ---")
            print(f"Number of Layers: {self.num_layers}")
        
        if hasattr(self, 'hidden_size') and self.hidden_size is not None:
            print(f"Hidden Size: {self.hidden_size}")
            
        if hasattr(self, 'num_heads') and self.num_heads is not None:
            print(f"Number of Heads: {self.num_heads}")
            
        if hasattr(self, 'num_kv_heads') and self.num_kv_heads is not None:
            print(f"Number of KV Heads: {self.num_kv_heads}")
            
        if hasattr(self, 'vocab_size') and self.vocab_size is not None:
            print(f"Vocabulary Size: {self.vocab_size}")
            
        if hasattr(self, 'model_type') and self.model_type is not None:
            print(f"Model Type: {self.model_type}")
        
        print("="*60)

    def read_llama_config(self):
        """
        Read Llama model configuration from config.json and set all parameters.
        """
        config_path = os.path.join(self.model_path, "config.json")
        
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found at: {config_path}")
            return False
        
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"📋 Reading Llama configuration from: {config_path}")
            
            # Model identification
            self.model_type = config.get('model_type', 'llama')
            self.architectures = config.get('architectures', ['LlamaForCausalLM'])[0]
            
            # Core model dimensions
            self.hidden_size = config.get('hidden_size', None)
            self.num_hidden_layers = config.get('num_hidden_layers', 0)
            self.num_attention_heads = config.get('num_attention_heads', 0)
            self.num_key_value_heads = config.get('num_key_value_heads', None)
            self.intermediate_size = config.get('intermediate_size', 0)
            self.vocab_size = config.get('vocab_size', 0)
            self.max_position_embeddings = config.get('max_position_embeddings', 0)

            # Keep a consistent layer count for progress tracking
            if not self.num_layers:
                self.num_layers = self.num_hidden_layers
            
            # Model hyperparameters
            self.rms_norm_eps = config.get('rms_norm_eps', 1e-6)
            self.rope_theta = config.get('rope_theta', 10000.0)
            self.rope_scaling = config.get('rope_scaling', None)
            self.hidden_act = config.get('hidden_act', 'silu')
            self.initializer_range = config.get('initializer_range', 0.02)
            self.pretraining_tp = config.get('pretraining_tp', 1)
            
            # Tokenizer settings
            self.bos_token_id = config.get('bos_token_id', 1)
            self.eos_token_id = config.get('eos_token_id', 2)
            self.tie_word_embeddings = config.get('tie_word_embeddings', False)
            
            # Training/inference settings
            self.torch_dtype = config.get('torch_dtype', 'float16')
            self.use_cache = config.get('use_cache', True)
            
            # Derived parameters (not in config but useful)
            if self.hidden_size and self.num_attention_heads:
                self.head_dim = self.hidden_size // self.num_attention_heads
            
            if self.num_key_value_heads and self.head_dim:
                self.kv_dim = self.num_key_value_heads * self.head_dim
            
            # Set defaults for parameters not in config
            self.attention_bias = False  # Llama doesn't have attention bias
            self.attention_dropout = False  # Llama typically doesn't use attention dropout
            self.mlp_bias = False  # Llama doesn't have MLP bias
            
            # Store full config for reference
            self.config = config
            
            # Print summary
            self._print_llama_config_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading config file: {e}")
            return False
    
    def _print_llama_config_summary(self):
        """Print a formatted summary of the Llama configuration"""
        print("\n" + "="*60)
        print("LLAMA MODEL CONFIGURATION")
        print("="*60)
        
        summary_items = [
            ("Model Type", self.model_type),
            ("Architecture", self.architectures),
            ("Hidden Size", self.hidden_size),
            ("Number of Layers", self.num_hidden_layers),
            ("Number of Attention Heads", self.num_attention_heads),
            ("Number of KV Heads", self.num_key_value_heads),
            ("Head Dimension", getattr(self, 'head_dim', 'N/A')),
            ("KV Dimension", getattr(self, 'kv_dim', 'N/A')),
            ("Intermediate Size", self.intermediate_size),
            ("Vocabulary Size", self.vocab_size),
            ("Context Length", self.max_position_embeddings),
            ("RMS Norm Epsilon", self.rms_norm_eps),
            ("RoPE Theta", self.rope_theta),
            ("RoPE Scaling", self.rope_scaling or 'None'),
            ("Hidden Activation", self.hidden_act),
            ("BOS Token ID", self.bos_token_id),
            ("EOS Token ID", self.eos_token_id),
            ("Tie Word Embeddings", self.tie_word_embeddings),
            ("Torch Dtype", self.torch_dtype),
            ("Use Cache", self.use_cache),
            ("Attention Bias", self.attention_bias),
            ("MLP Bias", self.mlp_bias),
        ]
        
        for label, value in summary_items:
            print(f"{label:25} : {value}")
        
        print("="*60)

    def _load_tokenizer(self):
        """
        Force load the correct LLaMA tokenizer from the model folder.
        Must match the embedding matrix (lm_head.weight).
        """
        try:
            model_dir = self.model_path
            # Use the LLaMA tokenizer explicitly
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_dir,
                local_files_only=True
            )
            print(f"✅ Correct LLaMA tokenizer loaded from {model_dir}")
            print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        except Exception as e:
            print(f"❌ Failed to load LLaMA tokenizer: {e}")
            self.tokenizer = None

    # ----------------- TENSOR SORTING & INFO -----------------
    def extract_layer_number(self, tensor_name) -> int:
        """Extract layer number safely from a string (or tuple/list first element)."""
        import re

        # If it's a tuple/list, take first element
        if isinstance(tensor_name, (tuple, list)) and len(tensor_name) > 0:
            tensor_name = tensor_name[0]

        # Must be string now
        tensor_name = str(tensor_name)

        patterns = [
            r'layers\.(\d+)\.',
            r'layer\.(\d+)\.',
            r'blocks\.(\d+)\.',
            r'blk\.(\d+)\.',
            r'\[(\d+)\]',
            r'_(\d+)_',
        ]
        for pattern in patterns:
            match = re.search(pattern, tensor_name)
            if match:
                return int(match.group(1))
        return -1

    def sort_tensors_by_layer(self, tensor_list: list) -> list:
        """Sort a list of tensors (or tuples) by layer number, then name."""
        if not tensor_list:
            return []

        def get_name(t):
            if isinstance(t, (tuple, list)) and len(t) > 0:
                return str(t[0])
            return str(t)

        return [t for _, t in sorted(
            ((self.extract_layer_number(x), x) for x in tensor_list),
            key=lambda x: (x[0], get_name(x[1]))
        )]

    def _print_sorted_summary(self):
        """Print summary of sorted Llama weights with sample layers and tensor names."""
        print("\n📊 Sorted Llama Weights Summary:")
        print("=" * 60)

        summary_groups = {
            "Q Projections": getattr(self, 'q_proj_list', []),
            "K Projections": getattr(self, 'k_proj_list', []),
            "V Projections": getattr(self, 'v_proj_list', []),
            "MLP Up": getattr(self, 'mlp_up_list', []),
            "MLP Down": getattr(self, 'mlp_down_list', []),
            "MLP Gate": getattr(self, 'mlp_gate_list', []),
            "Input LayerNorm": getattr(self, 'input_layernorm_list', []),
            "Post-Attention LayerNorm": getattr(self, 'post_attention_layernorm_list', []),
            "Operator Norm": getattr(self, 'operator_norm_list', []),
            "FFN Norm": getattr(self, 'ffn_norm_list', []),
        }

        for name, lst in summary_groups.items():
            if lst:
                layer_nums = []
                for t in lst[:3]:
                    layer = self.extract_layer_number(t)
                    if layer != -1:
                        layer_nums.append(layer)
                layer_range = f"layers {min(layer_nums)}-{max(layer_nums)}" if layer_nums else "global"
                print(f"  • {name:25} {len(lst):3} tensors ({layer_range})")
                for i, t in enumerate(lst[:3]):
                    if isinstance(t, (tuple, list)):
                        name_str = str(t[0])
                        shape_str = str(t[1]) if len(t) > 1 else ""
                        print(f"      {i+1}. {name_str} {shape_str}")
                    else:
                        print(f"      {i+1}. {t}")
                if len(lst) > 3:
                    print(f"      ... and {len(lst)-3} more")
        print("=" * 60)

    def sort_llama_weights(self):
        """Sort all Llama weight lists by layer number."""
        if getattr(self, 'model_type', None) != 'llama':
            print(f"⚠️ Model type '{getattr(self,'model_type', None)}' is not llama. Skipping sorting.")
            return

        print("🔍 Sorting Llama weights by layer...")

        tensor_lists = {attr: getattr(self, attr, []) for attr in dir(self) if attr.endswith('_list')}

        for list_name, lst in tensor_lists.items():
            if lst:
                sorted_list = self.sort_tensors_by_layer(lst)
                setattr(self, list_name, sorted_list)

        print(f"✅ Sorted tensor lists for Llama model.")
        self._print_sorted_summary()

    def sort_norms(self):
        """
        Sort all 1D norm lists (layer norms, etc.) by layer number.
        Assumes each entry is [tensor, key] or [tensor, key, ...].
        """
        norm_lists = {
            'input_layernorm_list': getattr(self, 'input_layernorm_list', []),
            'post_attention_layernorm_list': getattr(self, 'post_attention_layernorm_list', []),
            'k_layernorm_list': getattr(self, 'k_layernorm_list', []),
            'q_layernorm_list': getattr(self, 'q_layernorm_list', []),
            'operator_norm_list': getattr(self, 'operator_norm_list', []),
            'ffn_norm_list': getattr(self, 'ffn_norm_list', []),
        }

        for name, lst in norm_lists.items():
            if lst:
                # Sort by layer number extracted from the key
                lst.sort(key=lambda x: self.extract_layer_number(x[1]))
                setattr(self, name, lst)

        print("✅ Sorted all norm lists by layer number.")
