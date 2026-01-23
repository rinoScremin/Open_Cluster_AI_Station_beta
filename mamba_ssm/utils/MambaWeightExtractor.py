import os
import re
import gc
import torch
from safetensors.torch import safe_open
from transformers import AutoConfig



class MambaWeightExtractor:
    def __init__(self, model_path: str, output_dir: str = None):
        print("\n🚀 Initializing MambaWeightExtractor")

        self.model_path = model_path

        # Use custom output dir if provided, else default to model_path/model_matrices
        if output_dir is None:
            self.output_dir = os.path.join(model_path, "model_matrices")
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✅ Matrices will be saved to: {self.output_dir}")

        # Load config
        self.config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )

        # --------------------------------------------------
        # Discover safetensors shards
        # --------------------------------------------------
        self.weight_files = []
        for root, _, files in os.walk(model_path):
            for f in files:
                if f.endswith(".safetensors") and not f.endswith(".index.json"):
                    self.weight_files.append(os.path.join(root, f))

        # --------------------------------------------------
        # Clear diagnostic if only index exists
        # --------------------------------------------------
        index_file = os.path.join(model_path, "model.safetensors.index.json")

        if not self.weight_files:
            if os.path.exists(index_file):
                raise RuntimeError(
                    "Found model.safetensors.index.json but NO safetensor shards.\n"
                    "This means the model weights were NOT downloaded.\n\n"
                    "Fix:\n"
                    "huggingface-cli download mistralai/Mamba-Codestral-7B-v0.1 "
                    "--local-dir <this-folder> --local-dir-use-symlinks False"
                )
            else:
                raise RuntimeError(
                    f"No safetensors files found under {model_path}"
                )

        print(f"✅ Found {len(self.weight_files)} safetensors shards")

    # ----------------------------------------------------
    def discover_keys(self):
        """Discover all tensor keys across shards."""
        keys = set()
        for wf in self.weight_files:
            with safe_open(wf, framework="pt", device="cpu") as f:
                keys.update(f.keys())
        print(f"🔍 Discovered {len(keys)} total tensors")
        return sorted(keys)

    # ----------------------------------------------------
    def build_mamba_key_map(self, keys):
        """Map original tensor keys to local filenames."""
        mapping = {}

        for key in keys:
            out = None

            # HF Mamba2 models use `backbone.embeddings.weight` (plural).
            if key.endswith("embeddings.weight") or key.endswith("embedding.weight"):
                out = "embed_tokens_weight.pt"
            elif "norm_f.weight" in key:
                out = "model_norm_weight.pt"
            elif key.endswith("lm_head.weight"):
                out = "lm_head_weight.pt"

            m = re.search(r"layers\.(\d+)\.mixer\.(.+)", key)
            if m:
                layer = int(m.group(1))
                comp = m.group(2)
                out = f"layer_{layer}_{comp.replace('.', '_')}.pt"

            # Block RMSNorm sits outside the mixer: `backbone.layers.{i}.norm.weight`
            m = re.search(r"layers\.(\d+)\.norm\.(weight|bias)$", key)
            if m:
                layer = int(m.group(1))
                param = m.group(2)
                out = f"layer_{layer}_block_norm_{param}.pt"

            if out:
                mapping[key] = out

        print(f"🗺️ Mapped {len(mapping)} tensors")
        return mapping

    # ----------------------------------------------------
    def extract(self, dtype=torch.float16, overwrite=False):
        """Extract all tensors from safetensors shards and save to disk."""
        keys = self.discover_keys()
        key_map = self.build_mamba_key_map(keys)

        saved = 0

        for wf in self.weight_files:
            print(f"\n📦 Processing {os.path.basename(wf)}")

            with safe_open(wf, framework="pt", device="cpu") as f:
                f_keys = f.keys()  # Explicit keys list for safe lookup
                for k, out_name in key_map.items():
                    if k not in f_keys:
                        continue  # Skip missing tensors

                    out_path = os.path.join(self.output_dir, out_name)
                    if os.path.exists(out_path) and not overwrite:
                        continue

                    t = f.get_tensor(k).to(dtype).contiguous()
                    torch.save(t, out_path)
                    saved += 1

                    if saved <= 10 or saved % 25 == 0:
                        print(f"  ✓ {out_name} {tuple(t.shape)}")

                    del t
                    if saved % 25 == 0:
                        gc.collect()

        print(f"\n✅ Extraction complete — saved {saved} tensors")
        return saved

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    MODEL_PATH = (
        "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/llm_models/Mamba-Codestral-7B-v0.1"
    )

    # Specify custom folder for matrices
    OUTPUT_DIR = "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/model_matrices"

    extractor = MambaWeightExtractor(MODEL_PATH, output_dir=OUTPUT_DIR)
    extractor.extract(
        dtype=torch.float16,
        overwrite=False
    )
