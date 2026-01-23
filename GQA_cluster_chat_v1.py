from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Iterator, Optional, TextIO

from GQA_cluster_transformer_v1 import _repo_root, cluster_llm_transformer

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_name(s: str, *, max_len: int = 48) -> str:
    s = _SAFE_NAME_RE.sub("_", str(s)).strip("_")
    if not s:
        s = "model"
    if len(s) > int(max_len):
        s = s[: int(max_len)]
    return s


def _parse_csv_list(s: str) -> list[str]:
    parts: list[str] = []
    for raw in str(s).replace(" ", ",").split(","):
        v = raw.strip()
        if v:
            parts.append(v)
    return parts


def _parse_bool_list(s: str) -> list[bool]:
    out: list[bool] = []
    for raw in _parse_csv_list(s):
        v = raw.strip().lower()
        if v in ("1", "true", "t", "yes", "y", "gpu"):
            out.append(True)
        elif v in ("0", "false", "f", "no", "n", "cpu"):
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean value in list: {raw!r}")
    return out


def _prompt_slug(prompt: str, *, max_words: int = 3) -> str:
    words = _WORD_RE.findall(str(prompt).lower())
    words = words[: int(max_words)]
    if not words:
        return "chat"
    return "_".join(words)


def _allocate_chat_paths(prompt: str, *, transcript_dir: Path, log_dir: Path) -> tuple[Path, Path]:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    date = _dt.date.today().isoformat()
    base = f"{_prompt_slug(prompt)}-{date}"

    def _paths(name: str) -> tuple[Path, Path]:
        return transcript_dir / f"{name}.txt", log_dir / f"{name}_log.txt"

    transcript_path, log_path = _paths(base)
    if not transcript_path.exists() and not log_path.exists():
        return transcript_path, log_path

    for i in range(1, 10_000):
        transcript_path, log_path = _paths(f"{base}-{i}")
        if not transcript_path.exists() and not log_path.exists():
            return transcript_path, log_path

    raise RuntimeError("Unable to allocate unique chat/log filenames")


@contextlib.contextmanager
def _redirect_output(enabled: bool, log_fp: Optional[TextIO]):
    if not enabled or log_fp is None:
        yield
        return
    with contextlib.redirect_stdout(log_fp), contextlib.redirect_stderr(log_fp):
        yield


def _stream_text(
    *,
    model: cluster_llm_transformer,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool,
    temperature: float,
    top_k: int,
    micro_batch_size: int,
    log_fp: Optional[TextIO],
    log_transformer_output: bool,
) -> Iterator[str]:
    with _redirect_output(log_transformer_output, log_fp):
        yield from model.generate_text_stream(
            prompt,
            max_new_tokens=int(max_new_tokens),
            use_chat_template=bool(use_chat_template),
            temperature=float(temperature),
            top_k=int(top_k),
            micro_batch_size=int(micro_batch_size),
            skip_special_tokens=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Local model folder or HF id.")
    p.add_argument(
        "--model-matrices-dir",
        default=None,
        help="Optional override for extracted `.pt` weights dir (default: <model-dir>/model_matrices).",
    )
    p.add_argument(
        "--weight-cache-mode",
        default=os.environ.get("CLUSTER_WEIGHT_CACHE_MODE", "load"),
        choices=["save", "load"],
        help="Use 'save' to (re)generate and distribute shard files; 'load' assumes shards exist on all nodes.",
    )
    p.add_argument("--precache", action="store_true", help="Pre-generate/distribute all layer weight shards before chat.")
    p.add_argument("--precache-only", action="store_true", help="Exit after precache completes.")
    p.add_argument("--precache-start-layer", type=int, default=0)
    p.add_argument("--precache-end-layer", type=int, default=-1, help="Inclusive. -1 means last layer.")
    p.add_argument(
        "--auto-extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-generate `<model-matrices-dir>/*.pt` files if missing (from safetensors).",
    )
    p.add_argument(
        "--allow-full-model-load",
        action="store_true",
        default=False,
        help="Allow full HF model load fallback if no safetensors exist (may require lots of RAM).",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--use-chat-template", action="store_true")

    p.add_argument("--backend", default=os.environ.get("CLUSTER_BACKEND", "llama"), choices=["torch", "llama", "opencl"])
    p.add_argument("--use-gpu", action="store_true", default=False)
    p.add_argument("--node-ips", default=os.environ.get("CLUSTER_NODE_IPS", os.environ.get("HEAD_NODE_IP", "192.168.2.100")))
    p.add_argument(
        "--backend-select-list",
        default=os.environ.get("CLUSTER_BACKEND_SELECT_LIST", ""),
        help="Comma-separated backend per node slot (overrides --backend). Example: llama,llama,torch,torch",
    )
    p.add_argument(
        "--cpu-gpu-select-list",
        default=os.environ.get("CLUSTER_CPU_GPU_SELECT_LIST", ""),
        help="Comma-separated bool/int per node slot (overrides --use-gpu). Example: 1,1,0,0",
    )
    p.add_argument(
        "--node-percentages",
        default=os.environ.get("CLUSTER_NODE_PERCENTAGES", ""),
        help="Comma-separated percentages per node slot (must match --node-ips length). Default: uniform.",
    )

    p.add_argument("--transcript-dir", default=os.path.join(_repo_root(), "AI_chats"))
    p.add_argument("--log-dir", default=os.path.join(_repo_root(), "output_logs"))
    p.add_argument(
        "--log-transformer-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redirect noisy transformer/cluster stdout/stderr to a log file in --log-dir.",
    )
    args = p.parse_args()

    # Make cluster_matrix_v1's on-disk shard paths consistent when running from any folder.
    # IMPORTANT: use a per-model subfolder so different models can share the same project without shard name collisions.
    cluster_dir = os.path.join(_repo_root(), "cluster_matrix")
    model_dir_abs = os.path.abspath(str(args.model_dir))
    model_slug = _safe_name(os.path.basename(model_dir_abs) or model_dir_abs, max_len=32)
    model_hash = hashlib.sha1(model_dir_abs.encode("utf-8")).hexdigest()[:8]
    shard_subdir = f"{model_slug}_{model_hash}"

    # Override (not setdefault) because users may source scripts that set these env vars already.
    os.environ["LOCAL_DISK_FOLDER"] = os.path.join(cluster_dir, "matrix_shards", shard_subdir) + os.sep
    os.environ["REMOTE_DISK_FOLDER"] = f"matrix_shards/{shard_subdir}/"
    os.environ["LOCAL_PROJECT_DIR"] = cluster_dir + os.sep

    node_ips = _parse_csv_list(args.node_ips)
    if not node_ips:
        raise ValueError("--node-ips cannot be empty")
    nslots = len(node_ips)

    if str(args.cpu_gpu_select_list).strip():
        cpu_gpu_select = _parse_bool_list(args.cpu_gpu_select_list)
        if len(cpu_gpu_select) != nslots:
            raise ValueError(f"--cpu-gpu-select-list length {len(cpu_gpu_select)} must match --node-ips length {nslots}")
    else:
        cpu_gpu_select = [bool(args.use_gpu)] * nslots

    if str(args.backend_select_list).strip():
        backend_select = _parse_csv_list(args.backend_select_list)
        if len(backend_select) != nslots:
            raise ValueError(f"--backend-select-list length {len(backend_select)} must match --node-ips length {nslots}")
    else:
        backend_select = [str(args.backend)] * nslots

    if str(args.node_percentages).strip():
        node_percentages = [float(x) for x in _parse_csv_list(args.node_percentages)]
        if len(node_percentages) != nslots:
            raise ValueError(f"--node-percentages length {len(node_percentages)} must match --node-ips length {nslots}")
        s = float(sum(node_percentages))
        if s <= 0:
            raise ValueError("--node-percentages must sum to > 0")
        node_percentages = [float(x) / s for x in node_percentages]
    else:
        node_percentages = [1.0 / float(nslots)] * nslots

    log_transformer_output = bool(args.log_transformer_output)
    weight_cache_mode = str(args.weight_cache_mode).lower()

    def _required_weight_paths(model: cluster_llm_transformer) -> list[str]:
        base = model.model_matrix_fold_dir
        return [
            os.path.join(base, "embed_tokens_weight.pt"),
            os.path.join(base, "lm_head_weight.pt"),
            os.path.join(base, "model_norm_weight.pt"),
            os.path.join(base, "layers_0_input_layernorm_weight.pt"),
            os.path.join(base, "layers_0_self_attn_q_proj_weight.pt"),
            os.path.join(base, "layers_0_self_attn_k_proj_weight.pt"),
            os.path.join(base, "layers_0_self_attn_v_proj_weight.pt"),
            os.path.join(base, "layers_0_self_attn_o_proj_weight.pt"),
        ]

    def _ensure_extracted_weights(model: cluster_llm_transformer, *, log_fp: Optional[TextIO]) -> None:
        missing = [p for p in _required_weight_paths(model) if not os.path.exists(p)]
        if not missing:
            return
        if not bool(args.auto_extract):
            raise FileNotFoundError(
                "Missing extracted weights in model_matrices:\n"
                + "\n".join(f"- {p}" for p in missing[:20])
                + ("\n..." if len(missing) > 20 else "")
                + "\nFix: run with --auto-extract (default) or generate model_matrices first."
            )
        with _redirect_output(log_transformer_output, log_fp):
            model.save_all_model_layers(
                start_layer=0,
                end_layer=int(model.num_layers) - 1,
                dtype="float16",
                overwrite=False,
                prefer_safetensors=True,
                allow_full_model_load=bool(args.allow_full_model_load),
            )
        # Verify extraction actually produced the required baseline files.
        missing_after = [p for p in _required_weight_paths(model) if not os.path.exists(p)]
        if missing_after:
            raise FileNotFoundError(
                "Auto-extract completed but required weights are still missing:\n"
                + "\n".join(f"- {p}" for p in missing_after[:20])
                + ("\n..." if len(missing_after) > 20 else "")
                + "\nCheck the log for extraction details."
            )

    model: Optional[cluster_llm_transformer] = None
    try:
        # Print cluster init once at startup (do not redirect).
        model = cluster_llm_transformer(
            args.model_dir,
            node_ips,
            node_percentages,
            cpu_gpu_select,
            backend_select,
            model_matrices_dir=args.model_matrices_dir,
            weight_cache_mode=weight_cache_mode,
        )

        # Optional: precache (and/or extract) once before entering the chat loop.
        if bool(args.precache) or bool(args.precache_only):
            if not bool(args.precache):
                raise ValueError("--precache-only requires --precache")
            if weight_cache_mode != "save":
                raise ValueError("--precache requires --weight-cache-mode save")

            log_fp: Optional[TextIO] = None
            log_path: Optional[Path] = None
            try:
                if log_transformer_output:
                    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
                    stamp = _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
                    log_path = Path(args.log_dir) / f"precache-{stamp}_log.txt"
                    log_fp = open(log_path, "w", encoding="utf-8")

                _ensure_extracted_weights(model, log_fp=log_fp)

                start = int(args.precache_start_layer)
                end_inclusive = int(args.precache_end_layer)
                if end_inclusive < 0:
                    end_inclusive = int(model.num_layers) - 1
                if start < 0 or start >= int(model.num_layers):
                    raise ValueError(f"--precache-start-layer out of range: {start}")
                if end_inclusive < start or end_inclusive >= int(model.num_layers):
                    raise ValueError(f"--precache-end-layer out of range: {end_inclusive}")

                with _redirect_output(log_transformer_output, log_fp):
                    # Distribute shard caches for all weights used by the transformer layers.
                    model.save_distribute_model_matrices(
                        start_layer=start,
                        end_layer=end_inclusive + 1,  # function uses end-exclusive
                        include_embed_tokens=False,
                        include_lm_head=False,
                        include_final_norm=False,
                        transpose_for_runtime=True,
                        keep_ram_copies=False,
                    )
            except Exception as e:
                # Surface a concise error even if logs are redirected.
                msg = str(e).strip() or e.__class__.__name__
                print(f"[precache error] {msg}", file=sys.__stdout__)
                raise
            finally:
                if log_fp is not None:
                    log_fp.close()

            if log_path is not None:
                print(f"Saved log: {log_path}", file=sys.__stdout__)

            if bool(args.precache_only):
                raise SystemExit(0)

        while True:
            try:
                prompt = input("Enter prompt: ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nExiting.", file=sys.__stdout__)
                break

            if not prompt:
                break

            transcript_path, log_path = _allocate_chat_paths(
                prompt, transcript_dir=Path(args.transcript_dir), log_dir=Path(args.log_dir)
            )

            with open(transcript_path, "w", encoding="utf-8") as chat_fp:
                chat_fp.write(f"input: {prompt}\n\noutput: ")
                chat_fp.flush()

                log_fp: Optional[TextIO] = None
                try:
                    if log_transformer_output:
                        log_fp = open(log_path, "w", encoding="utf-8")

                    # Ensure model_matrices exist before generation (common first-run failure).
                    _ensure_extracted_weights(model, log_fp=log_fp)

                    print("output: ", end="", file=sys.__stdout__, flush=True)
                    first_output_written = False
                    for piece in _stream_text(
                        model=model,
                        prompt=prompt,
                        max_new_tokens=int(args.max_new_tokens),
                        use_chat_template=bool(args.use_chat_template),
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        micro_batch_size=int(args.micro_batch_size),
                        log_fp=log_fp,
                        log_transformer_output=log_transformer_output,
                    ):
                        if not first_output_written:
                            if piece.startswith(" "):
                                piece = piece[1:]
                            if piece:
                                first_output_written = True
                        chat_fp.write(piece)
                        chat_fp.flush()
                        print(piece, end="", file=sys.__stdout__, flush=True)

                    chat_fp.write("\n")
                    chat_fp.flush()
                    print("\n", file=sys.__stdout__, flush=True)
                except KeyboardInterrupt:
                    print("\nExiting.", file=sys.__stdout__)
                    break
                except Exception as e:
                    # Keep REPL alive; error details go to log when enabled.
                    msg = str(e).strip() or e.__class__.__name__
                    print(f"\n[error] {msg}\n", file=sys.__stdout__, flush=True)
                finally:
                    if log_fp is not None:
                        log_fp.close()

            print(f"Saved chat: {transcript_path}", file=sys.__stdout__)
            if log_transformer_output:
                print(f"Saved log: {log_path}", file=sys.__stdout__)
    finally:
        try:
            if model is not None and hasattr(model, "cluster_zmq_object") and hasattr(model.cluster_zmq_object, "cleanup"):
                model.cluster_zmq_object.cleanup()
        except Exception:
            pass
