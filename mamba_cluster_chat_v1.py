from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import os
import re
import sys
from pathlib import Path
from typing import Iterator, Optional, TextIO

import torch
from transformers import AutoTokenizer

from mamba_cluster_transformer import ClusterRuntime, MambaCodestral7BCluster, _import_cluster_matrix_v1, _repo_root


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


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

    # Avoid collisions if multiple runs share the same first 3 words and date.
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


def _stream_greedy_tokens(
    *,
    model: MambaCodestral7BCluster,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    log_fp: Optional[TextIO],
    log_transformer_output: bool,
) -> Iterator[str]:
    def _decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _diff(prev: str, cur: str) -> str:
        if cur.startswith(prev):
            return cur[len(prev) :]
        # Fallback: find common prefix length (handles rare tokenizer edge cases).
        n = 0
        limit = min(len(prev), len(cur))
        while n < limit and prev[n] == cur[n]:
            n += 1
        return cur[n:]

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(torch.long)
    if input_ids.shape[0] != 1:
        raise ValueError("This runner expects batch_size=1 for now")
    batch = int(input_ids.shape[0])
    prompt_ids = input_ids[0].tolist()
    generated_ids: list[int] = []

    with _redirect_output(log_transformer_output, log_fp):
        conv_states, ssm_states = model.allocate_cache(batch)

        # Prefill (updates states for all prompt tokens, keep only last logits)
        for t in range(int(input_ids.shape[1])):
            logits = model.step_token(input_ids[:, t], conv_states, ssm_states)

        eos_id = tokenizer.eos_token_id
        prev_text = _decode(prompt_ids)
        for _ in range(int(max_new_tokens)):
            next_id = int(torch.argmax(logits[0]).item())
            if eos_id is not None and next_id == int(eos_id):
                break
            generated_ids.append(next_id)
            cur_text = _decode(prompt_ids + generated_ids)
            piece = _diff(prev_text, cur_text)
            prev_text = cur_text
            if piece:
                yield piece
            next_tok = torch.tensor([next_id], dtype=torch.long)
            logits = model.step_token(next_tok, conv_states, ssm_states)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=os.path.join(_repo_root(), "llm_models", "Mamba-Codestral-7B-v0.1"))
    p.add_argument("--max-new-tokens", type=int, default=16)
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
    p.add_argument(
        "--weight-cache-mode",
        default=os.environ.get("CLUSTER_WEIGHT_CACHE_MODE", "save"),
        choices=["save", "load"],
        help="Use 'save' once to (re)generate and distribute shard files to all nodes; 'load' assumes shards exist on all nodes.",
    )
    p.add_argument(
        "--precache",
        action="store_true",
        help="Pre-generate/distribute all layer weight shards before running the prompt (recommended for multi-node).",
    )
    p.add_argument("--precache-only", action="store_true", help="Exit after precache completes.")
    p.add_argument("--precache-start-layer", type=int, default=0)
    p.add_argument("--precache-end-layer", type=int, default=-1, help="Inclusive. -1 means last layer.")
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
    cluster_dir = os.path.join(_repo_root(), "cluster_matrix")
    os.environ.setdefault("LOCAL_DISK_FOLDER", os.path.join(cluster_dir, "matrix_shards") + os.sep)
    os.environ.setdefault("REMOTE_DISK_FOLDER", "matrix_shards" + os.sep)

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
        node_percentages = [1.0 / nslots] * nslots

    cm = _import_cluster_matrix_v1()

    # Keep cluster initialization output visible (high-level startup info).
    cluster_zmq_object = cm.cluster_zmq(node_ips)
    rt = ClusterRuntime(
        cluster_zmq_object=cluster_zmq_object,
        node_ips=node_ips,
        cpu_gpu_select=cpu_gpu_select,
        node_percentages=node_percentages,
        backend_select=backend_select,
        weight_cache_mode=str(args.weight_cache_mode),
    )

    if bool(args.precache_only):
        # Allow running `--precache --precache-only` without prompting for chat input.
        log_fp: Optional[TextIO] = None
        log_path: Optional[Path] = None
        try:
            if bool(args.log_transformer_output):
                Path(args.log_dir).mkdir(parents=True, exist_ok=True)
                stamp = _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
                log_path = Path(args.log_dir) / f"precache-{stamp}_log.txt"
                log_fp = open(log_path, "w", encoding="utf-8")

            with _redirect_output(bool(args.log_transformer_output), log_fp):
                model = MambaCodestral7BCluster(args.model_dir, rt)
                if not bool(args.precache):
                    raise ValueError("--precache-only requires --precache")

                start = int(args.precache_start_layer)
                end = int(args.precache_end_layer)
                if end < 0:
                    end = int(model.num_layers) - 1
                if start < 0 or start >= model.num_layers:
                    raise ValueError(f"--precache-start-layer out of range: {start}")
                if end < start or end >= model.num_layers:
                    raise ValueError(f"--precache-end-layer out of range: {end}")
                if str(rt.weight_cache_mode).lower() != "save":
                    raise ValueError("--precache requires --weight-cache-mode save")
                for layer_idx in range(start, end + 1):
                    model._matmul.get_weight(model._in_proj_paths[layer_idx], split_dim=0)
                    model._matmul.get_weight(model._out_proj_paths[layer_idx], split_dim=0)
                model._matmul.get_weight(model._lm_head_path, split_dim=0)
        finally:
            if log_fp is not None:
                log_fp.close()
        if log_path is not None:
            print(f"Saved log: {log_path}", file=sys.__stdout__)
        sys.exit(0)

    log_transformer_output = bool(args.log_transformer_output)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = MambaCodestral7BCluster(args.model_dir, rt)

    # Optional: pre-cache weights once before entering the chat loop.
    if bool(args.precache):
        with _redirect_output(log_transformer_output, None):
            if str(rt.weight_cache_mode).lower() != "save":
                raise ValueError("--precache requires --weight-cache-mode save")
            start = int(args.precache_start_layer)
            end = int(args.precache_end_layer)
            if end < 0:
                end = int(model.num_layers) - 1
            if start < 0 or start >= model.num_layers:
                raise ValueError(f"--precache-start-layer out of range: {start}")
            if end < start or end >= model.num_layers:
                raise ValueError(f"--precache-end-layer out of range: {end}")
            for layer_idx in range(start, end + 1):
                model._matmul.get_weight(model._in_proj_paths[layer_idx], split_dim=0)
                model._matmul.get_weight(model._out_proj_paths[layer_idx], split_dim=0)
            model._matmul.get_weight(model._lm_head_path, split_dim=0)

    try:
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

                    print("output: ", end="", file=sys.__stdout__, flush=True)
                    first_output_written = False
                    for tok in _stream_greedy_tokens(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=int(args.max_new_tokens),
                        log_fp=log_fp,
                        log_transformer_output=log_transformer_output,
                    ):
                        if not first_output_written:
                            # Avoid leading whitespace in the transcript/terminal (common with BPE/SentencePiece).
                            if tok.startswith(" "):
                                tok = tok[1:]
                            if tok:
                                first_output_written = True
                        chat_fp.write(tok)
                        chat_fp.flush()
                        print(tok, end="", file=sys.__stdout__, flush=True)

                    chat_fp.write("\n")
                    chat_fp.flush()
                    print("\n", file=sys.__stdout__, flush=True)
                except KeyboardInterrupt:
                    print("\nExiting.", file=sys.__stdout__)
                    break
                finally:
                    if log_fp is not None:
                        log_fp.close()

            print(f"Saved chat: {transcript_path}", file=sys.__stdout__)
            if log_transformer_output:
                print(f"Saved log: {log_path}", file=sys.__stdout__)
    finally:
        # Avoid shutdown hangs by explicitly closing ZMQ sockets/context.
        try:
            if hasattr(cluster_zmq_object, "cleanup"):
                cluster_zmq_object.cleanup()
        except Exception:
            pass
