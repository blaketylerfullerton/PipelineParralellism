#!/usr/bin/env python3
"""Run a single-machine llama.cpp baseline for the Relay pivot.

This script is intentionally separate from the legacy HuggingFace/ZMQ runtime.
Its job is to land M3.5a cleanly: drive llama.cpp locally, capture timings, and
write a small machine-readable artifact we can compare later against the
two-machine RPC baseline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = ROOT / "logs" / "llama-baselines"
GENERATION_TPS_RE = re.compile(
    r"(?P<label>prompt eval|eval) time\s*=\s*"
    r"(?P<total_ms>[\d.]+)\s*ms\s*/\s*"
    r"(?P<tokens>\d+)\s+tokens?"
    r"\s*\([^)]*?,\s*(?P<tps>[\d.]+)\s*(?:tokens per second|t/s)\)",
    re.IGNORECASE,
)
BRACKET_TPS_RE = re.compile(
    r"\[\s*Prompt:\s*(?P<prompt_tps>[\d.]+)\s*t/s\s*\|\s*Generation:\s*(?P<eval_tps>[\d.]+)\s*t/s\s*\]",
    re.IGNORECASE,
)

QUANT_ALIAS = {
    "q4_k_m": "Q4_K_M",
    "q5_k_m": "Q5_K_M",
    "q8_0": "Q8_0",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise SystemExit("PyYAML is required. Run inside the project venv.") from exc
    return yaml.safe_load(path.read_text())


def _find_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise SystemExit(f"Missing required binary: {name}")
    return binary


def _quant_from_config(config: dict[str, Any]) -> str:
    quant = str(config.get("quantization", {}).get("weight_mode", "")).strip().lower()
    if not quant:
        raise ValueError("config is missing quantization.weight_mode for llama.cpp baseline")
    return QUANT_ALIAS.get(quant, quant.upper())


def _derive_hf_repo(config: dict[str, Any]) -> str:
    llama_cfg = config.get("llama_cpp", {}) or {}
    if llama_cfg.get("hf_repo"):
        repo = str(llama_cfg["hf_repo"]).strip()
        if ":" in repo:
            return repo
        return f"{repo}:{_quant_from_config(config)}"

    model_name = str(config.get("model", {}).get("name", "")).strip()
    if not model_name:
        raise ValueError("config is missing model.name")
    return f"{model_name}-Instruct-GGUF:{_quant_from_config(config)}"


def resolve_model_args(config: dict[str, Any]) -> list[str]:
    llama_cfg = config.get("llama_cpp", {}) or {}
    if llama_cfg.get("model_path"):
        return ["--model", str(llama_cfg["model_path"])]
    if llama_cfg.get("hf_file") and llama_cfg.get("hf_repo"):
        return ["--hf-repo", str(llama_cfg["hf_repo"]), "--hf-file", str(llama_cfg["hf_file"])]
    return ["--hf-repo", _derive_hf_repo(config)]


def parse_timings(text: str) -> dict[str, dict[str, float | int]]:
    timings: dict[str, dict[str, float | int]] = {}
    for match in GENERATION_TPS_RE.finditer(text):
        label = "prompt_eval" if match.group("label").lower().startswith("prompt") else "eval"
        timings[label] = {
            "total_ms": float(match.group("total_ms")),
            "tokens": int(match.group("tokens")),
            "tps": float(match.group("tps")),
        }
    if "eval" not in timings:
        bracket = BRACKET_TPS_RE.search(text)
        if bracket:
            timings["prompt_eval"] = {"total_ms": 0.0, "tokens": 0, "tps": float(bracket.group("prompt_tps"))}
            timings["eval"] = {"total_ms": 0.0, "tokens": 0, "tps": float(bracket.group("eval_tps"))}
    return timings


def build_generate_cmd(config: dict[str, Any], prompt: str, *, n_predict: int, n_gpu_layers: int, threads: int) -> list[str]:
    llama_cfg = config.get("llama_cpp", {}) or {}
    ctx_size = int(llama_cfg.get("ctx_size", 4096))
    temperature = float(config.get("model", {}).get("temperature", 0.7))
    cmd = [
        _find_binary("llama-cli"),
        *resolve_model_args(config),
        "--prompt",
        prompt,
        "--n-predict",
        str(n_predict),
        "--ctx-size",
        str(ctx_size),
        "--threads",
        str(threads),
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--device",
        "none",
        "--fit",
        "off",
        "--simple-io",
        "--no-display-prompt",
        "--single-turn",
        "--perf",
        "--no-warmup",
        "--temp",
        str(temperature),
    ]
    if bool(llama_cfg.get("use_mlock", False)):
        cmd.append("--mlock")
    return cmd


def run_generate(config_path: Path, prompt: str, *, output_dir: Path, n_gpu_layers: int | None, threads: int | None) -> int:
    config = _load_yaml(config_path)
    llama_cfg = config.get("llama_cpp", {}) or {}
    n_predict = int(config.get("model", {}).get("max_new_tokens", 32))
    resolved_threads = threads or int(llama_cfg.get("threads", os.cpu_count() or 1))
    resolved_ngl = n_gpu_layers if n_gpu_layers is not None else int(llama_cfg.get("n_gpu_layers", 0))

    cmd = build_generate_cmd(
        config,
        prompt,
        n_predict=n_predict,
        n_gpu_layers=resolved_ngl,
        threads=resolved_threads,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)

    merged = "\n".join([result.stdout, result.stderr])
    summary = {
        "config": str(config_path),
        "command": cmd,
        "exit_code": result.returncode,
        "prompt": prompt,
        "timings": parse_timings(merged),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"logs: {output_dir}")
    print("cmd:", " ".join(cmd))
    if summary["timings"].get("eval"):
        eval_stats = summary["timings"]["eval"]
        print(
            f"eval: {eval_stats['tokens']} tokens in {eval_stats['total_ms']:.2f} ms "
            f"({eval_stats['tps']:.2f} TPS)"
        )
    else:
        print("eval: no llama.cpp timing block found; inspect stderr.txt")
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single-machine llama.cpp baseline for Relay")
    parser.add_argument("--config", default=str(ROOT / "config.llama.local.yaml"))
    parser.add_argument("--prompt", default="the future of distributed compute is")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_LOG_ROOT / timestamp
    return run_generate(
        Path(args.config).resolve(),
        args.prompt,
        output_dir=output_dir,
        n_gpu_layers=args.n_gpu_layers,
        threads=args.threads,
    )


if __name__ == "__main__":
    raise SystemExit(main())
