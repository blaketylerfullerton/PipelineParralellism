#!/usr/bin/env python3
"""Run local benchmark variants and summarize Stage 0 throughput.

This uses deploy/run_local.py under the hood, so it is intentionally local-only:
no DigitalOcean state changes, no SSH, no tmux. It writes generated configs,
per-stage logs, and machine-readable results under logs/benchmarks/.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


DONE_RE = re.compile(
    r"\[Stage 0\] Done\. (?P<tokens>\d+) tokens, avg (?P<avg_ms>[\d.]+) ms/token "
    r"\((?P<tps>[\d.]+) TPS\)"
)
SPEC_RE = re.compile(
    r"\[Stage 0\] Spec:\s+avg (?P<out_per_step>[\d.]+) output tokens/step\s+"
    r"\(k=(?P<k>\d+), draft_accept=(?P<draft_accept>[\d.]+)%, full=(?P<full>[^)]+)\)"
)
CASCADE_RE = re.compile(r"\[Stage 0\] Cascade:\s+(?P<text>.+)")
PIPELINE_RE = re.compile(r"\[Stage 0\] Pipeline:\s+(?P<text>.+)")
PREFETCH_RE = re.compile(r"\[Stage 0\] Prefetch:\s+(?P<text>.+)")


def _default_python() -> str:
    for candidate in [ROOT / ".venv" / "bin" / "python", ROOT / "venv" / "bin" / "python"]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _load_config(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise SystemExit("PyYAML is required. Run inside the project venv.") from exc
    return yaml.safe_load(path.read_text())


def _write_config(path: Path, config: dict) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise SystemExit("PyYAML is required. Run inside the project venv.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def _parse_stage0_log(path: Path) -> dict:
    result = {
        "tokens": None,
        "avg_ms": None,
        "tps": None,
        "out_per_step": None,
        "draft_accept_pct": None,
        "full_accepts": None,
        "cascade": "",
        "pipeline": "",
        "prefetch": "",
    }
    if not path.exists():
        result["error"] = f"missing {path}"
        return result

    text = path.read_text(errors="replace")
    for line in text.splitlines():
        if match := DONE_RE.search(line):
            result["tokens"] = int(match.group("tokens"))
            result["avg_ms"] = float(match.group("avg_ms"))
            result["tps"] = float(match.group("tps"))
        elif match := SPEC_RE.search(line):
            result["out_per_step"] = float(match.group("out_per_step"))
            result["draft_accept_pct"] = float(match.group("draft_accept"))
            result["full_accepts"] = match.group("full")
        elif match := CASCADE_RE.search(line):
            result["cascade"] = match.group("text")
        elif match := PIPELINE_RE.search(line):
            result["pipeline"] = match.group("text")
        elif match := PREFETCH_RE.search(line):
            result["prefetch"] = match.group("text")
    return result


def _variant_config(base: dict, name: str, *, k: int | None, args: argparse.Namespace, offset: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg.setdefault("network", {})["base_port"] = int(base["network"].get("base_port", 5550)) + offset
    cfg.setdefault("network", {})["workers"] = ["127.0.0.1"] * int(cfg["pipeline"]["num_stages"])
    if args.max_new_tokens is not None:
        cfg.setdefault("model", {})["max_new_tokens"] = args.max_new_tokens

    spec = cfg.setdefault("speculative", {})
    cascade = cfg.setdefault("cascade", {})
    if name == "nospec":
        spec["enabled"] = False
        spec["k"] = 1
        spec["pipeline_prefetch"] = False
        cascade["enabled"] = False
    else:
        spec["enabled"] = True
        spec["k"] = k
        spec["pipeline_prefetch"] = bool(args.prefetch)
        cascade["enabled"] = bool(args.cascade)
    return cfg


def _print_table(results: list[dict]) -> None:
    headers = ["variant", "exit", "tps", "avg_ms", "draft_accept", "out/step", "full", "pipeline"]
    rows = []
    for item in results:
        parsed = item["parsed"]
        rows.append([
            item["variant"],
            str(item["exit_code"]),
            "" if parsed.get("tps") is None else f"{parsed['tps']:.2f}",
            "" if parsed.get("avg_ms") is None else f"{parsed['avg_ms']:.1f}",
            "" if parsed.get("draft_accept_pct") is None else f"{parsed['draft_accept_pct']:.1f}%",
            "" if parsed.get("out_per_step") is None else f"{parsed['out_per_step']:.2f}",
            parsed.get("full_accepts") or "",
            parsed.get("pipeline") or "",
        ])

    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print(" | ".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Relay benchmark matrix")
    parser.add_argument("--base-config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--prompt", default="the future of distributed compute is")
    parser.add_argument("--ks", default="1,2,4,6")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--python", default=_default_python())
    parser.add_argument("--prefetch", action="store_true", help="Enable speculative pipeline prefetch variants")
    parser.add_argument("--cascade", action="store_true", help="Enable cascade in spec variants")
    parser.add_argument("--no-nospec", action="store_true", help="Skip the no-spec baseline")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    base_path = Path(args.base_config).resolve()
    base = _load_config(base_path)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    root_log_dir = Path(args.log_dir).resolve() if args.log_dir else ROOT / "logs" / "benchmarks" / timestamp
    config_dir = root_log_dir / "configs"
    root_log_dir.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, int | None]] = []
    if not args.no_nospec:
        variants.append(("nospec", None))
    for k in [int(value.strip()) for value in args.ks.split(",") if value.strip()]:
        variants.append((f"spec-k{k}", k))

    results = []
    for idx, (name, k) in enumerate(variants):
        cfg = _variant_config(base, name, k=k, args=args, offset=idx * 100)
        config_path = config_dir / f"{name}.yaml"
        variant_log_dir = root_log_dir / name
        _write_config(config_path, cfg)

        cmd = [
            args.python,
            str(ROOT / "deploy" / "run_local.py"),
            "--config",
            str(config_path),
            "--prompt",
            args.prompt,
            "--timeout",
            str(args.timeout),
            "--log-dir",
            str(variant_log_dir),
        ]
        print(f"\n=== {name} ===")
        print(" ".join(cmd))
        proc = subprocess.run(cmd, cwd=ROOT)
        parsed = _parse_stage0_log(variant_log_dir / "stage-0.log")
        results.append({
            "variant": name,
            "config": str(config_path),
            "log_dir": str(variant_log_dir),
            "exit_code": proc.returncode,
            "parsed": parsed,
        })

    (root_log_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults written to {root_log_dir}")
    _print_table(results)
    return 1 if any(item["exit_code"] != 0 for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
