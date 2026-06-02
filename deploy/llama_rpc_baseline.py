#!/usr/bin/env python3
"""Plan and run the first two-machine llama.cpp RPC baseline.

This script is the bridge from M3.5a (single-machine local llama.cpp smoke) to
M3.5b (two-machine RPC baseline). It does three practical things:

1. Reads the existing DigitalOcean deployment state and picks the remote stage.
2. Generates the exact `rpc-server` and `llama-bench --rpc` commands.
3. Optionally runs the local `llama-bench` command once the remote RPC server is up.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_STATE = ROOT / "deploy" / "state.json"

try:
    from llama_baseline import resolve_model_args
except ModuleNotFoundError:  # pragma: no cover - execution-path specific
    from deploy.llama_baseline import resolve_model_args  # type: ignore


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise SystemExit("PyYAML is required. Run inside the project venv.") from exc
    return yaml.safe_load(path.read_text())


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing deployment state: {path}")
    return json.loads(path.read_text())


def _stage_by_id(state: dict[str, Any], stage_id: int) -> dict[str, Any]:
    for stage in state.get("stages", []):
        if int(stage["stage"]) == stage_id:
            return stage
    raise SystemExit(f"Stage {stage_id} not found in {DEPLOY_STATE}")


def _find_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise SystemExit(f"Missing required binary: {name}")
    return binary


def _resolve_binary(binary: str) -> str | None:
    if "/" in binary:
        return binary if Path(binary).exists() else None
    return shutil.which(binary)


def _llama_bench_supports_rpc(binary: str) -> bool:
    result = subprocess.run([binary, "--help"], text=True, capture_output=True, check=False)
    return "--rpc" in result.stdout or "--rpc" in result.stderr or "-rpc" in result.stdout or "-rpc" in result.stderr


def _rpc_endpoint(config: dict[str, Any], state: dict[str, Any]) -> str:
    rpc_cfg = config.get("rpc", {}) or {}
    remote_stage = int(rpc_cfg.get("remote_stage", 1))
    remote = _stage_by_id(state, remote_stage)
    host = str(rpc_cfg.get("remote_rpc_host") or remote["wireguard_ip"])
    port = int(rpc_cfg.get("port", 50052))
    return f"{host}:{port}"


def build_remote_rpc_command(config: dict[str, Any], state: dict[str, Any]) -> tuple[str, str]:
    rpc_cfg = config.get("rpc", {}) or {}
    remote_stage = int(rpc_cfg.get("remote_stage", 1))
    remote = _stage_by_id(state, remote_stage)
    port = int(rpc_cfg.get("port", 50052))
    bind_host = str(rpc_cfg.get("bind_host", "0.0.0.0"))
    binary = str(rpc_cfg.get("remote_binary", "/opt/llama.cpp-rpc/bin/rpc-server"))
    ssh_user = str(rpc_cfg.get("ssh_user", "root"))

    remote_cmd = []
    if bool(rpc_cfg.get("debug", True)):
        remote_cmd.append("GGML_RPC_DEBUG=1")
    remote_cmd.append(binary)
    remote_cmd.extend(["--host", bind_host, "-p", str(port)])
    if bool(rpc_cfg.get("cache", True)):
        remote_cmd.append("-c")
    if rpc_cfg.get("device"):
        remote_cmd.extend(["--device", str(rpc_cfg["device"])])

    ssh_target = f"{ssh_user}@{remote['public_ip']}"
    return ssh_target, " ".join(remote_cmd)


def build_bench_cmd(config: dict[str, Any], state: dict[str, Any]) -> list[str]:
    bench_cfg = config.get("bench", {}) or {}
    rpc_cfg = config.get("rpc", {}) or {}
    binary = str(bench_cfg.get("binary") or _find_binary("llama-bench"))
    cmd = [
        binary,
        *resolve_model_args(config),
        "--rpc",
        _rpc_endpoint(config, state),
        "--n-prompt",
        str(int(bench_cfg.get("n_prompt", 0))),
        "--n-gen",
        str(int(bench_cfg.get("n_gen", 128))),
        "--repetitions",
        str(int(bench_cfg.get("repetitions", 3))),
        "--threads",
        str(int(bench_cfg.get("threads", 8))),
        "--n-gpu-layers",
        str(int(bench_cfg.get("n_gpu_layers", 0))),
        "--device",
        str(bench_cfg.get("device", "none")),
        "--output",
        str(bench_cfg.get("output", "json")),
    ]
    if bench_cfg.get("tensor_split"):
        cmd.extend(["--tensor-split", str(bench_cfg["tensor_split"])])
    if bench_cfg.get("batch_size"):
        cmd.extend(["--batch-size", str(int(bench_cfg["batch_size"]))])
    if bench_cfg.get("ubatch_size"):
        cmd.extend(["--ubatch-size", str(int(bench_cfg["ubatch_size"]))])
    if bool(rpc_cfg.get("fit_off", True)):
        cmd.extend(["--fit-target", "0"])
    return cmd


def cmd_plan(config_path: Path, state_path: Path) -> int:
    config = _load_yaml(config_path)
    state = _load_state(state_path)
    ssh_target, remote_cmd = build_remote_rpc_command(config, state)
    bench_cmd = build_bench_cmd(config, state)
    endpoint = _rpc_endpoint(config, state)

    print(f"state: {state_path}")
    print(f"config: {config_path}")
    print(f"rpc endpoint: {endpoint}")
    print("")
    print("Remote Stage")
    print(f"  ssh {ssh_target}")
    print(f"  {remote_cmd}")
    print("")
    print("Main Host")
    print(f"  {' '.join(bench_cmd)}")
    return 0


def cmd_doctor(config_path: Path, state_path: Path, ssh_check: bool) -> int:
    config = _load_yaml(config_path)
    state = _load_state(state_path)
    bench_cmd = build_bench_cmd(config, state)
    bench_bin = bench_cmd[0]
    resolved_bench = _resolve_binary(bench_bin)
    ok = True

    print(f"[{'ok' if resolved_bench else 'fail'}] local llama-bench: {resolved_bench or bench_bin}")
    ok &= resolved_bench is not None
    supports_rpc = bool(resolved_bench) and _llama_bench_supports_rpc(resolved_bench)
    print(f"[{'ok' if supports_rpc else 'fail'}] llama-bench --rpc support")
    ok &= supports_rpc

    ssh_target, remote_cmd = build_remote_rpc_command(config, state)
    print(f"[ok] remote rpc command template: {remote_cmd}")

    if ssh_check:
        import shlex

        rpc_cfg = config.get("rpc", {}) or {}
        remote_binary = str(rpc_cfg.get("remote_binary", "/opt/llama.cpp-rpc/bin/rpc-server"))
        ssh = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=8",
            ssh_target,
            f"test -x {shlex.quote(remote_binary)} && echo {shlex.quote(remote_binary)}",
        ]
        result = subprocess.run(ssh, text=True, capture_output=True, check=False)
        remote_ok = result.returncode == 0
        detail = (result.stdout or result.stderr).strip() or remote_binary
        print(f"[{'ok' if remote_ok else 'fail'}] remote rpc-server binary: {detail}")
        ok &= remote_ok

    print(f"[ok] rpc endpoint: {_rpc_endpoint(config, state)}")
    return 0 if ok else 1


def cmd_bench(config_path: Path, state_path: Path) -> int:
    config = _load_yaml(config_path)
    state = _load_state(state_path)
    cmd = build_bench_cmd(config, state)
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Relay two-machine llama.cpp RPC baseline helper")
    parser.add_argument("--config", default=str(ROOT / "config.llama.rpc.yaml"))
    parser.add_argument("--state", default=str(DEPLOY_STATE))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="Print the remote rpc-server and local llama-bench commands")
    doctor = sub.add_parser("doctor", help="Check local prerequisites and optionally SSH-check the remote host")
    doctor.add_argument("--ssh-check", action="store_true")
    sub.add_parser("bench", help="Run the local llama-bench command against the configured RPC endpoint")

    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    state_path = Path(args.state).resolve()

    if args.command == "plan":
        return cmd_plan(config_path, state_path)
    if args.command == "doctor":
        return cmd_doctor(config_path, state_path, ssh_check=bool(args.ssh_check))
    if args.command == "bench":
        return cmd_bench(config_path, state_path)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
