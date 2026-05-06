#!/usr/bin/env python3
"""Small deployment control plane for Relay.

The shell scripts still do the heavy lifting, but this gives the project a
structured way to inspect state, run health checks, collect logs, and call the
existing provision/run/teardown entry points.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "deploy"
LEGACY_STATE = DEPLOY_DIR / ".deploy-state"
JSON_STATE = DEPLOY_DIR / "state.json"
ENV_FILE = DEPLOY_DIR / ".env"

SSH_OPTS = [
    "ssh",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "BatchMode=yes",
    "-o",
    "ServerAliveInterval=5",
    "-o",
    "ServerAliveCountMax=2",
]


def _load_env_file() -> dict[str, str]:
    env = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def _read_legacy_state(path: Path = LEGACY_STATE) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _state_from_legacy(values: dict[str, str]) -> dict:
    if not values:
        return {}
    num_stages = int(values.get("NUM_STAGES", "2"))
    return {
        "schema_version": 1,
        "num_stages": num_stages,
        "region": values.get("REGION"),
        "size": values.get("SIZE"),
        "git_branch": values.get("GIT_BRANCH"),
        "git_commit": values.get("GIT_COMMIT"),
        "stages": [
            {
                "stage": i,
                "droplet_id": values.get(f"DROPLET_ID_{i}") or values.get(f"DROPLET{i}_ID"),
                "public_ip": values.get(f"IP{i}"),
                "wireguard_ip": values.get(f"WG{i}") or values.get(f"WG{i}_IP"),
            }
            for i in range(num_stages)
        ],
    }


def load_state() -> dict:
    if JSON_STATE.exists():
        return json.loads(JSON_STATE.read_text())
    return _state_from_legacy(_read_legacy_state())


def write_state_from_legacy() -> dict:
    state = _state_from_legacy(_read_legacy_state())
    if not state:
        raise SystemExit("No deploy state found to import.")
    JSON_STATE.write_text(json.dumps(state, indent=2) + "\n")
    return state


def _stages(state: dict) -> list[dict]:
    return list(state.get("stages") or [])


def _selected_stages(state: dict, stage_arg: str) -> list[dict]:
    stages = _stages(state)
    if stage_arg == "all":
        return stages
    wanted = int(stage_arg)
    selected = [stage for stage in stages if int(stage["stage"]) == wanted]
    if not selected:
        raise SystemExit(f"Stage {wanted} not found in state.")
    return selected


def _run(cmd: list[str], *, check: bool = False, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
        env=env,
    )


def _print_result(label: str, ok: bool, detail: str = "") -> None:
    status = "ok" if ok else "fail"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def _ssh(ip: str, remote: str) -> subprocess.CompletedProcess:
    return _run([*SSH_OPTS, f"root@{ip}", remote])


def _require_state() -> dict:
    state = load_state()
    if not state:
        raise SystemExit("No deploy state found. Run ./deploy/deploy.sh first.")
    return state


def cmd_status(args: argparse.Namespace) -> int:
    state = load_state()
    if not state:
        print("No deploy state found.")
        return 1

    if args.import_legacy:
        state = write_state_from_legacy()

    print(f"stages: {state.get('num_stages')}")
    print(f"region: {state.get('region') or '?'}")
    print(f"size:   {state.get('size') or '?'}")
    branch = state.get("git_branch") or "?"
    commit = state.get("git_commit") or "?"
    print(f"git:    {branch}@{commit}")
    for stage in _stages(state):
        sid = stage.get("stage")
        print(
            f"stage {sid}: "
            f"droplet={stage.get('droplet_id') or '?'} "
            f"public={stage.get('public_ip') or '?'} "
            f"wg={stage.get('wireguard_ip') or '?'}"
        )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    env_file = _load_env_file()
    state = load_state()
    failures = 0

    for name in ["doctl", "ssh", "tmux", "wg", "python3"]:
        found = shutil.which(name) is not None
        _print_result(f"local command: {name}", found)
        failures += 0 if found else 1

    for key in ["DO_TOKEN", "GITHUB_REPO"]:
        found = bool(os.environ.get(key) or env_file.get(key))
        _print_result(f"env: {key}", found, "from environment or deploy/.env")
        failures += 0 if found else 1

    if not state:
        _print_result("deploy state", False, "run ./deploy/deploy.sh first")
        return failures + 1

    _print_result("deploy state", True, f"{state.get('num_stages')} stages")
    for stage in _selected_stages(state, args.stage):
        sid = stage["stage"]
        ip = stage.get("public_ip")
        if not ip:
            _print_result(f"stage {sid} public IP", False)
            failures += 1
            continue

        print(f"\nstage {sid} ({ip})")
        checks = [
            ("ssh", "true"),
            ("cloud-init", "cloud-init status 2>/dev/null || true"),
            ("repo", "test -d /opt/pipeline && echo present"),
            ("start.sh", "test -x /opt/pipeline/start.sh && echo present"),
            ("wireguard config", "test -f /etc/wireguard/wg0.conf && echo present"),
            ("wireguard service", "systemctl is-active wg-quick@wg0 2>/dev/null || true"),
            ("model log", "tail -1 /var/log/pipeline-models.log 2>/dev/null || echo missing"),
        ]
        for label, remote in checks:
            result = _ssh(ip, remote)
            ok = result.returncode == 0
            output = (result.stdout or result.stderr).strip().splitlines()
            detail = output[-1] if output else ""
            if label in {"cloud-init", "wireguard service", "model log"}:
                ok = result.returncode == 0 and detail not in {"missing", "failed", "inactive"}
            _print_result(f"stage {sid}: {label}", ok, detail)
            failures += 0 if ok else 1

    return 1 if failures else 0


def cmd_logs(args: argparse.Namespace) -> int:
    state = _require_state()
    files = {
        "init": "/var/log/pipeline-init.log",
        "models": "/var/log/pipeline-models.log",
        "pipeline": "/var/log/pipeline.log",
    }
    remote_file = files[args.kind]
    exit_code = 0
    for stage in _selected_stages(state, args.stage):
        sid = stage["stage"]
        ip = stage.get("public_ip")
        print(f"\n===== stage {sid} {args.kind} ({ip}) =====")
        result = _ssh(ip, f"tail -n {args.lines} {remote_file} 2>/dev/null || echo missing")
        if result.returncode != 0:
            exit_code = result.returncode
        print((result.stdout or result.stderr).rstrip())
    return exit_code


def cmd_provision(args: argparse.Namespace) -> int:
    cmd = [str(DEPLOY_DIR / "deploy.sh")]
    env = os.environ.copy()
    if args.stages is not None:
        env["NUM_STAGES"] = str(args.stages)
    if args.config is not None:
        env["CONFIG_FILE"] = str(Path(args.config).resolve())
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode == 0 and LEGACY_STATE.exists():
        write_state_from_legacy()
    return result.returncode


def cmd_run(args: argparse.Namespace) -> int:
    cmd = [str(DEPLOY_DIR / "run.sh"), "--prompt", args.prompt]
    if args.no_wait:
        cmd.append("--no-wait")
    if args.dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd, cwd=ROOT).returncode


def cmd_teardown(args: argparse.Namespace) -> int:
    return subprocess.run([str(DEPLOY_DIR / "teardown.sh")], cwd=ROOT).returncode


def _add_stage_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", default="all", help="'all' or a stage number")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relay deployment control")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show local deploy state")
    status.add_argument("--import-legacy", action="store_true", help="Regenerate deploy/state.json from .deploy-state")
    status.set_defaults(func=cmd_status)

    doctor = sub.add_parser("doctor", help="Check local tools and remote droplet health")
    _add_stage_arg(doctor)
    doctor.set_defaults(func=cmd_doctor)

    logs = sub.add_parser("logs", help="Fetch logs from one or all stages")
    _add_stage_arg(logs)
    logs.add_argument("--kind", choices=["init", "models", "pipeline"], default="models")
    logs.add_argument("--lines", type=int, default=80)
    logs.set_defaults(func=cmd_logs)

    provision = sub.add_parser("provision", help="Run deploy/deploy.sh")
    provision.add_argument("--stages", type=int, default=None)
    provision.add_argument("--config", default=None)
    provision.set_defaults(func=cmd_provision)

    run = sub.add_parser("run", help="Run deploy/run.sh")
    run.add_argument("--prompt", required=True)
    run.add_argument("--no-wait", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)

    teardown = sub.add_parser("teardown", help="Run deploy/teardown.sh")
    teardown.set_defaults(func=cmd_teardown)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
