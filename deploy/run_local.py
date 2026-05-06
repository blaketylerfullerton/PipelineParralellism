#!/usr/bin/env python3
"""Run a local multi-stage pipeline as subprocesses.

This is the non-tmux inner loop: it starts stages on localhost, writes one log
file per stage, waits for Stage 0 to finish, then collects the remaining stages.
Use config.smoke.yaml for fast startup and config.yaml when you want the real
model path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _default_python() -> str:
    for candidate in [ROOT / ".venv" / "bin" / "python", ROOT / "venv" / "bin" / "python"]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _read_num_stages(config_path: Path) -> int:
    text = config_path.read_text()
    match = re.search(r"(?m)^\s*num_stages\s*:\s*(\d+)", text)
    if not match:
        raise SystemExit(f"Could not find pipeline.num_stages in {config_path}")
    return int(match.group(1))


def _open_log(log_dir: Path, stage: int):
    log_dir.mkdir(parents=True, exist_ok=True)
    return (log_dir / f"stage-{stage}.log").open("w")


def _start_stage(
    stage: int,
    stages: int,
    config: Path,
    python_bin: str,
    prompt: str | None,
    log_dir: Path,
) -> tuple[subprocess.Popen, object]:
    ips = ["127.0.0.1"] * stages
    cmd = [
        python_bin,
        str(ROOT / "src" / "launch.py"),
        "--stage",
        str(stage),
        "--stages",
        str(stages),
        "--peer-ip",
        *ips,
        "--config",
        str(config),
    ]
    if stage == 0 and prompt is not None:
        cmd.extend(["--prompt", prompt])

    log = _open_log(log_dir, stage)
    print(f"stage {stage}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, log


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Relay stages without tmux")
    parser.add_argument("--config", default=str(ROOT / "config.smoke.yaml"))
    parser.add_argument("--stages", type=int, default=None)
    parser.add_argument("--prompt", default="the future of distributed compute is")
    parser.add_argument("--python", default=_default_python())
    parser.add_argument("--start-delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Defaults to logs/local-runs/<timestamp>",
    )
    args = parser.parse_args()

    config = Path(args.config).resolve()
    stages = args.stages or _read_num_stages(config)
    if stages < 2:
        raise SystemExit("--stages must be at least 2")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_dir).resolve() if args.log_dir else ROOT / "logs" / "local-runs" / timestamp

    print(f"config: {config}")
    print(f"stages: {stages}")
    print(f"logs:   {log_dir}")

    processes: dict[int, subprocess.Popen] = {}
    logs = []
    try:
        for stage in range(stages - 1, 0, -1):
            proc, log = _start_stage(stage, stages, config, args.python, None, log_dir)
            processes[stage] = proc
            logs.append(log)
            time.sleep(0.5)

        time.sleep(args.start_delay)
        proc0, log0 = _start_stage(0, stages, config, args.python, args.prompt, log_dir)
        processes[0] = proc0
        logs.append(log0)

        try:
            exit_code = proc0.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"stage 0 timed out after {args.timeout:.0f}s")
            return 124

        deadline = time.time() + 20
        for stage, proc in processes.items():
            if stage == 0:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"stage {stage} did not exit after Stage 0 completed")
                return 125

        print(f"done: stage 0 exited with {exit_code}")
        print(f"inspect logs under {log_dir}")
        return int(exit_code)
    finally:
        for proc in processes.values():
            _stop_process(proc)
        for proc in processes.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for log in logs:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
