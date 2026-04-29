"""
Auto-discovery launcher for pipeline parallelism.

Each machine runs this script with its stage number. Machines find each other
via UDP broadcast on the LAN, a live display shows who is online, and the
pipeline starts automatically once all stages are present.

Usage:
  Machine A (driver):  python launch.py --stage 0
  Machine B:           python launch.py --stage 1

Optional:
  --stages 2      override num_stages from config (useful for 2-machine runs)
  --prompt "..."  provide prompt upfront on Stage 0 (otherwise it asks)
  --config path   path to config.yaml (default: ./config.yaml)
"""

import argparse
import json
import os
import socket
import subprocess
import threading
import time
from typing import Dict, List, Optional

import torch
import zmq

torch.set_num_threads(os.cpu_count() or 1)
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

import worker as worker_module
from model import get_stage
from utils import load_config

DISCOVERY_PORT = 5599
PEER_TIMEOUT = 6.0      # seconds before marking a peer as offline
READY_HOLD = 2.0        # seconds all peers must be present before auto-start
TAILSCALE_REFRESH = 10.0  # how often to re-query Tailscale peer list


def _get_tailscale_ip() -> Optional[str]:
    """Return this machine's Tailscale IP, or None if Tailscale isn't running."""
    candidates = ["tailscale", "/usr/bin/tailscale", "/usr/sbin/tailscale",
                  "/usr/local/bin/tailscale", "/snap/bin/tailscale"]
    for binary in candidates:
        try:
            result = subprocess.run(
                [binary, "ip", "-4"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                ip = result.stdout.strip()
                if ip:
                    return ip
        except Exception:
            continue
    return None


def _get_tailscale_peer_ips() -> List[str]:
    """Return IPs of currently-online Tailscale peers."""
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        peers = []
        for peer in data.get("Peer", {}).values():
            if peer.get("Online") and peer.get("TailscaleIPs"):
                peers.append(peer["TailscaleIPs"][0])
        return peers
    except Exception:
        return []


def get_local_ip() -> str:
    """Return this machine's best reachable IP — Tailscale if available, else LAN."""
    ts_ip = _get_tailscale_ip()
    if ts_ip:
        return ts_ip
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


class DiscoveryManager:
    """
    Broadcasts this machine's presence on the LAN and collects peer info.
    Each worker announces: stage_id, IP address, hostname.
    """

    def __init__(self, my_stage: int, num_stages: int, my_ip: str, hostname: str):
        self.my_stage = my_stage
        self.num_stages = num_stages
        self.my_ip = my_ip
        self.hostname = hostname

        self._lock = threading.Lock()
        self._peers: Dict[int, dict] = {
            my_stage: {"ip": my_ip, "hostname": hostname, "last_seen": time.time()}
        }
        self._running = False
        self._tailscale_peers: List[str] = []
        self._tailscale_last_refresh: float = 0.0

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._broadcast_loop, daemon=True).start()
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop(self) -> None:
        self._running = False

    def peers(self) -> Dict[int, dict]:
        with self._lock:
            return dict(self._peers)

    def all_online(self) -> bool:
        with self._lock:
            cutoff = time.time() - PEER_TIMEOUT
            alive = [s for s, p in self._peers.items() if p["last_seen"] > cutoff]
            return len(alive) == self.num_stages

    def worker_ips(self) -> List[str]:
        with self._lock:
            return [self._peers[i]["ip"] for i in range(self.num_stages)]

    def _broadcast_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        msg = json.dumps({
            "type": "pp_peer",
            "stage": self.my_stage,
            "num_stages": self.num_stages,
            "ip": self.my_ip,
            "hostname": self.hostname,
        }).encode()
        while self._running:
            # LAN broadcast (same-subnet peers)
            try:
                sock.sendto(msg, ("255.255.255.255", DISCOVERY_PORT))
            except Exception:
                pass

            # Refresh Tailscale peer list periodically
            now = time.time()
            if now - self._tailscale_last_refresh > TAILSCALE_REFRESH:
                self._tailscale_peers = _get_tailscale_peer_ips()
                self._tailscale_last_refresh = now

            # Unicast to each Tailscale peer (works across subnets/VPN)
            for peer_ip in self._tailscale_peers:
                try:
                    sock.sendto(msg, (peer_ip, DISCOVERY_PORT))
                except Exception:
                    pass

            time.sleep(0.8)
        sock.close()

    def _listen_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        try:
            # SO_REUSEPORT lets multiple processes on the same host each receive
            # the same UDP broadcast — essential when testing with 2 workers locally.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            sock.bind(("", DISCOVERY_PORT))
        except OSError as e:
            print(f"[Discovery] Warning: could not bind port {DISCOVERY_PORT}: {e}")
            return
        while self._running:
            try:
                data, _ = sock.recvfrom(512)
                msg = json.loads(data)
                if msg.get("type") != "pp_peer":
                    continue
                stage = msg["stage"]
                if 0 <= stage < self.num_stages:
                    with self._lock:
                        self._peers[stage] = {
                            "ip": msg["ip"],
                            "hostname": msg.get("hostname", "?"),
                            "last_seen": time.time(),
                        }
            except (socket.timeout, json.JSONDecodeError):
                pass
            except Exception:
                pass
        sock.close()


def _build_discovery_panel(disc: DiscoveryManager) -> Panel:
    peers = disc.peers()
    now = time.time()
    cutoff = now - PEER_TIMEOUT

    lines = Text()
    n_online = 0
    for i in range(disc.num_stages):
        peer = peers.get(i)
        alive = peer and peer["last_seen"] > cutoff
        if alive:
            n_online += 1
            color = "green"
            dot = "●"
            info = f"{peer['ip']}  ({peer['hostname']})"
            tag = "  ← this machine" if i == disc.my_stage else ""
        else:
            color = "yellow"
            dot = "○"
            info = "not seen yet"
            tag = "  ← this machine" if i == disc.my_stage else ""

        lines.append(f"  Stage {i}  ", style="bold white")
        lines.append(f"{info}{tag} ", style="dim")
        lines.append(f"{dot}\n", style=f"bold {color}")

    lines.append("\n")
    missing = disc.num_stages - n_online
    if missing == 0:
        lines.append("  All stages online! Starting in a moment...\n", style="bold green")
    else:
        s = "stage" if missing == 1 else "stages"
        lines.append(f"  Waiting for {missing} more {s}...\n", style="bold yellow")
        lines.append(
            f"  On the other machine run:  python launch.py --stage N --stages {disc.num_stages}\n",
            style="dim"
        )

    title = "[bold green]All Online!" if missing == 0 else "[bold magenta]Discovering Peers"
    border = "green" if missing == 0 else "magenta"
    return Panel(lines, title=title, border_style=border, padding=(0, 1))


def run_discovery_ui(disc: DiscoveryManager, console: Console) -> None:
    """Block until all stages are online, showing a live status panel."""
    console.print()
    with Live(
        _build_discovery_panel(disc),
        console=console,
        refresh_per_second=4,
        transient=False,
    ) as live:
        ready_since: Optional[float] = None
        while True:
            live.update(_build_discovery_panel(disc))
            if disc.all_online():
                if ready_since is None:
                    ready_since = time.time()
                elif time.time() - ready_since >= READY_HOLD:
                    break
            else:
                ready_since = None
            time.sleep(0.25)


def _show_static_peers(worker_ips: List[str], my_stage: int, console: Console) -> None:
    """Show a one-shot panel when IPs are provided manually (no discovery needed)."""
    lines = Text()
    for i, ip in enumerate(worker_ips):
        tag = "  ← this machine" if i == my_stage else ""
        lines.append(f"  Stage {i}  ", style="bold white")
        lines.append(f"{ip}{tag} ", style="dim")
        lines.append("●\n", style="bold green")
    lines.append("\n  All stages configured — starting pipeline.\n", style="bold green")
    console.print(Panel(lines, title="[bold green]Peers (manual)", border_style="green", padding=(0, 1)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-discover peers and run pipeline parallelism"
    )
    parser.add_argument("--stage", type=int, required=True, help="This machine's stage (0-indexed)")
    parser.add_argument("--stages", type=int, default=None, help="Total number of stages (overrides config)")
    parser.add_argument(
        "--peer-ip", nargs="+", metavar="IP",
        help=(
            "Skip auto-discovery and use these IPs directly, listed in stage order. "
            "Example (2 machines): --peer-ip 172.16.0.162 192.168.1.66"
        ),
    )
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt (Stage 0 only)")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stages is not None:
        config["pipeline"]["num_stages"] = args.stages
    num_stages = config["pipeline"]["num_stages"]

    if args.stage >= num_stages:
        raise SystemExit(f"--stage {args.stage} is out of range for {num_stages} stages (0–{num_stages-1})")

    my_ip = get_local_ip()
    hostname = socket.gethostname()

    console = Console()
    console.rule("[bold magenta]Pipeline Parallelism")
    console.print(
        f"  Stage [bold cyan]{args.stage}[/]  |  "
        f"IP [cyan]{my_ip}[/]  |  "
        f"Host [cyan]{hostname}[/]  |  "
        f"Model [cyan]{config['model'].get('name', 'gpt2')}[/]  |  "
        f"Stages [cyan]{num_stages}[/]  |  "
        f"Torch threads [cyan]{torch.get_num_threads()}[/] / {os.cpu_count()} cores"
    )

    # --- Discovery or manual IP phase ---
    if args.peer_ip:
        if len(args.peer_ip) != num_stages:
            raise SystemExit(
                f"--peer-ip needs exactly {num_stages} IPs (one per stage), got {len(args.peer_ip)}: {args.peer_ip}"
            )
        ips = args.peer_ip
        _show_static_peers(ips, args.stage, console)
    else:
        disc = DiscoveryManager(args.stage, num_stages, my_ip, hostname)
        disc.start()
        run_discovery_ui(disc, console)
        disc.stop()
        ips = disc.worker_ips()

    config["network"]["workers"] = ips
    console.print(f"\n  Workers: {dict(enumerate(ips))}")

    # --- Load this stage's model slice ---
    console.print(f"\n  Loading Stage {args.stage} model slice...")
    stage_model = get_stage(args.stage, num_stages, config)
    stage_model.eval()

    # --- Set up ZMQ sockets ---
    ctx = zmq.Context()
    sockets = worker_module.setup_sockets(ctx, config, args.stage, num_stages)

    # Brief pause so all machines finish binding sockets before Stage 0 starts sending
    console.print("  Sockets ready, syncing with peers...")
    time.sleep(2.0)

    # --- Get prompt on Stage 0 ---
    prompt = args.prompt
    if args.stage == 0 and prompt is None:
        console.print()
        prompt = console.input("[bold cyan]Prompt:[/] ")

    console.rule("[bold green]Running Pipeline")

    try:
        worker_module.generation_loop(
            stage_model=stage_model,
            sockets=sockets,
            config=config,
            stage_id=args.stage,
            num_stages=num_stages,
            host=my_ip,
            prompt=prompt,
        )
    finally:
        for sock in sockets.values():
            sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
