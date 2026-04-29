import argparse
import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import zmq
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from utils import get_telemetry_port, load_config, make_sub_socket


@dataclass
class StageStats:
    stage_id: int
    host: str = "?"
    state: str = "idle"        # idle | waiting | running | forward | done
    current_step: int = 0
    elapsed_ms: float = 0.0
    last_token: str = ""
    last_update: float = field(default_factory=time.time)


@dataclass
class PipelineStats:
    num_stages: int
    start_time: float = field(default_factory=time.time)
    stages: Dict[int, StageStats] = field(default_factory=dict)


_STATE_COLORS = {
    "idle": "grey50",
    "waiting": "yellow",
    "running": "green",
    "forward": "green",
    "done": "cyan",
    "error": "red",
}


class PipelineDashboard:
    def __init__(self, config: dict, standalone: bool = False):
        self._config = config
        self._standalone = standalone
        self._lock = threading.Lock()
        n = config["pipeline"]["num_stages"]
        self._stats = PipelineStats(num_stages=n)
        for i in range(n):
            self._stats.stages[i] = StageStats(stage_id=i)
        self._live: Optional[Live] = None
        self._console = Console()
        self._sub_thread: Optional[threading.Thread] = None
        self._running = False

    def update_stage(self, stage_id: int, **kwargs) -> None:
        with self._lock:
            s = self._stats.stages.get(stage_id)
            if s is None:
                return
            for k, v in kwargs.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            s.last_update = time.time()

    def update_pipeline(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._stats, k):
                    setattr(self._stats, k, v)

    def start(self) -> None:
        self._running = True
        if self._standalone:
            self._sub_thread = threading.Thread(target=self._zmq_subscriber_thread, daemon=True)
            self._sub_thread.start()
        refresh = self._config.get("dashboard", {}).get("refresh_rate_hz", 10)
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=refresh,
            screen=False,
        )
        self._live.__enter__()

    def stop(self) -> None:
        self._running = False
        if self._live:
            self._live.__exit__(None, None, None)

    def tick(self) -> None:
        if self._live:
            self._live.update(self._build_layout())

    def _build_layout(self) -> Layout:
        with self._lock:
            stats = self._stats

        elapsed = time.time() - stats.start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"

        header = Panel(
            f"[bold magenta]Pipeline Parallelism[/]  |  "
            f"Stages: [cyan]{stats.num_stages}[/]  |  "
            f"Elapsed: [cyan]{elapsed_str}[/]",
            border_style="magenta",
        )

        stage_panels = []
        for i in range(stats.num_stages):
            s = stats.stages.get(i, StageStats(stage_id=i))
            color = _STATE_COLORS.get(s.state, "white")
            stale = (time.time() - s.last_update) > 3.0

            lines = Text()
            lines.append(f"Host:   {s.host}\n", style="white")
            lines.append(f"State:  ", style="white")
            lines.append(f"{s.state.upper()}\n", style=f"bold {color}")
            lines.append(f"Step:   {s.current_step}\n", style="white")
            lines.append(f"Lat:    {s.elapsed_ms:.1f} ms\n", style="white")
            if s.last_token:
                lines.append(f"Token:  ", style="white")
                lines.append(f"{s.last_token}\n", style=f"bold {color}")

            stage_panels.append(Panel(
                lines,
                title=f"[bold {color}]STAGE {i}",
                border_style="dim" if stale else color,
                expand=True,
            ))

        layout = Layout()
        layout.split_column(
            Layout(header, size=3),
            Layout(Columns(stage_panels, equal=True, expand=True)),
        )
        return layout

    def _zmq_subscriber_thread(self) -> None:
        ctx = zmq.Context()
        config = self._config
        n = config["pipeline"]["num_stages"]
        endpoints = [
            (config["network"]["workers"][i], get_telemetry_port(config, i))
            for i in range(n)
        ]
        sock = make_sub_socket(ctx, endpoints)
        sock.setsockopt(zmq.RCVTIMEO, 500)

        while self._running:
            try:
                data = sock.recv()
                msg = pickle.loads(data)
                if msg.get("msg_type") == "telemetry":
                    sid = msg["stage_id"]
                    self.update_stage(
                        sid,
                        host=msg.get("host", "?"),
                        state=msg.get("state", "idle"),
                        current_step=msg.get("step", 0),
                        elapsed_ms=msg.get("elapsed_ms", 0.0),
                        last_token=msg.get("last_token", ""),
                    )
                    self.tick()
            except zmq.Again:
                self.tick()
            except Exception:
                pass

        sock.close()
        ctx.term()


_instance: Optional[PipelineDashboard] = None


def init_dashboard(config: dict, standalone: bool = False) -> PipelineDashboard:
    global _instance
    _instance = PipelineDashboard(config, standalone=standalone)
    _instance.start()
    return _instance


def get_dashboard() -> Optional[PipelineDashboard]:
    return _instance


def main():
    parser = argparse.ArgumentParser(description="Standalone pipeline dashboard")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    dash = init_dashboard(config, standalone=True)
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        dash.stop()


if __name__ == "__main__":
    main()
