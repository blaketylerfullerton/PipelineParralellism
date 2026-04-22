import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
import zmq


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required = ["pipeline", "model", "network"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"config.yaml missing required section: {key}")
    return cfg


def get_forward_port(config: dict, stage_id: int) -> int:
    return config["network"]["base_port"] + stage_id


def get_backward_port(config: dict, stage_id: int) -> int:
    n = config["pipeline"]["num_stages"]
    return config["network"]["base_port"] + n + stage_id


def get_telemetry_port(config: dict, stage_id: int) -> int:
    n = config["pipeline"]["num_stages"]
    return config["network"]["base_port"] + 2 * n + stage_id


def get_token_return_port(config: dict) -> int:
    """Dedicated port: last stage PUSH → Stage 0 PULL for generated tokens."""
    n = config["pipeline"]["num_stages"]
    return config["network"]["base_port"] + 3 * n


def get_worker_address(config: dict, stage_id: int) -> str:
    host = config["network"]["workers"][stage_id]
    port = get_forward_port(config, stage_id)
    return f"tcp://{host}:{port}"


# --- Socket factories ---

def make_pull_socket(context: zmq.Context, port: int) -> zmq.Socket:
    sock = context.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, 20)
    sock.bind(f"tcp://*:{port}")
    return sock


def make_push_socket(context: zmq.Context, host: str, port: int) -> zmq.Socket:
    sock = context.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 20)
    sock.connect(f"tcp://{host}:{port}")
    return sock


def make_pub_socket(context: zmq.Context, port: int) -> zmq.Socket:
    sock = context.socket(zmq.PUB)
    sock.bind(f"tcp://*:{port}")
    return sock


def make_sub_socket(context: zmq.Context, hosts_and_ports: List[Tuple[str, int]]) -> zmq.Socket:
    sock = context.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    for host, port in hosts_and_ports:
        sock.connect(f"tcp://{host}:{port}")
    return sock


# --- Tensor serialization ---

def tensor_to_bytes(t: torch.Tensor) -> Tuple[bytes, tuple, str]:
    arr = t.detach().cpu().numpy()
    return arr.tobytes(), arr.shape, arr.dtype.str


def bytes_to_tensor(data: bytes, shape: tuple, dtype_str: str, requires_grad: bool = False) -> torch.Tensor:
    arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape).copy()
    t = torch.from_numpy(arr)
    if requires_grad:
        t.requires_grad_(True)
    return t


# --- Message constructors ---

def make_activation_msg(micro_batch_id: int, stage_id: int, tensor: torch.Tensor) -> bytes:
    raw, shape, dtype = tensor_to_bytes(tensor)
    msg = {
        "msg_type": "activation",
        "micro_batch_id": micro_batch_id,
        "stage_id": stage_id,
        "tensor": raw,
        "shape": shape,
        "dtype": dtype,
        "timestamp_sent": time.time(),
    }
    return pickle.dumps(msg)


def make_gradient_msg(micro_batch_id: int, stage_id: int, tensor: torch.Tensor) -> bytes:
    raw, shape, dtype = tensor_to_bytes(tensor)
    msg = {
        "msg_type": "gradient",
        "micro_batch_id": micro_batch_id,
        "stage_id": stage_id,
        "tensor": raw,
        "shape": shape,
        "dtype": dtype,
        "timestamp_sent": time.time(),
    }
    return pickle.dumps(msg)


def make_control_msg(msg_type: str, **kwargs) -> bytes:
    msg = {"msg_type": msg_type, "timestamp_sent": time.time(), **kwargs}
    return pickle.dumps(msg)


# --- Send / Recv ---

def send_msg(socket: zmq.Socket, msg_bytes: bytes) -> None:
    socket.send(msg_bytes)


def recv_msg(socket: zmq.Socket, timeout_ms: int = 5000) -> Optional[Dict[str, Any]]:
    if socket.poll(timeout_ms, zmq.POLLIN):
        data = socket.recv()
        return pickle.loads(data)
    return None
