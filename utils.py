import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

from transformers.cache_utils import DynamicCache

import numpy as np
import torch
import yaml
import zmq

_SOCKET_BUF = 4 * 1024 * 1024  # 4 MB — reduces latency jitter on WAN


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
    sock.setsockopt(zmq.RCVBUF, _SOCKET_BUF)
    sock.bind(f"tcp://*:{port}")
    return sock


def make_push_socket(context: zmq.Context, host: str, port: int) -> zmq.Socket:
    sock = context.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 20)
    sock.setsockopt(zmq.SNDBUF, _SOCKET_BUF)
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


# --- Activation codec ---

def encode_activation(tensor: torch.Tensor, mode: str) -> dict:
    """Encode tensor for wire transmission. Returns codec fields to embed in a message."""
    if mode == "fp16":
        arr = tensor.detach().cpu().to(torch.float16).numpy()
        return {"codec": "fp16", "data": arr.tobytes(), "shape": arr.shape, "dtype": arr.dtype.str}
    if mode == "int8":
        t = tensor.detach().cpu().float()
        scale = float(t.abs().max()) / 127.0 or 1.0
        q = (t / scale).round().clamp(-127, 127).to(torch.int8)
        arr = q.numpy()
        return {"codec": "int8", "data": arr.tobytes(), "shape": arr.shape, "dtype": arr.dtype.str, "scale": scale}
    # fp32 (default)
    arr = tensor.detach().cpu().float().numpy()
    return {"codec": "fp32", "data": arr.tobytes(), "shape": arr.shape, "dtype": arr.dtype.str}


def decode_activation(encoded: dict) -> torch.Tensor:
    """Decode a tensor from a codec dict (as embedded in an activation message)."""
    arr = np.frombuffer(encoded["data"], dtype=np.dtype(encoded["dtype"])).reshape(encoded["shape"]).copy()
    t = torch.from_numpy(arr)
    if encoded.get("codec") == "int8":
        return t.float() * encoded["scale"]
    if encoded.get("codec") == "fp16":
        return t.float()
    return t


def tensor_from_activation_msg(msg: dict) -> torch.Tensor:
    """Decode the hidden-state tensor from a received activation message."""
    return decode_activation(msg)


# --- Message constructors ---

def trim_dynamic_cache(cache: DynamicCache, keep_len: int) -> None:
    """Trim a DynamicCache in-place to keep_len sequence positions."""
    cache.crop(keep_len)


def make_activation_msg(
    micro_batch_id: int,
    stage_id: int,
    tensor: torch.Tensor,
    is_prefill: bool = False,
    codec: str = "fp32",
    draft_tokens: Optional[List[int]] = None,
    cascade_prefix_len: int = 0,
) -> bytes:
    encoded = encode_activation(tensor, codec)
    msg = {
        "msg_type": "activation",
        "micro_batch_id": micro_batch_id,
        "stage_id": stage_id,
        "is_prefill": is_prefill,
        "draft_tokens": draft_tokens,        # list of K ints (spec mode) or None
        "cascade_prefix_len": cascade_prefix_len,  # N cascade hiddens prepended to this tensor
        "timestamp_sent": time.time(),
        **encoded,
    }
    return pickle.dumps(msg)


def make_spec_result_msg(step: int, target_probs: list, target_samples: list) -> bytes:
    """
    Sent by last stage → Stage 0 in speculative mode.
    target_probs: [K] floats — p(draft_i | context) for each draft position.
    target_samples: [K+1] ints — one sampled token per position (for rejection fallback + bonus).
    For prefill: target_probs=[], target_samples=[first_token].
    """
    msg = {
        "msg_type": "spec_result",
        "step": step,
        "target_probs": target_probs,
        "target_samples": target_samples,
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
