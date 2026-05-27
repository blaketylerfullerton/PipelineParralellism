import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

from llama_rpc_baseline import (  # noqa: E402
    _rpc_endpoint,
    build_bench_cmd,
    build_remote_rpc_command,
)


class LlamaRpcBaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model": {"name": "unsloth/Llama-3.2-3B"},
            "quantization": {"weight_mode": "q4_k_m"},
            "rpc": {
                "remote_stage": 1,
                "port": 50052,
                "bind_host": "0.0.0.0",
                "ssh_user": "root",
                "remote_binary": "/opt/llama.cpp-rpc/bin/rpc-server",
                "cache": True,
                "debug": True,
            },
            "bench": {
                "binary": "/opt/homebrew/bin/llama-bench",
                "n_prompt": 0,
                "n_gen": 128,
                "repetitions": 3,
                "threads": 8,
                "n_gpu_layers": 0,
                "device": "none",
                "output": "json",
            },
        }
        self.state = {
            "stages": [
                {"stage": 0, "public_ip": "143.198.72.167", "wireguard_ip": "10.99.0.1"},
                {"stage": 1, "public_ip": "64.23.177.125", "wireguard_ip": "10.99.0.2"},
            ]
        }

    def test_rpc_endpoint_uses_remote_wireguard_ip(self):
        self.assertEqual(_rpc_endpoint(self.config, self.state), "10.99.0.2:50052")

    def test_build_remote_rpc_command_uses_public_ip_for_ssh(self):
        ssh_target, remote_cmd = build_remote_rpc_command(self.config, self.state)
        self.assertEqual(ssh_target, "root@64.23.177.125")
        self.assertIn("/opt/llama.cpp-rpc/bin/rpc-server", remote_cmd)
        self.assertIn("--host 0.0.0.0", remote_cmd)
        self.assertIn("-p 50052", remote_cmd)
        self.assertIn("-c", remote_cmd)

    def test_build_bench_cmd_uses_rpc_endpoint_and_hf_repo(self):
        cmd = build_bench_cmd(self.config, self.state)
        joined = " ".join(cmd)
        self.assertIn("--rpc", cmd)
        self.assertIn("10.99.0.2:50052", cmd)
        self.assertIn("--hf-repo", cmd)
        self.assertIn("unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", joined)
        self.assertIn("--output", cmd)
        self.assertIn("json", cmd)


if __name__ == "__main__":
    unittest.main()
