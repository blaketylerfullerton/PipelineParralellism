import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if torch is not None:
    from utils import (  # noqa: E402
        decode_activation,
        encode_activation,
        get_forward_port,
        get_telemetry_port,
        get_token_return_port,
        load_config,
    )


class ConfigAndCodecTests(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("torch is not installed")

    def test_smoke_config_loads(self):
        config = load_config(str(ROOT / "config.smoke.yaml"))
        self.assertEqual(config["pipeline"]["num_stages"], 2)
        self.assertEqual(config["model"]["arch"], "gpt2")

    def test_port_layout_tracks_stage_count(self):
        config = {
            "pipeline": {"num_stages": 4},
            "network": {"base_port": 5550},
        }
        self.assertEqual(get_forward_port(config, 3), 5553)
        self.assertEqual(get_telemetry_port(config, 0), 5558)
        self.assertEqual(get_token_return_port(config), 5562)

    def test_int8_activation_round_trip_preserves_shape(self):
        tensor = torch.tensor([[[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]]])
        encoded = encode_activation(tensor, "int8")
        decoded = decode_activation(encoded)
        self.assertEqual(tuple(decoded.shape), tuple(tensor.shape))
        self.assertTrue(torch.allclose(decoded, tensor, atol=0.03))


if __name__ == "__main__":
    unittest.main()
