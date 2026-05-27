import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

from llama_baseline import build_generate_cmd, parse_timings, resolve_model_args  # noqa: E402


class LlamaBaselineTests(unittest.TestCase):
    def test_resolve_model_args_from_derived_hf_repo(self):
        config = {
            "model": {"name": "unsloth/Llama-3.2-3B"},
            "quantization": {"weight_mode": "q4_k_m"},
        }
        self.assertEqual(
            resolve_model_args(config),
            ["--hf-repo", "unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"],
        )

    def test_resolve_model_args_prefers_explicit_model_path(self):
        config = {
            "model": {"name": "ignored"},
            "quantization": {"weight_mode": "q4_k_m"},
            "llama_cpp": {"model_path": "/tmp/model.gguf"},
        }
        self.assertEqual(resolve_model_args(config), ["--model", "/tmp/model.gguf"])

    def test_parse_timings_extracts_prompt_and_eval_tps(self):
        text = """
llama_print_timings: prompt eval time =     120.00 ms /    12 tokens (   10.00 ms per token,   100.00 tokens per second)
llama_print_timings: eval time =     500.00 ms /    25 tokens (   20.00 ms per token,    50.00 tokens per second)
"""
        timings = parse_timings(text)
        self.assertEqual(timings["prompt_eval"]["tokens"], 12)
        self.assertEqual(timings["eval"]["tokens"], 25)
        self.assertAlmostEqual(timings["prompt_eval"]["tps"], 100.0)
        self.assertAlmostEqual(timings["eval"]["tps"], 50.0)

    def test_parse_timings_falls_back_to_bracket_summary(self):
        timings = parse_timings("[ Prompt: 47.3 t/s | Generation: 30.3 t/s ]")
        self.assertAlmostEqual(timings["prompt_eval"]["tps"], 47.3)
        self.assertAlmostEqual(timings["eval"]["tps"], 30.3)

    def test_build_generate_cmd_uses_expected_llama_flags(self):
        config = {
            "model": {
                "name": "unsloth/Llama-3.2-3B",
                "temperature": 0.7,
            },
            "quantization": {"weight_mode": "q4_k_m"},
            "llama_cpp": {"ctx_size": 2048, "n_gpu_layers": 0},
        }
        cmd = build_generate_cmd(
            config,
            "hello",
            n_predict=16,
            n_gpu_layers=0,
            threads=4,
        )
        self.assertIn("--hf-repo", cmd)
        self.assertIn("unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", cmd)
        self.assertIn("--n-predict", cmd)
        self.assertIn("--ctx-size", cmd)
        self.assertIn("--device", cmd)
        self.assertIn("none", cmd)
        self.assertIn("--single-turn", cmd)
        self.assertIn("--simple-io", cmd)
        self.assertIn("--perf", cmd)


if __name__ == "__main__":
    unittest.main()
