import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from utils import trim_dynamic_cache


class DraftModel:
    """Small local draft model for speculative decoding (runs entirely on Stage 0)."""

    def __init__(self, model_name: str, temperature: float = 1.0, torch_dtype: torch.dtype = torch.float32):
        print(f"  [Draft] Loading {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.eval()
        self.temperature = temperature
        self._kv: DynamicCache = None

    def reset(self):
        self._kv = None

    def init_cache(self, input_ids: torch.Tensor) -> None:
        """Prefill the draft KV cache on a prompt (call once after target prefill)."""
        with torch.no_grad():
            out = self.model(input_ids, use_cache=True)
        self._kv = out.past_key_values

    def draft_k_tokens(self, seed_token: torch.Tensor, k: int):
        """
        Autoregressively draft k tokens starting from seed_token (shape (1,1),
        which is NOT yet in self._kv). Returns (tokens (1,k), probs (k,)).
        """
        tokens, probs = [], []
        current = seed_token
        with torch.no_grad():
            for _ in range(k):
                out = self.model(current, past_key_values=self._kv, use_cache=True)
                self._kv = out.past_key_values
                logits = out.logits[:, -1, :] / max(self.temperature, 1e-6)
                p = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(p, num_samples=1)  # (1, 1)
                probs.append(p[0, next_tok[0, 0]].item())
                tokens.append(next_tok[0, 0].item())
                current = next_tok
        return (
            torch.tensor([tokens], dtype=torch.long),
            torch.tensor(probs),
        )

    def advance_cache(self, token_id: int) -> None:
        """Process one token to advance KV by 1 position (used to sync after full acceptance)."""
        tok = torch.tensor([[token_id]], dtype=torch.long)
        with torch.no_grad():
            out = self.model(tok, past_key_values=self._kv, use_cache=True)
        self._kv = out.past_key_values

    def trim_cache(self, keep_len: int) -> None:
        """Trim KV cache to keep_len sequence positions (mirrors target cache trim)."""
        if self._kv is None:
            return
        trim_dynamic_cache(self._kv, keep_len)
