import copy
import threading
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from utils import get_device, trim_dynamic_cache


def _draft_k_tokens_on_cache(
    model: torch.nn.Module,
    device: torch.device,
    temperature: float,
    kv: DynamicCache,
    seed_token: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DynamicCache]:
    """
    Autoregressively draft k tokens starting from seed_token using a given KV cache.
    Does NOT touch any instance state — safe to call from a background thread.
    Returns (tokens (1,k), sampled_probs (k,), max_probs (k,), final_kv).
    """
    tokens, probs, max_probs = [], [], []
    current = seed_token.to(device)
    with torch.no_grad():
        for _ in range(k):
            out = model(current, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            logits = out.logits[:, -1, :] / max(temperature, 1e-6)
            p = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(p, num_samples=1)
            probs.append(p[0, next_tok[0, 0]].item())
            max_probs.append(p.max().item())
            tokens.append(next_tok[0, 0].item())
            current = next_tok
    return (
        torch.tensor([tokens], dtype=torch.long),  # CPU
        torch.tensor(probs),                        # CPU
        torch.tensor(max_probs),                    # CPU
        kv,
    )


class DraftModel:
    """Small local draft model for speculative decoding (runs entirely on Stage 0)."""

    def __init__(self, model_name: str, temperature: float = 1.0, torch_dtype: torch.dtype = torch.float32):
        self._device = get_device()
        print(f"  [Draft] Loading {model_name} on {self._device}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
        self.model = self.model.to(self._device)
        self.model.eval()
        self.temperature = temperature
        self._kv: Optional[DynamicCache] = None

        # Speculative pipeline prefetch state
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_result: Optional[dict] = None

    def reset(self):
        self._kv = None
        self._prefetch_thread = None
        self._prefetch_result = None

    def init_cache(self, input_ids: torch.Tensor) -> None:
        """Prefill the draft KV cache on a prompt (call once after target prefill)."""
        with torch.no_grad():
            out = self.model(input_ids.to(self._device), use_cache=True)
        self._kv = out.past_key_values

    def draft_peek(self, seed_token: torch.Tensor):
        """
        Run one draft step and return (token (1,1) CPU, sampled_prob, max_prob).
        Advances self._kv by one position (seed_token processed).
        max_prob is the peak of the distribution — use this for cascade decisions,
        not sampled_prob, which is the probability of whichever token was sampled.
        """
        current = seed_token.to(self._device)
        with torch.no_grad():
            out = self.model(current, past_key_values=self._kv, use_cache=True)
        self._kv = out.past_key_values
        logits = out.logits[:, -1, :] / max(self.temperature, 1e-6)
        p = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(p, num_samples=1)
        sampled_prob = p[0, next_tok[0, 0]].item()
        max_prob = p.max().item()
        return next_tok.cpu(), sampled_prob, max_prob

    def draft_k_tokens(
        self, seed_token: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressively draft k tokens starting from seed_token (shape (1,1),
        which is NOT yet in self._kv).
        Returns (tokens (1,k), sampled_probs (k,), max_probs (k,)).
        """
        tokens, probs, max_probs, self._kv = _draft_k_tokens_on_cache(
            self.model, self._device, self.temperature, self._kv, seed_token, k
        )
        return tokens, probs, max_probs

    def advance_cache(self, token_id: int) -> None:
        """Process one token to advance KV by 1 position (used to sync after full acceptance)."""
        tok = torch.tensor([[token_id]], dtype=torch.long, device=self._device)
        with torch.no_grad():
            out = self.model(tok, past_key_values=self._kv, use_cache=True)
        self._kv = out.past_key_values

    def trim_cache(self, keep_len: int) -> None:
        """Trim KV cache to keep_len sequence positions (mirrors target cache trim)."""
        if self._kv is None:
            return
        trim_dynamic_cache(self._kv, keep_len)

    # ------------------------------------------------------------------
    # Speculative pipeline prefetch

    def start_prefetch(self, seed_token: torch.Tensor, k: int) -> None:
        """
        Snapshot the current KV cache and launch a daemon thread that drafts k
        tokens from seed_token.  The caller's optimistic assumption is that
        seed_token will be the first token of the next round (i.e. all current
        draft tokens were accepted).

        The background thread uses an independent deep copy of the KV cache so
        there is no shared mutable state with the main thread.  The main thread
        is expected to be blocked on ZMQ recv while this thread runs, giving
        PyTorch the GIL time it needs.
        """
        # Deep copy so the background thread owns its own KV tensors.
        kv_snapshot = copy.deepcopy(self._kv)

        result: dict = {}
        model = self.model
        device = self._device
        temperature = self.temperature

        def _worker() -> None:
            toks, probs, max_probs, final_kv = _draft_k_tokens_on_cache(
                model, device, temperature, kv_snapshot, seed_token, k
            )
            result["tokens"] = toks
            result["probs"] = probs
            result["max_probs"] = max_probs
            result["kv"] = final_kv

        self._prefetch_result = result
        t = threading.Thread(target=_worker, daemon=True, name="spec_prefetch")
        t.start()
        self._prefetch_thread = t

    def consume_prefetch(
        self, actual_seed_id: int, assumed_seed_id: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DynamicCache]]:
        """
        Join the prefetch thread and, if the actual seed matches the assumed seed,
        return (tokens (1,k), sampled_probs (k,), max_probs (k,), final_kv).
        Returns None on mismatch (stale prefetch) or if no prefetch is in flight.

        Always joins the thread so no concurrent model use can occur after this call.
        The caller is responsible for setting self._kv = final_kv when adopting the result.
        """
        if self._prefetch_thread is None:
            return None
        self._prefetch_thread.join()
        self._prefetch_thread = None
        result = self._prefetch_result
        self._prefetch_result = None
        if actual_seed_id != assumed_seed_id:
            return None
        return result["tokens"], result["probs"], result["max_probs"], result["kv"]
