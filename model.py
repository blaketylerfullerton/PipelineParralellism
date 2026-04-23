import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache
from typing import Optional, Tuple


class Stage0Module(nn.Module):
    """Token embedding + positional embedding + first N transformer blocks."""

    def __init__(self, transformer, num_blocks: int):
        super().__init__()
        self.wte = transformer.wte
        self.wpe = transformer.wpe
        self.drop = transformer.drop
        self.blocks = nn.ModuleList(list(transformer.h)[:num_blocks])

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
    ) -> Tuple[torch.Tensor, DynamicCache]:
        seq_len = input_ids.shape[1]
        # Offset positional IDs by the number of tokens already cached.
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        pos_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.drop(self.wte(input_ids) + self.wpe(pos_ids))

        if past_key_values is None:
            past_key_values = DynamicCache()

        for block in self.blocks:
            hidden = block(hidden, past_key_values=past_key_values, use_cache=True)

        return hidden, past_key_values


class MiddleModule(nn.Module):
    """A slice of transformer blocks — no embedding, no head."""

    def __init__(self, blocks: list):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        hidden: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
    ) -> Tuple[torch.Tensor, DynamicCache]:
        if past_key_values is None:
            past_key_values = DynamicCache()

        for block in self.blocks:
            hidden = block(hidden, past_key_values=past_key_values, use_cache=True)

        return hidden, past_key_values


class LastModule(nn.Module):
    """Last transformer blocks + final LayerNorm + LM head → logits."""

    def __init__(self, blocks: list, ln_f: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = ln_f
        self.lm_head = lm_head

    def forward(
        self,
        hidden: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
    ) -> Tuple[torch.Tensor, DynamicCache]:
        if past_key_values is None:
            past_key_values = DynamicCache()

        for block in self.blocks:
            hidden = block(hidden, past_key_values=past_key_values, use_cache=True)

        return self.lm_head(self.ln_f(hidden)), past_key_values


def _block_slices(num_blocks: int, num_stages: int) -> list:
    """Return list of (start, end) index pairs, one per stage."""
    base = num_blocks // num_stages
    remainder = num_blocks % num_stages
    slices, start = [], 0
    for i in range(num_stages):
        count = base + (1 if i < remainder else 0)
        slices.append((start, start + count))
        start += count
    return slices


def get_stage(stage_id: int, num_stages: int, config: dict) -> nn.Module:
    model_name = config["model"].get("name", "gpt2")
    print(f"  Loading {model_name} weights (stage {stage_id})...")
    full = GPT2LMHeadModel.from_pretrained(model_name)
    full.eval()

    all_blocks = list(full.transformer.h)
    slices = _block_slices(len(all_blocks), num_stages)
    s, e = slices[stage_id]

    if stage_id == 0:
        return Stage0Module(full.transformer, e - s)
    elif stage_id == num_stages - 1:
        return LastModule(all_blocks[s:e], full.transformer.ln_f, full.lm_head)
    else:
        return MiddleModule(all_blocks[s:e])


def get_tokenizer(config: dict) -> GPT2Tokenizer:
    model_name = config["model"].get("name", "gpt2")
    tok = GPT2Tokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return tok
