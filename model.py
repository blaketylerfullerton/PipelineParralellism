import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def _run_block(block: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Call a GPT-2 transformer block, handling both tuple (<=4.x) and tensor (>=5.x) return formats."""
    out = block(hidden)
    return out[0] if isinstance(out, tuple) else out


class Stage0Module(nn.Module):
    """Token embedding + positional embedding + first N transformer blocks."""

    def __init__(self, transformer, num_blocks: int):
        super().__init__()
        self.wte = transformer.wte
        self.wpe = transformer.wpe
        self.drop = transformer.drop
        self.blocks = nn.ModuleList(list(transformer.h)[:num_blocks])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.drop(self.wte(input_ids) + self.wpe(pos_ids))
        for block in self.blocks:
            hidden = _run_block(block, hidden)
        return hidden


class MiddleModule(nn.Module):
    """A slice of transformer blocks — no embedding, no head."""

    def __init__(self, blocks: list):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = _run_block(block, hidden)
        return hidden


class LastModule(nn.Module):
    """Last transformer blocks + final LayerNorm + LM head → logits."""

    def __init__(self, blocks: list, ln_f: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = ln_f
        self.lm_head = lm_head

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = _run_block(block, hidden)
        return self.lm_head(self.ln_f(hidden))


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
