import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from typing import List, Optional, Tuple

from utils import get_device


# ===================================================================== GPT-2

class Stage0Module(nn.Module):
    """Token embedding + positional embedding + first N transformer blocks (GPT-2)."""

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
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        pos_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.drop(self.wte(input_ids) + self.wpe(pos_ids))

        if past_key_values is None:
            past_key_values = DynamicCache()

        for block in self.blocks:
            hidden = block(hidden, past_key_values=past_key_values, use_cache=True)

        return hidden, past_key_values


class MiddleModule(nn.Module):
    """GPT-2 middle stage: a slice of transformer blocks."""

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
    """GPT-2 last stage: transformer blocks + final LayerNorm + LM head."""

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


# ===================================================================== Llama

def _relabel_layer_idx(layers: List[nn.Module]) -> None:
    """After slicing layers across stages, reset each layer's self_attn.layer_idx
    to its local position — keeps per-stage DynamicCache dense (entries 0..N-1)."""
    for new_idx, layer in enumerate(layers):
        layer.self_attn.layer_idx = new_idx


def _run_llama_layers(
    layers: nn.ModuleList,
    rotary_emb: nn.Module,
    config,
    hidden: torch.Tensor,
    past_key_values: DynamicCache,
) -> torch.Tensor:
    """Replicates the per-layer loop inside LlamaModel.forward for a stage subset."""
    past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    seq_len = hidden.shape[1]
    position_ids = torch.arange(past_len, past_len + seq_len, device=hidden.device).unsqueeze(0)

    position_embeddings = rotary_emb(hidden, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    for layer in layers:
        hidden = layer(
            hidden,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
    return hidden


class Stage0Llama(nn.Module):
    """Llama stage 0: embed_tokens + first N decoder layers."""

    def __init__(self, full_model, num_layers: int):
        super().__init__()
        base = full_model.model
        self.embed_tokens = base.embed_tokens
        self.rotary_emb = base.rotary_emb
        layers = list(base.layers)[:num_layers]
        _relabel_layer_idx(layers)
        self.layers = nn.ModuleList(layers)
        self.config = full_model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
    ) -> Tuple[torch.Tensor, DynamicCache]:
        if past_key_values is None:
            past_key_values = DynamicCache()
        input_ids = input_ids.to(self.embed_tokens.weight.device)
        hidden = self.embed_tokens(input_ids)
        hidden = _run_llama_layers(self.layers, self.rotary_emb, self.config, hidden, past_key_values)
        return hidden.cpu(), past_key_values


def _match_weight_dtype(module: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Move hidden states to match the stage's device and dtype.
    Hidden arrives over the wire as CPU fp32; this handles both casts in one call."""
    try:
        param = next(module.parameters())
    except StopIteration:
        return hidden
    return hidden.to(device=param.device, dtype=param.dtype)


class MiddleLlama(nn.Module):
    """Llama middle stage: a slice of decoder layers."""

    def __init__(self, full_model, start: int, end: int):
        super().__init__()
        self.rotary_emb = full_model.model.rotary_emb
        layers = list(full_model.model.layers)[start:end]
        _relabel_layer_idx(layers)
        self.layers = nn.ModuleList(layers)
        self.config = full_model.config

    def forward(
        self,
        hidden: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
    ) -> Tuple[torch.Tensor, DynamicCache]:
        if past_key_values is None:
            past_key_values = DynamicCache()
        hidden = _match_weight_dtype(self, hidden)
        hidden = _run_llama_layers(self.layers, self.rotary_emb, self.config, hidden, past_key_values)
        return hidden.cpu(), past_key_values


class LastLlama(nn.Module):
    """Llama last stage: decoder layers + final RMSNorm + LM head → logits."""

    def __init__(self, full_model, start: int, end: int):
        super().__init__()
        base = full_model.model
        self.rotary_emb = base.rotary_emb
        layers = list(base.layers)[start:end]
        _relabel_layer_idx(layers)
        self.layers = nn.ModuleList(layers)
        self.norm = base.norm
        # Tied embeddings: lm_head.weight shares storage with embed_tokens.weight.
        # Keep lm_head as-is; its parameter still holds the tied tensor.
        self.lm_head = full_model.lm_head
        self.config = full_model.config

    def forward(
        self,
        hidden: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
    ) -> Tuple[torch.Tensor, DynamicCache]:
        if past_key_values is None:
            past_key_values = DynamicCache()
        hidden = _match_weight_dtype(self, hidden)
        hidden = _run_llama_layers(self.layers, self.rotary_emb, self.config, hidden, past_key_values)
        return self.lm_head(self.norm(hidden)).cpu(), past_key_values


# =================================================================== Helpers

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


_DTYPE_MAP = {
    "fp32": torch.float32, "float32": torch.float32,
    "fp16": torch.float16, "float16": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
}


def resolve_dtype(config: dict):
    name = str(config["model"].get("dtype", "fp32")).lower()
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown model.dtype '{name}' — use fp32 / fp16 / bf16")
    return _DTYPE_MAP[name]


# =================================================================== Dispatch

def get_stage(stage_id: int, num_stages: int, config: dict) -> nn.Module:
    arch = config["model"].get("arch", "gpt2").lower()
    if arch == "llama":
        return _get_stage_llama(stage_id, num_stages, config)
    return _get_stage_gpt2(stage_id, num_stages, config)


def _get_stage_gpt2(stage_id: int, num_stages: int, config: dict) -> nn.Module:
    model_name = config["model"].get("name", "gpt2")
    print(f"  Loading {model_name} weights (stage {stage_id})...")
    full = GPT2LMHeadModel.from_pretrained(model_name)
    full.eval()

    all_blocks = list(full.transformer.h)
    s, e = _block_slices(len(all_blocks), num_stages)[stage_id]

    if stage_id == 0:
        return Stage0Module(full.transformer, e - s)
    elif stage_id == num_stages - 1:
        return LastModule(all_blocks[s:e], full.transformer.ln_f, full.lm_head)
    else:
        return MiddleModule(all_blocks[s:e])


def _get_stage_llama(stage_id: int, num_stages: int, config: dict) -> nn.Module:
    model_name = config["model"]["name"]
    dtype = resolve_dtype(config)
    print(f"  Loading {model_name} ({dtype}) — stage {stage_id}/{num_stages - 1}...")
    full = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    full.eval()

    num_layers = len(full.model.layers)
    s, e = _block_slices(num_layers, num_stages)[stage_id]
    print(f"  Stage {stage_id}: layers [{s}, {e}) of {num_layers}")

    if stage_id == 0:
        stage = Stage0Llama(full, e)
    elif stage_id == num_stages - 1:
        stage = LastLlama(full, s, e)
    else:
        stage = MiddleLlama(full, s, e)

    device = get_device()
    print(f"  Moving stage {stage_id} to {device}...")
    stage = stage.to(device)

    # Release the unused slice of the full model. Tied lm_head / embed_tokens
    # parameters still referenced by the stage stay alive via reference counting.
    del full
    gc.collect()
    return stage


def get_tokenizer(config: dict):
    model_name = config["model"].get("name", "gpt2")
    arch = config["model"].get("arch", "gpt2").lower()
    if arch == "llama":
        tok = AutoTokenizer.from_pretrained(model_name)
    else:
        tok = GPT2Tokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
