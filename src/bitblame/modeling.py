"""Model handling — loading, unit enumeration, forward passes.

Supports two backends through one interface:
  * "tiny"  — small randomly-initialized decoder-only transformer in torch
              (Phase A plumbing; HF-style attribute layout so the same
              unit-enumeration code runs on both).
  * "hf"    — a real Hugging Face checkpoint loaded with transformers
              (Phase B: SmolLM2-135M), fp16 on CPU.

Units (spec §2.2, §4.1): default unit = one transformer block's Linear
weights. embed_tokens and lm_head are their own units. Norm scales are
kept at high precision (unperturbed), matching real-quantizer practice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ---------------------------------------------------------------- units

@dataclass
class Unit:
    id: str
    weight_names: list[str]          # dotted attribute paths to weight tensors
    gguf_names: list[str] = field(default_factory=list)  # llama.cpp tensor names
    n_params: int = 0


_BLOCK_LINEARS = [
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_output"),
    ("mlp.gate_proj", "ffn_gate"),
    ("mlp.up_proj", "ffn_up"),
    ("mlp.down_proj", "ffn_down"),
]


def _resolve(root: nn.Module, dotted: str) -> torch.nn.Parameter:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def list_units(model: nn.Module, n_layers: int, granularity: str = "block") -> list[Unit]:
    """Enumerate attribution units. granularity='block' (spec default).

    If lm_head is tied to embed_tokens (same tensor, e.g. SmolLM2), the
    two roles merge into one "embed" unit (perturbing the shared tensor
    once); gguf_names carries both tensor names for recipe emission.
    """
    units: list[Unit] = []
    if granularity == "block":
        for i in range(n_layers):
            wnames, gnames, n = [], [], 0
            for attr, gguf_role in _BLOCK_LINEARS:
                dotted = f"model.layers.{i}.{attr}.weight"
                p = _resolve(model, dotted)
                wnames.append(dotted)
                gnames.append(f"blk.{i}.{gguf_role}.weight")
                n += p.numel()
            units.append(Unit(id=f"blk.{i}", weight_names=wnames,
                              gguf_names=gnames, n_params=n))
        # embed + lm_head as their own units (§4.1 roles)
        pe = _resolve(model, "model.embed_tokens.weight")
        gnames = ["token_embd.weight"]
        try:
            ph = _resolve(model, "lm_head.weight")
            tied = ph is pe
        except AttributeError:
            tied = False
        if tied:
            gnames.append("output.weight")  # same tensor; one unit
        units.append(Unit(id="embed", weight_names=["model.embed_tokens.weight"],
                          gguf_names=gnames, n_params=pe.numel()))
        if not tied:
            units.append(Unit(id="lm_head", weight_names=["lm_head.weight"],
                              gguf_names=["output.weight"], n_params=ph.numel()))
    elif granularity == "fine":
        for i in range(n_layers):
            for attr, gguf_role in _BLOCK_LINEARS:
                dotted = f"model.layers.{i}.{attr}.weight"
                p = _resolve(model, dotted)
                units.append(Unit(id=f"blk.{i}.{attr.split('.')[-1]}",
                                  weight_names=[dotted],
                                  gguf_names=[f"blk.{i}.{gguf_role}.weight"],
                                  n_params=p.numel()))
    else:
        raise ValueError(f"unknown granularity {granularity!r}")
    return units


# ---------------------------------------------------------------- tiny model (Phase A)

class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class _TinyBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dff: int):
        super().__init__()
        self.input_layernorm = _RMSNorm(d)
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Linear(d, d, bias=False),
            "k_proj": nn.Linear(d, d, bias=False),
            "v_proj": nn.Linear(d, d, bias=False),
            "o_proj": nn.Linear(d, d, bias=False),
        })
        self.post_attention_layernorm = _RMSNorm(d)
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(d, dff, bias=False),
            "up_proj": nn.Linear(d, dff, bias=False),
            "down_proj": nn.Linear(dff, d, bias=False),
        })
        self.n_heads = n_heads
        self.head_dim = d // n_heads

    def forward(self, x):
        B, T, D = x.shape
        h = self.input_layernorm(x)

        def proj(m, t):
            p = m(t)
            return p.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = (proj(self.self_attn[n], h) for n in ("q_proj", "k_proj", "v_proj"))
        att = torch.softmax(q @ k.transpose(-1, -2) / (self.head_dim ** 0.5), dim=-1) @ v
        att = att.transpose(1, 2).reshape(B, T, D)
        x = x + self.self_attn["o_proj"](att)
        h2 = self.post_attention_layernorm(x)
        # SiLU-gated MLP (matches SmolLM2 form; prototype used ReLU)
        x = x + self.mlp["down_proj"](
            torch.nn.functional.silu(self.mlp["gate_proj"](h2)) * self.mlp["up_proj"](h2))
        return x


class _TinyOut:
    def __init__(self, logits, hidden):
        self.logits = logits
        self.hidden_states = (hidden,)


class TinyTransformer(nn.Module):
    """Random decoder-only transformer with HF-style attribute layout."""

    def __init__(self, vocab=256, d=64, n_heads=4, n_layers=4, dff=128, seed=7):
        super().__init__()
        torch.manual_seed(seed)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, d)
        self.model.layers = nn.ModuleList(
            [_TinyBlock(d, n_heads, dff) for _ in range(n_layers)])
        self.model.norm = _RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.config_n_layers = n_layers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0, 0.05)
        # inference-only measurement model: never trained, so gradients
        # are off — weight mutation in the scan (lolo) is in-place.
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        x = self.model.embed_tokens(input_ids)
        for blk in self.model.layers:
            x = blk(x)
        x = self.model.norm(x)
        return _TinyOut(self.lm_head(x), x)


# ---------------------------------------------------------------- HF model (Phase B)

def load_hf_model(model_dir: str, dtype=torch.float16) -> nn.Module:
    """Load a local HF checkpoint on CPU (no downloads — files pre-fetched)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, low_cpu_mem_usage=True,
        local_files_only=True, trust_remote_code=False,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def forward_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """(B, T, V) logits; fp16 preserved (no upcast — memory bound)."""
    return model(input_ids).logits


@torch.no_grad()
def forward_with_hidden(
    model: nn.Module, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """((B,T,V) logits, (B,T,D) final hidden states) for outlier-token tagging."""
    out = model(input_ids, output_hidden_states=True)
    return out.logits, out.hidden_states[-1]
