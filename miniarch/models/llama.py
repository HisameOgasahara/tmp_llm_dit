"""A1 · Llama: GQA + RoPE + RMSNorm(Pre-norm) + SwiGLU. 이후 단계들의 '현대 표준' 베이스라인.

파라미터 이름은 HF 와 같게 맞췄다 (이후 Qwen/DeepSeek 등도 이 명명을 따른다).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput, RMSNorm, apply_rope, attention, rope_cos_sin, rope_inv_freq


@dataclass
class LlamaConfig:
    vocab_size: int = 128
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2
    d_ff: int = 704
    head_dim: int | None = None
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self):
        self.head_dim = self.head_dim or self.d_model // self.n_head


class Attention(nn.Module):
    def __init__(self, c: LlamaConfig):
        super().__init__()
        self.h, self.kv, self.d = c.n_head, c.n_kv_head, c.head_dim
        self.q_proj = nn.Linear(c.d_model, self.h * self.d, bias=False)
        self.k_proj = nn.Linear(c.d_model, self.kv * self.d, bias=False)
        self.v_proj = nn.Linear(c.d_model, self.kv * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, c.d_model, bias=False)
        self.attn_probs = nn.Identity()  # hook 지점: [B, H, T, T]

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv, self.d).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # GQA: KV head 하나를 query head 여러 개가 공유
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = attention(q, k, v, causal=True, probs_hook=self.attn_probs)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, c: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.up_proj = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down_proj = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))  # SwiGLU


class Block(nn.Module):
    def __init__(self, c: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(c.d_model, c.rms_eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c.d_model, c.rms_eps)
        self.mlp = MLP(c)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Llama(nn.Module):
    def __init__(self, c: LlamaConfig):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model, c.rms_eps)
        self.lm_head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.register_buffer("inv_freq", rope_inv_freq(c.head_dim, c.rope_theta), persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> LMOutput:
        T = input_ids.shape[1]
        cos, sin = rope_cos_sin(torch.arange(T, device=input_ids.device), self.inv_freq)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        h = self.norm(x)
        logits = self.lm_head(h) if self.lm_head is not None else h @ self.embed_tokens.weight.T
        return LMOutput(h, logits)


# ---------------------------------------------------------------- 검증

def _rope_theta(cfg) -> float:
    if getattr(cfg, "rope_theta", None):
        return float(cfg.rope_theta)
    rp = getattr(cfg, "rope_parameters", None) or {}
    return float(rp.get("rope_theta", 10000.0))


def verify_pair(ref) -> dict:
    """ref = transformers.LlamaModel (이름이 같으므로 name_map 불필요)"""
    c = ref.config
    mine = Llama(LlamaConfig(
        vocab_size=c.vocab_size, d_model=c.hidden_size, n_layer=c.num_hidden_layers,
        n_head=c.num_attention_heads, n_kv_head=c.num_key_value_heads, d_ff=c.intermediate_size,
        head_dim=getattr(c, "head_dim", None), rope_theta=_rope_theta(c), rms_eps=c.rms_norm_eps))
    return dict(mine=mine, name_map=None,
                hook_pairs=[(f"layers.{i}", f"layers.{i}") for i in range(c.num_hidden_layers)])
