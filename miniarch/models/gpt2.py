"""A0 · GPT-2: MHA + 학습형 위치 임베딩 + Pre-LayerNorm + GELU(tanh) MLP."""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput, attention


@dataclass
class GPT2Config:
    vocab_size: int = 128
    n_positions: int = 256
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    ln_eps: float = 1e-5
    dropout: float = 0.0


class Attention(nn.Module):
    def __init__(self, c: GPT2Config):
        super().__init__()
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.attn_probs = nn.Identity()  # hook 지점: [B, H, T, T]
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, -1).transpose(1, 2) for t in (q, k, v))
        y = attention(q, k, v, causal=True, probs_hook=self.attn_probs)
        return self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class MLP(nn.Module):
    def __init__(self, c: GPT2Config):
        super().__init__()
        self.fc = nn.Linear(c.d_model, 4 * c.d_model)
        self.proj = nn.Linear(4 * c.d_model, c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, c: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model, eps=c.ln_eps)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.d_model, eps=c.ln_eps)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT2(nn.Module):
    def __init__(self, c: GPT2Config):
        super().__init__()
        self.config = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_emb = nn.Embedding(c.n_positions, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model, eps=c.ln_eps)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> LMOutput:
        T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for b in self.blocks:
            x = b(x)
        h = self.ln_f(x)
        return LMOutput(h, h @ self.tok_emb.weight.T)  # 출력층은 임베딩과 공유


# ---------------------------------------------------------------- 검증

_RENAME = [
    (r"^wte\.", "tok_emb."), (r"^wpe\.", "pos_emb."), (r"^h\.(\d+)\.", r"blocks.\1."),
    (r"\.ln_1\.", ".ln1."), (r"\.ln_2\.", ".ln2."),
    (r"\.attn\.c_attn\.", ".attn.qkv."), (r"\.attn\.c_proj\.", ".attn.proj."),
    (r"\.mlp\.c_fc\.", ".mlp.fc."), (r"\.mlp\.c_proj\.", ".mlp.proj."),
]


def hf_name_map(name: str) -> str | None:
    # 구버전 HF 의 causal mask 버퍼 (h.N.attn.bias). c_attn.bias 는 진짜 파라미터라 제외하면 안 됨
    if re.search(r"\.attn\.(bias|masked_bias)$", name):
        return None
    for a, b in _RENAME:
        name = re.sub(a, b, name)
    return name


def verify_pair(ref) -> dict:
    """ref = transformers.GPT2Model"""
    c = ref.config
    mine = GPT2(GPT2Config(vocab_size=c.vocab_size, n_positions=c.n_positions, d_model=c.n_embd,
                           n_layer=c.n_layer, n_head=c.n_head, ln_eps=c.layer_norm_epsilon))
    # HF GPT-2 는 Linear 대신 Conv1D(weight [in, out]) → 전치해서 복사
    transforms = {k: (lambda t: t.t()) for k in mine.state_dict()
                  if re.search(r"\.(qkv|proj|fc)\.weight$", k)}
    return dict(mine=mine, name_map=hf_name_map, transforms=transforms,
                hook_pairs=[(f"h.{i}", f"blocks.{i}") for i in range(c.n_layer)])
