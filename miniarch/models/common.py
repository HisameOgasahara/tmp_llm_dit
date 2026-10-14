"""여러 모델이 공유하는 기본 부품."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LMOutput:
    last_hidden_state: torch.Tensor
    logits: torch.Tensor | None = None


class RMSNorm(nn.Module):
    """x / rms(x) * w. 정규화는 fp32 로 계산 (HF Llama 와 동일)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


# ---------------------------------------------------------------- RoPE

def rope_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))


def rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """positions [T] → cos, sin [T, head_dim]. 앞 절반/뒤 절반이 같은 주파수 (rotate_half 방식)."""
    freqs = positions.float()[:, None] * inv_freq.to(positions.device)[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, D], cos/sin [T, D]"""
    return x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)


# ---------------------------------------------------------------- attention

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool,
              probs_hook: nn.Module | None = None, scale: float | None = None) -> torch.Tensor:
    """명시적 softmax attention. q/k/v [B, H, T, D].

    probs_hook(nn.Identity) 를 통과시켜 attention 확률 [B, H, Tq, Tk] 을 hook 으로 캡처할 수 있다.
    """
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        tq, tk = q.shape[-2], k.shape[-2]
        mask = torch.ones(tq, tk, dtype=torch.bool, device=q.device).triu(tk - tq + 1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = scores.float().softmax(dim=-1).to(q.dtype)
    if probs_hook is not None:
        probs = probs_hook(probs)
    return probs @ v
