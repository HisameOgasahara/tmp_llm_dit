"""attention map 캡처와 시각화. 모든 모델의 `*.attn_probs` 지점을 사용한다."""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ..verify.equivalence import capture


def attn_point_names(model: nn.Module) -> list[str]:
    return [n for n, _ in model.named_modules() if n.endswith("attn_probs")]


def capture_attention(model: nn.Module, call: Callable[[], object]) -> dict[str, torch.Tensor]:
    """call() 을 실행하며 각 층의 attention 확률 [B, H, Tq, Tk] 을 모은다."""
    names = attn_point_names(model)
    with capture(model, names) as store, torch.no_grad():
        call()
    return {n: store[n].float().cpu() for n in names}


def grid_coords(g: int) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    return torch.stack([ys.flatten(), xs.flatten()], dim=1).float()


def mean_attention_distance(P: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """P [..., N, N] → 평균 attention 거리 [...]. 작으면 국소적, 크면 전역적."""
    d = torch.cdist(coords, coords)
    return (P * d).sum(-1).mean(-1)


def attention_entropy(P: torch.Tensor) -> torch.Tensor:
    """P [..., N, N] → 행별 엔트로피 평균 [...]. 작을수록 소수 토큰에 집중."""
    return -(P * (P + 1e-12).log()).sum(-1).mean(-1)


def plot_heads(P: torch.Tensor, labels_q=None, labels_k=None, title: str = "", vmax: float = 1.0):
    """P [H, Tq, Tk] → head 별 heatmap"""
    import matplotlib.pyplot as plt
    H = P.shape[0]
    fig, axes = plt.subplots(1, H, figsize=(2.4 * H, 2.6), squeeze=False)
    for h, ax in enumerate(axes[0]):
        ax.imshow(P[h], vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(f"head {h}", fontsize=9)
        if labels_k is not None:
            ax.set_xticks(range(len(labels_k)), labels_k, fontsize=8)
        if labels_q is not None:
            ax.set_yticks(range(len(labels_q)), labels_q, fontsize=8)
        for i in range(P.shape[1]):
            for j in range(P.shape[2]):
                if P.shape[1] * P.shape[2] <= 25:
                    ax.text(j, i, f"{P[h, i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="w" if P[h, i, j] < 0.6 * vmax else "k")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
