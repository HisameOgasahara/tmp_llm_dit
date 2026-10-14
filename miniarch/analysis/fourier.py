"""정수 0..p-1 위의 푸리에 분석 (modular addition 해석용)."""
from __future__ import annotations

import math

import torch


def fourier_basis(p: int, device=None) -> tuple[torch.Tensor, list[str]]:
    """정규직교 기저 [p, p]: 상수, cos k, sin k (k = 1..p//2)."""
    x = torch.arange(p, dtype=torch.float32)
    rows, names = [torch.ones(p)], ["const"]
    for k in range(1, p // 2 + 1):
        rows += [torch.cos(2 * math.pi * k * x / p), torch.sin(2 * math.pi * k * x / p)]
        names += [f"cos{k}", f"sin{k}"]
    B = torch.stack(rows)
    B = B / B.norm(dim=1, keepdim=True)
    return B.to(device) if device is not None else B, names


def freq_energy(W: torch.Tensor, p: int) -> torch.Tensor:
    """W [p, d] (숫자별 임베딩) → 주파수 k=1..p//2 별 에너지 [p//2]."""
    B, _ = fourier_basis(p, W.device)
    C = B @ W.float()                      # 각 기저 방향 성분 [p, d]
    e = C.pow(2).sum(dim=1)                # 기저별 에너지
    return e[1:].view(-1, 2).sum(dim=1)    # cos k + sin k


def topk_energy_frac(W: torch.Tensor, p: int, k: int = 5) -> float:
    """상위 k 개 주파수가 차지하는 에너지 비율. grokking 후 1 에 가까워진다."""
    e = freq_energy(W, p)
    return (e.topk(k).values.sum() / e.sum()).item()


def fourier_2d(M: torch.Tensor) -> torch.Tensor:
    """[p, p] 격자 → 2D 푸리에 크기 [p//2+1, p//2+1] (평균 제거, 주파수 0..p//2 만).

    (ka, kb) 칸이 크면 '=' 의 attention 이 a 방향 주파수 ka, b 방향 주파수 kb 로 진동한다는 뜻.
    """
    p = M.shape[-1]
    F = torch.fft.fft2(M.float() - M.float().mean()).abs()
    return F[: p // 2 + 1, : p // 2 + 1]
