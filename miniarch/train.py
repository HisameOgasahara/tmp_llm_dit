"""학습 공용 유틸."""
from __future__ import annotations

import copy
import json
import os
import random

import numpy as np
import torch
from torch import nn


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


class EMA:
    """가중치 지수이동평균. 초반에는 decay 를 낮춰 빨리 따라가게 한다."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.lerp_(p.detach(), 1 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)
