"""태스크 공통 인터페이스.

태스크가 정하는 것: 배치 만들기(split 별, seed 고정 가능), 손실, 평가 지표.
학습 루프(miniarch/trainer.py)는 모델·태스크와 무관하게 하나만 쓴다.

LM 태스크 규약
- batch = {"input_ids": [B, T] long, "targets": [B, T] long}
- targets[t] = 위치 t 에서 맞혀야 할 다음 토큰, 손실을 안 매기는 위치는 IGNORE(-100)
- 모델은 forward(input_ids) → .logits [B, T, V] (LMOutput)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IGNORE = -100


def model_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids)
    return out.logits if hasattr(out, "logits") else out


class Task:
    name = "task"
    splits = ("train", "eval")

    def batch(self, split: str, gen: torch.Generator, batch_size: int) -> dict:
        raise NotImplementedError

    def loss(self, model: nn.Module, batch: dict) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @torch.no_grad()
    def metrics(self, model: nn.Module, batch: dict) -> dict:
        raise NotImplementedError

    def eval_batches(self, n: int, batch_size: int, seed: int = 1234) -> list[dict]:
        """평가용 고정 배치 (학습 생성기와 다른 seed 흐름)."""
        g = torch.Generator().manual_seed(seed)
        return [self.batch("eval", g, batch_size) for _ in range(n)]


class LMTask(Task):
    """targets 가 표시된 위치만 cross-entropy. 모델에 kl_loss(MSA indexer 등)가 있으면 kl_coef 배로 더한다."""
    kl_coef = 1.0

    def loss(self, model, batch):
        logits = model_logits(model, batch["input_ids"])
        tgt = batch["targets"]
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=IGNORE)
        logs = {"ce": ce.detach()}
        kl = getattr(model, "kl_loss", None)
        if isinstance(kl, torch.Tensor):
            logs["kl"] = kl.detach()
            ce = ce + self.kl_coef * kl
        return ce, logs

    @torch.no_grad()
    def metrics(self, model, batch):
        logits = model_logits(model, batch["input_ids"])
        tgt = batch["targets"]
        m = tgt != IGNORE
        pred = logits.argmax(-1)
        ce = F.cross_entropy(logits.float()[m], tgt[m])
        return {"acc": (pred[m] == tgt[m]).float().mean().item(), "ce": ce.item()}
