"""기존 태스크를 공용 학습 루프용 Task 로 감싼 것."""
from __future__ import annotations

import torch

from .base import IGNORE, LMTask
from .modadd import make_data


class ModAddTask(LMTask):
    """[a, b, =] → '=' 위치에서 (a + b) mod p. 학습/평가는 (a, b) 쌍이 겹치지 않게 나눈 고정 데이터."""
    name = "modadd"

    def __init__(self, p: int = 113, train_frac: float = 0.3, seed: int = 0):
        self.data = make_data(p, train_frac, seed)
        self.p = p

    def _pack(self, x, y):
        tgt = torch.full_like(x, IGNORE)
        tgt[:, -1] = y
        return {"input_ids": x, "targets": tgt}

    def batch(self, split, gen, batch_size):
        x, y = (self.data.train_x, self.data.train_y) if split == "train" else (self.data.test_x, self.data.test_y)
        i = torch.randint(0, len(x), (batch_size,), generator=gen)
        return self._pack(x[i], y[i])

    def eval_batches(self, n, batch_size, seed=1234):
        """평가는 표본 대신 테스트 전체."""
        x, y = self.data.test_x, self.data.test_y
        return [self._pack(x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(x), batch_size)]


class FlowMatchingTask:
    """DiT 용: x1 이미지·y 클래스로 flow matching 손실 (miniarch/tasks/fashion.fm_loss). 지표는 평가 손실."""
    name = "flow_matching"

    def __init__(self, x: torch.Tensor, y: torch.Tensor, eval_frac: float = 0.1, seed: int = 0):
        perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
        n = int(len(x) * eval_frac)
        self.ev, self.tr = perm[:n], perm[n:]
        self.x, self.y = x, y

    def batch(self, split, gen, batch_size):
        pool = self.tr if split == "train" else self.ev
        i = pool[torch.randint(0, len(pool), (batch_size,), generator=gen)]
        return {"x": self.x[i], "y": self.y[i], "noise_seed": int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen))}

    def loss(self, model, batch):
        from .fashion import fm_loss
        torch.manual_seed(batch["noise_seed"])      # t·노이즈를 배치 seed 에 묶어 재현 가능하게
        loss = fm_loss(model, batch["x"], batch["y"])
        return loss, {"fm": loss.detach()}

    @torch.no_grad()
    def metrics(self, model, batch):
        return {"fm": self.loss(model, batch)[0].item()}

    def eval_batches(self, n, batch_size, seed=1234):
        g = torch.Generator().manual_seed(seed)
        return [self.batch("eval", g, batch_size) for _ in range(n)]
