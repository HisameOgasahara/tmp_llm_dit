"""FashionMNIST 조건부 생성: rectified flow (flow matching) 학습과 샘플링."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..train import EMA

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def load_fashion(root: str = "data", train: bool = True, device: str = "cpu"):
    """전체 데이터셋을 GPU 텐서로 올린다 (6만 장 × 28×28 이라 작음). 픽셀 범위 [-1, 1]."""
    from torchvision.datasets import FashionMNIST
    ds = FashionMNIST(root, train=train, download=True)
    x = ds.data.float().div(255).mul(2).sub(1).unsqueeze(1).to(device)
    return x, ds.targets.to(device)


def fm_loss(model: nn.Module, x1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x_t = (1-t)·noise + t·data, 모델이 속도 v = data - noise 를 예측.

    t 는 logit-normal 분포 (SD3): 중간 시점을 더 자주 학습한다.
    """
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.sigmoid(torch.randn(B, device=x1.device))
    tt = t.view(B, 1, 1, 1)
    xt = (1 - tt) * x0 + tt * x1
    v = model(xt, t * 1000, y)
    return F.mse_loss(v.float(), (x1 - x0).float())


@torch.no_grad()
def sample(model: nn.Module, y: torch.Tensor, noise: torch.Tensor, steps: int = 30, cfg: float = 2.0):
    """Euler ODE 로 noise(t=0) → data(t=1). cfg>1 이면 classifier-free guidance."""
    model.eval()
    null = model.config.n_classes
    x = noise.clone()
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        if cfg == 1.0:
            v = model(x, t * 1000, y)
        else:
            vc, vu = model(torch.cat([x, x]), torch.cat([t, t]) * 1000,
                           torch.cat([y, torch.full_like(y, null)])).chunk(2)
            v = vu + cfg * (vc - vu)
        x = x + v.float() / steps
    return x.clamp(-1, 1)


def train_dit(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int, batch: int = 256,
              lr: float = 3e-4, warmup: int = 500, ema_decay: float = 0.999, amp: bool = True,
              log_every: int = 100, snapshot_steps=(), snap_y=None, snap_noise=None,
              snap_cfg: float = 2.0, progress: bool = True):
    """반환: (history, snapshots{step: 이미지}, ema)"""
    from tqdm.auto import tqdm

    dev = x.device
    use_amp = amp and dev.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, ema_decay)
    hist = {"step": [], "loss": []}
    snaps = {}
    run_loss, n = 0.0, 0
    bar = tqdm(range(steps + 1), disable=not progress, mininterval=1.0)
    for step in bar:
        if step in snapshot_steps and snap_y is not None:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                snaps[step] = sample(ema.model, snap_y, snap_noise, cfg=snap_cfg).float().cpu()
        if step == steps:
            break
        model.train()
        idx = torch.randint(0, x.shape[0], (batch,), device=dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = fm_loss(model, x[idx], y[idx])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model, step)
        run_loss += loss.item()
        n += 1
        if (step + 1) % log_every == 0:
            hist["step"].append(step + 1)
            hist["loss"].append(run_loss / n)
            bar.set_postfix(loss=f"{run_loss / n:.4f}")
            run_loss, n = 0.0, 0
    return hist, snaps, ema


def to_grid(imgs: torch.Tensor, nrow: int, pad: int = 2):
    """[N, 1, H, W] (-1..1) → numpy [H', W'] (0..1)"""
    from torchvision.utils import make_grid
    g = make_grid(imgs.float().cpu().add(1).div(2), nrow=nrow, padding=pad, pad_value=1.0)
    return g[0].numpy()
