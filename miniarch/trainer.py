"""공용 학습 루프 — 모든 모델·태스크가 이것 하나로 학습하고, 같은 이름으로 기록한다.

기록 (TensorBoard + metrics.jsonl, 태그 이름은 모든 모델에서 같다 → 한 차트에 겹쳐 비교)
  train/loss, train/<태스크 항목>, train/grad_norm, train/lr, train/step_ms
  eval/<지표>                                   태스크가 정의 (acc, 거리별 acc, ce …)
  attn/entropy/L{i}, attn/distance/L{i}          모델의 attn_probs hook 지점에서 자동 수집
  sink/mass/L{i}                                 sink_probe 지점
  moe/load_cv/{모듈}                              route_probe 지점 (expert 부하 변동계수, 0 = 고르게 분산)
hook 지점이 없는 모델은 해당 항목이 그냥 안 생긴다.

run 폴더 = runs/<태스크>/<모델>/<변형>/seed<k>  (TensorBoard 에서 정규식으로 골라 겹쳐 본다)

학습 전용 동작
- aux-loss-free 라우터 bias 업데이트(RouterBalancer): route_probe 와 bias(e_score_correction_bias / router_bias /
  Gate.bias)를 가진 모든 라우터에 같은 규칙 적용. bias 는 optimizer 에서 뺀다 (기울기·weight decay 대상이 아님).
- weight decay 는 2차원 이상 가중치에만 (norm·bias·sink·스칼라 제외).
- 체크포인트에 모델·optimizer·step·RNG 상태를 모두 저장 → 이어서 학습해도 처음부터 한 것과 같다.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

BIAS_NAMES = ("e_score_correction_bias", "router_bias", "bias")


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 64
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 4
    log_every: int = 20
    mi_every: int = 200                 # MI 지표 수집 주기 (0 = 끔)
    ckpt_every: int = 0                 # 0 = 마지막에만
    seed: int = 0
    run_dir: str = "runs/debug"
    bias_update_rate: float = 1e-3      # aux-loss-free 라우터 bias 업데이트 속도 (0 = 끔)
    tensorboard: bool = True


# ---------------------------------------------------------------- 라우터 bias (aux-loss-free 부하 분산)

class RouterBalancer:
    """route_probe 로 직전 forward 의 expert 선택을 모아, 적게 뽑힌 expert 의 bias 를 올리고 많이 뽑힌 쪽은 내린다.

        b_e ← b_e + rate · sign(mean_load − load_e)        (DeepSeek-V3 aux-loss-free)
    """

    def __init__(self, model: nn.Module):
        self.routers = []          # (이름, 모듈, bias 텐서)
        self._idx, self._hooks = {}, []
        for name, m in model.named_modules():
            probe = getattr(m, "route_probe", None)
            if probe is None:
                continue
            for bn in BIAS_NAMES:
                b = getattr(m, bn, None)
                if isinstance(b, torch.Tensor) and b.dim() == 1 and b.numel() > 1:
                    self.routers.append((name, m, b))
                    self._hooks.append(probe.register_forward_hook(
                        lambda _m, _i, out, name=name: self._idx.__setitem__(name, out.detach())))
                    break

    def bias_params(self) -> set:
        return {id(b) for _, _, b in self.routers if isinstance(b, nn.Parameter)}

    @torch.no_grad()
    def update(self, rate: float) -> dict:
        logs = {}
        for name, _, b in self.routers:
            idx = self._idx.pop(name, None)
            if idx is None:
                continue
            load = torch.bincount(idx.flatten().long(), minlength=b.numel()).float()
            if rate:
                b += rate * torch.sign(load.mean() - load).to(b.dtype)
            logs[name] = (load.std() / (load.mean() + 1e-9)).item()
        return logs

    def remove(self):
        for h in self._hooks:
            h.remove()


# ---------------------------------------------------------------- MI 지표 자동 수집

class MIProbe:
    """attn_probs / sink_probe / route_probe 지점 출력을 한 번의 forward 동안 모아 요약한다."""

    def __init__(self, model: nn.Module):
        self.points = [(n, m) for n, m in model.named_modules()
                       if n.endswith(("attn_probs", "sink_probe")) and isinstance(m, nn.Identity)]

    @torch.no_grad()
    def collect(self, call) -> dict:
        store, hooks = {}, []
        for n, m in self.points:
            hooks.append(m.register_forward_hook(lambda _m, _i, out, n=n: store.__setitem__(n, out.detach().float())))
        try:
            call()
        finally:
            for h in hooks:
                h.remove()
        logs, li = {}, 0
        for n, _ in self.points:
            t = store.get(n)
            if t is None:
                continue
            if n.endswith("attn_probs") and t.dim() == 4:
                P = t.clamp_min(1e-12)
                logs[f"attn/entropy/L{li}"] = (-(P * P.log()).sum(-1)).mean().item()
                Tq, Tk = P.shape[-2], P.shape[-1]
                if Tk >= Tq:                                     # 앞 Tq 칸이 토큰 위치인 경우만 거리 계산
                    q = torch.arange(Tq, device=P.device).float()[:, None]
                    k = torch.arange(Tq, device=P.device).float()[None]
                    logs[f"attn/distance/L{li}"] = (P[..., :Tq] * (q - k).abs()).sum(-1).mean().item()
                li += 1
            elif n.endswith("sink_probe"):
                logs[f"sink/mass/{n.rsplit('.', 2)[0]}"] = t.mean().item()
        return logs


# ---------------------------------------------------------------- 유틸

def lr_at(step: int, cfg: TrainConfig) -> float:
    """선형 warmup 후 cosine 으로 min_lr 까지."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    prog = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def param_groups(model: nn.Module, weight_decay: float, exclude: set) -> list[dict]:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in exclude:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay, "name": "decay"},
            {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"}]


def _to(batch: dict, device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- 학습 루프

class Trainer:
    def __init__(self, model: nn.Module, task, cfg: TrainConfig, device: str | None = None):
        self.cfg, self.task = cfg, task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(cfg.seed)
        self.model = model.to(self.device)
        self.balancer = RouterBalancer(self.model)
        self.mi = MIProbe(self.model)
        self.opt = torch.optim.AdamW(param_groups(self.model, cfg.weight_decay, self.balancer.bias_params()),
                                     lr=cfg.lr, betas=cfg.betas)
        self.gen = torch.Generator().manual_seed(cfg.seed)          # 학습 배치 전용
        self.step = 0
        self.eval_set = task.eval_batches(cfg.eval_batches, cfg.batch_size)
        self.history: list[dict] = []
        os.makedirs(cfg.run_dir, exist_ok=True)
        self.writer = None
        if cfg.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(cfg.run_dir)

    # -- 기록
    def log(self, logs: dict, step: int | None = None) -> None:
        step = self.step if step is None else step
        rec = {"step": step, **{k: float(v) for k, v in logs.items()}}
        self.history.append(rec)
        with open(os.path.join(self.cfg.run_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self.writer:
            for k, v in logs.items():
                self.writer.add_scalar(k, float(v), step)

    # -- 한 step
    def train_step(self) -> dict:
        cfg, m = self.cfg, self.model
        m.train()
        for g in self.opt.param_groups:
            g["lr"] = lr_at(self.step, cfg)
        batch = _to(self.task.batch("train", self.gen, cfg.batch_size), self.device)
        t0 = time.time()
        loss, parts = self.task.loss(m, batch)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]],
                                            cfg.grad_clip if cfg.grad_clip else float("inf"))
        self.opt.step()
        loads = self.balancer.update(cfg.bias_update_rate)
        if self.device == "cuda":
            torch.cuda.synchronize()
        logs = {"train/loss": loss.item(), "train/grad_norm": gn.item(), "train/lr": self.opt.param_groups[0]["lr"],
                "train/step_ms": 1000 * (time.time() - t0)}
        logs.update({f"train/{k}": float(v) for k, v in parts.items()})
        logs.update({f"moe/load_cv/{k}": v for k, v in loads.items()})
        self.step += 1
        return logs

    # -- 평가 (가중치·RNG·train 모드를 건드리지 않는다)
    @torch.no_grad()
    def evaluate(self) -> dict:
        was = self.model.training
        self.model.eval()
        acc: dict[str, list] = {}
        for b in self.eval_set:
            for k, v in self.task.metrics(self.model, _to(b, self.device)).items():
                acc.setdefault(k, []).append(v)
        self.model.train(was)
        return {f"eval/{k}": sum(v) / len(v) for k, v in acc.items()}

    @torch.no_grad()
    def mi_metrics(self) -> dict:
        if not self.mi.points or "input_ids" not in self.eval_set[0]:
            return {}
        was = self.model.training
        self.model.eval()
        ids = self.eval_set[0]["input_ids"][:8].to(self.device)
        out = self.mi.collect(lambda: self.model(ids))
        self.model.train(was)
        return out

    def train(self, steps: int | None = None, progress: bool = True) -> list[dict]:
        cfg = self.cfg
        end = self.step + (steps if steps is not None else cfg.steps - self.step)
        from tqdm.auto import tqdm
        bar = tqdm(total=end - self.step, disable=not progress, mininterval=1.0)
        while self.step < end:
            logs = self.train_step()
            if self.step % cfg.log_every == 0 or self.step == end:
                if not math.isfinite(logs["train/loss"]):
                    self.log(logs)
                    raise FloatingPointError(f"step {self.step}: 손실이 {logs['train/loss']}")
                if cfg.eval_every and (self.step % cfg.eval_every == 0 or self.step == end):
                    logs.update(self.evaluate())
                if cfg.mi_every and self.step % cfg.mi_every == 0:
                    logs.update(self.mi_metrics())
                self.log(logs)
                bar.set_postfix({k.split("/")[-1]: f"{v:.3g}" for k, v in logs.items()
                                 if k in ("train/loss", "eval/acc", "eval/fm")})
            if cfg.ckpt_every and self.step % cfg.ckpt_every == 0:
                self.save()
            bar.update(1)
        bar.close()
        self.save()
        if self.writer:
            self.writer.flush()
        return self.history

    # -- 체크포인트
    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(self.cfg.run_dir, "ckpt.pt")
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "step": self.step,
                    "gen": self.gen.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                    "config": asdict(self.cfg)}, path)
        return path

    def load(self, path: str | None = None) -> None:
        path = path or os.path.join(self.cfg.run_dir, "ckpt.pt")
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.step = ck["step"]
        self.gen.set_state(ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        if ck["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        random.setstate(ck["py_rng"])
        np.random.set_state(ck["np_rng"])
