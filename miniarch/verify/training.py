"""학습 동작 확인: forward 가 원본과 같아도 backward/학습이 되는지는 따로 봐야 한다.

1. grad_equivalence : 같은 가중치·같은 손실에서 원본과 내 구현의 파라미터 기울기가 같은가
                      (in-place 연산, detach, 잘못 끊긴 그래프, 커스텀 커널 대체의 backward 오류를 잡는다)
2. overfit_batch    : 내 구현이 고정된 배치 하나를 외울 수 있는가 (손실이 충분히 내려가는가)
                      + NaN/inf 없음, 기울기를 한 번도 못 받는 파라미터 목록
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn

from .equivalence import _first_tensor, _resolve, copy_weights, force_eager


def _to(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to(v, device) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(_to(v, device) for v in x)
    return x


def _call(m, x):
    return m(**x) if isinstance(x, dict) else m(*x)


def _mine_inputs(pair: dict, inputs):
    if pair.get("mine_inputs") is not None:
        return pair["mine_inputs"]
    if pair.get("mine_input_fn") is not None:
        return pair["mine_input_fn"](inputs)
    return inputs


@dataclass
class GradReport:
    ok: bool
    compared: int
    max_rel: float
    worst: str
    skipped_ref: list[str] = field(default_factory=list)   # 원본에서 기울기가 없는 파라미터 (requires_grad=False 등)
    note: str = ""


def grad_equivalence(ref: nn.Module, pair: dict, inputs, device: str, rtol: float = 1e-3) -> GradReport:
    """손실 = Σ(출력 · 고정 난수) 로 두 모델을 backward 하고 파라미터 기울기를 비교한다.

    상대 오차 = |g_ref − g_mine|.max / (|g_ref|.max + 1e-12), 파라미터별 최댓값이 rtol 이하이면 통과.
    """
    mine = pair["mine"]
    ref = ref.to(device).float().train()
    mine = mine.to(device).float().train()
    force_eager(ref)
    rep = copy_weights(ref, mine, pair.get("name_map"), pair.get("transforms"))
    if not rep.ok:
        return GradReport(False, 0, float("nan"), "", note=f"가중치 복사 실패: {rep}")
    x_ref = _to(inputs, device)
    x_mine = _to(_mine_inputs(pair, x_ref), device)
    out_r = pair.get("out_fn_ref", _first_tensor)(_call(ref, x_ref))
    out_m = pair.get("out_fn_mine", _first_tensor)(_call(mine, x_mine))
    if not out_r.requires_grad:
        return GradReport(False, 0, float("nan"), "", note="원본 출력에 기울기가 없음 (추론 전용 코드: no_grad/inference_mode)")
    g = torch.randn(out_r.shape, generator=torch.Generator().manual_seed(7)).to(device)
    (out_r.float() * g).sum().backward()
    (out_m.float() * g).sum().backward()

    mine_params = dict(mine.named_parameters())
    transformed = set((pair.get("transforms") or {}).keys())
    worst, max_rel, n, skipped = "", 0.0, 0, []
    for name, p in ref.named_parameters():
        mname = _resolve(pair.get("name_map"), name)
        if mname is None or mname not in mine_params or mname in transformed:
            continue
        if p.grad is None:
            skipped.append(name)
            continue
        gm = mine_params[mname].grad
        if gm is None:
            return GradReport(False, n, float("inf"), f"{name} -> {mname}", skipped, "내 구현에서 기울기가 안 흐름")
        rel = ((p.grad - gm).abs().max() / (p.grad.abs().max() + 1e-12)).item()
        n += 1
        if rel > max_rel:
            max_rel, worst = rel, name
    return GradReport(max_rel <= rtol and n > 0, n, max_rel, worst, skipped)


@dataclass
class OverfitReport:
    ok: bool
    loss_start: float
    loss_end: float
    kind: str
    finite: bool
    no_grad_params: list[str]
    curve: list[float]


def overfit_batch(mine: nn.Module, pair: dict, inputs, device: str, steps: int = 300, lr: float = 3e-3,
                  target_ratio: float = 0.2) -> OverfitReport:
    """고정 배치 하나를 외우는지. LM(logits + input_ids)은 다음 토큰 cross-entropy, 그 외는 고정 난수 목표에 대한 MSE.

    통과: 마지막 손실 ≤ target_ratio × 처음 손실, 그리고 모든 손실·기울기가 유한.
    """
    mine = copy.deepcopy(mine).to(device).float().train()
    x = _to(_mine_inputs(pair, _to(inputs, device)), device)
    out_fn = pair.get("out_fn_mine", _first_tensor)
    ids = x.get("input_ids") if isinstance(x, dict) else (x[0] if isinstance(x, (tuple, list)) and torch.is_tensor(x[0])
                                                          and x[0].dtype == torch.long else None)
    with torch.no_grad():
        probe = _call(mine, x)
    use_ce = ids is not None and getattr(probe, "logits", None) is not None
    target = None if use_ce else torch.randn(out_fn(probe).shape, generator=torch.Generator().manual_seed(3)).to(device)

    params = [p for p in mine.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    curve, finite, touched = [], True, set()
    names = {id(p): n for n, p in mine.named_parameters()}
    for _ in range(steps):
        out = _call(mine, x)
        if use_ce:
            logits = out.logits[:, :-1].float()
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        else:
            loss = nn.functional.mse_loss(out_fn(out).float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in params:
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    finite = False
                if p.grad.abs().sum() > 0:
                    touched.add(id(p))
        if not torch.isfinite(loss):
            finite = False
            curve.append(float("nan"))
            break
        opt.step()
        curve.append(loss.item())
    no_grad = sorted(names[id(p)] for p in params if id(p) not in touched)
    ok = finite and curve[-1] <= target_ratio * curve[0]
    return OverfitReport(ok, curve[0], curve[-1], "next-token CE" if use_ce else "MSE(고정 목표)", finite, no_grad, curve)
