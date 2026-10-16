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

from .equivalence import _first_tensor, _resolve, copy_weights, force_eager, perturb_params


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


def _ref_call(ref: nn.Module, x):
    """원본 forward. 학습 모드를 거부하는 추론 코드(예: MiMo noaux_tc, Kimi)는 eval 모드로 다시 부른다
    (작은 config 에 dropout 이 없어 값은 같고, autograd 는 eval 모드에서도 그대로 동작한다)."""
    try:
        return _call(ref, x)
    except Exception:
        if not ref.training:
            raise
        ref.eval()
        return _call(ref, x)


def _prepare(ref: nn.Module, pair: dict, inputs, device: str, copy_mine: bool):
    """원본·내 구현을 같은 가중치로 맞추고 입력을 준비한다. 원본은 복사하지 않는다:
    일부 원본은 forward 를 인스턴스 closure 로 감싸 두어 deepcopy 한 사본이 여전히 원래 모델을 부른다."""
    mine = copy.deepcopy(pair["mine"]) if copy_mine else pair["mine"]
    ref = ref.to(device).float().train()
    mine = mine.to(device).float().train()
    force_eager(ref)
    rep = copy_weights(ref, mine, pair.get("name_map"), pair.get("transforms"))
    x_ref = _to(inputs, device)
    x_mine = _to(_mine_inputs(pair, x_ref), device)
    return ref, mine, rep, x_ref, x_mine


def _mapped(ref: nn.Module, mine: nn.Module, pair: dict) -> list[tuple[str, str]]:
    mine_params = dict(mine.named_parameters())
    transformed = set((pair.get("transforms") or {}).keys())
    out = []
    for name, _ in ref.named_parameters():
        mname = _resolve(pair.get("name_map"), name)
        if mname is not None and mname in mine_params and mname not in transformed:
            out.append((name, mname))
    return out


@dataclass
class GradReport:
    ok: bool
    compared: int
    max_rel: float
    worst: str
    skipped_ref: list[str] = field(default_factory=list)   # 원본에서 기울기가 안 흐르는 파라미터 (추론 전용 경로 등)
    note: str = ""


INFERENCE_ONLY = "원본 출력에 기울기가 없음 (추론 전용 코드: no_grad/inference_mode)"
TRUNCATED = "원본 그래프 중간에서 기울기가 끊김 (일부 모듈이 추론 전용 경로) → 원본과 기울기 비교 불가"


def grad_equivalence(ref: nn.Module, pair: dict, inputs, device: str, rtol: float = 1e-3) -> GradReport:
    """손실 = Σ(출력 · 고정 난수) 로 두 모델을 backward 하고 파라미터 기울기를 비교한다.

    상대 오차 = |g_ref − g_mine|.max / (|g_ref|.max + 1e-12), 파라미터별 최댓값이 rtol 이하이면 통과.
    원본에서 기울기가 안 흐르는 파라미터(skipped_ref)는 비교하지 않고 따로 보여준다.
    """
    ref, mine, rep, x_ref, x_mine = _prepare(ref, pair, inputs, device, copy_mine=False)
    if not rep.ok:
        return GradReport(False, 0, float("nan"), "", note=f"가중치 복사 실패: {rep}")
    out_r = pair.get("out_fn_ref", _first_tensor)(_ref_call(ref, x_ref))
    if not ref.training:
        mine.eval()
    out_m = pair.get("out_fn_mine", _first_tensor)(_call(mine, x_mine))
    if not out_r.requires_grad:
        return GradReport(False, 0, float("nan"), "", note=INFERENCE_ONLY)
    g = torch.randn(out_r.shape, generator=torch.Generator().manual_seed(7)).to(device)
    (out_r.float() * g).sum().backward()
    (out_m.float() * g).sum().backward()

    rp, mp = dict(ref.named_parameters()), dict(mine.named_parameters())
    worst, max_rel, n, skipped = "", 0.0, 0, []
    for name, mname in _mapped(ref, mine, pair):
        p = rp[name]
        if p.grad is None:
            skipped.append(name)
            continue
        gm = mp[mname].grad
        if gm is None:
            return GradReport(False, n, float("inf"), f"{name} -> {mname}", skipped, "내 구현에서 기울기가 안 흐름")
        rel = ((p.grad - gm).abs().max() / (p.grad.abs().max() + 1e-12)).item()
        n += 1
        if rel > max_rel:
            max_rel, worst = rel, name
    note = "원본 eval 모드로 비교 (원본이 학습 모드를 거부)" if not ref.training else ""
    ok = max_rel <= rtol and n > 0
    # 원본에서는 기울기가 안 흐르는데 내 쪽은 흐르는 파라미터가 있으면, 원본 그래프가 그 지점에서 끊긴 것이고
    # 그보다 앞쪽 파라미터의 기울기도 원본과 비교할 수 없다 (예: Kimi 원본 MoE 는 eval 모드에서 no_grad 추론 경로)
    cut = [n_ for n_ in skipped if mp[_resolve(pair.get("name_map"), n_)].grad is not None
           and mp[_resolve(pair.get("name_map"), n_)].grad.abs().sum() > 0]
    if cut and not ok:
        return GradReport(False, n, max_rel, worst, skipped, TRUNCATED)
    return GradReport(ok, n, max_rel, worst, skipped, note)


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
    """고정 배치 하나를 외우는지. LM(logits + input_ids)은 다음 토큰 cross-entropy.
    그 외는 같은 구조에 가중치만 살짝 바꾼 teacher 의 출력을 목표로 한 MSE:
    난수 목표는 출력 일부가 가중치와 무관한 모델(입력이 그대로 통과하는 칸 등)에서 원리적으로 도달할 수 없다.

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
    target = None
    if not use_ce:
        teacher = copy.deepcopy(mine)
        perturb_params(teacher, std=0.05, seed=17)
        with torch.no_grad():
            target = out_fn(_call(teacher, x)).float().detach()
        del teacher

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
    return OverfitReport(ok, curve[0], curve[-1], "next-token CE" if use_ce else "MSE(teacher 출력)", finite, no_grad, curve)


@dataclass
class ParityReport:
    ok: bool
    steps: int
    loss_ref: list[float]
    loss_mine: list[float]
    max_loss_rel: float
    max_param_rel: float
    worst_param: str
    note: str = ""
    frozen: list[str] = field(default_factory=list)


def train_parity(ref: nn.Module, pair: dict, inputs, device: str, steps: int = 30, lr: float = 0.01,
                 rtol: float = 1e-3) -> ParityReport:
    """원본과 내 구현을 같은 가중치·같은 데이터·같은 optimizer 로 나란히 학습해 step 별 손실과 최종 가중치를 비교한다.

    손실 = MSE(출력, 고정 난수 목표). optimizer 는 SGD(momentum 0.9):
    Adam 은 기울기를 크기로 나눠서 1e-9 수준의 잡음 차이도 같은 크기의 업데이트 차이로 키우므로 궤적 비교에 맞지 않다.
    원본에서 기울기가 안 흐르는 파라미터(추론 전용 경로, requires_grad=False)는 양쪽 다 얼린다 (frozen).
    주의: 원본(ref)을 그대로 학습시키므로 호출 뒤 ref 의 가중치는 바뀐다.
    """
    ref, mine, rep, x_ref, x_mine = _prepare(ref, pair, inputs, device, copy_mine=True)
    if not rep.ok:
        return ParityReport(False, 0, [], [], float("nan"), float("nan"), "", f"가중치 복사 실패: {rep}")
    moe_note = ""
    if any(n.endswith("route_probe") for n, _ in mine.named_modules()):
        # MoE: 두 구현의 1e-7 수준 합산 순서 차이로 top-k 동점 근처 선택이 갈라지면 그 뒤 궤적은 원리적으로 달라진다
        steps = min(steps, 10)
        moe_note = "MoE → 라우팅이 갈라지기 전 10 step 만 비교"
    out_fn_r, out_fn_m = pair.get("out_fn_ref", _first_tensor), pair.get("out_fn_mine", _first_tensor)
    out0 = out_fn_r(_ref_call(ref, x_ref))
    if not ref.training:
        mine.eval()
    if not out0.requires_grad:
        return ParityReport(False, 0, [], [], float("nan"), float("nan"), "", INFERENCE_ONLY)
    target = torch.randn(out0.shape, generator=torch.Generator().manual_seed(5)).to(device)
    # 원본에서 기울기가 안 흐르는 파라미터 → 양쪽 다 얼림
    ref.zero_grad(set_to_none=True)
    nn.functional.mse_loss(out0.float(), target).backward()
    rp, mp = dict(ref.named_parameters()), dict(mine.named_parameters())
    mapped, frozen = [], []
    for rn, mn in _mapped(ref, mine, pair):
        if rp[rn].grad is None or not rp[rn].requires_grad:
            rp[rn].requires_grad_(False)
            mp[mn].requires_grad_(False)
            frozen.append(rn)
        else:
            mapped.append((rn, mn))
    ref.zero_grad(set_to_none=True)
    opt_r = torch.optim.SGD([p for p in ref.parameters() if p.requires_grad], lr=lr, momentum=0.9)
    opt_m = torch.optim.SGD([p for p in mine.parameters() if p.requires_grad], lr=lr, momentum=0.9)

    lr_, lm_ = [], []
    for _ in range(steps):
        loss_r = nn.functional.mse_loss(out_fn_r(_call(ref, x_ref)).float(), target)
        loss_m = nn.functional.mse_loss(out_fn_m(_call(mine, x_mine)).float(), target)
        for opt, loss in ((opt_r, loss_r), (opt_m, loss_m)):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        lr_.append(loss_r.item())
        lm_.append(loss_m.item())
        if not (torch.isfinite(loss_r) and torch.isfinite(loss_m)):
            return ParityReport(False, len(lr_), lr_, lm_, float("nan"), float("nan"), "",
                                "손실이 발산 (NaN/inf) — lr 을 낮춰 다시 확인", frozen)

    max_loss_rel = max(abs(a - b) / (abs(a) + 1e-12) for a, b in zip(lr_, lm_))
    worst, max_param_rel = "", 0.0
    with torch.no_grad():
        for rn, mn in mapped:
            a, b = rp[rn], mp[mn]
            rel = ((a - b).abs().max() / (a.abs().max() + 1e-12)).item()
            if rel > max_param_rel:
                max_param_rel, worst = rel, rn
    note = ("원본 eval 모드로 비교 (원본이 학습 모드를 거부) " if not ref.training else "") + moe_note
    if lr_[-1] >= lr_[0]:
        note += " 원본 손실이 줄지 않음"
    ok = max_loss_rel <= rtol and max_param_rel <= rtol and lr_[-1] < lr_[0]
    return ParityReport(ok, steps, lr_, lm_, max_loss_rel, max_param_rel, worst, note, frozen)


# ---------------------------------------------------------------- ④ 원본과 비교할 수 없는 부분

@dataclass
class FDReport:
    ok: bool
    rel_errors: list[float]
    eps: float
    note: str = ""


def directional_grad_check(mine: nn.Module, pair: dict, inputs, device: str, n_dirs: int = 3,
                           eps_list=(1e-2, 3e-3, 1e-3), rtol: float = 5e-2) -> FDReport:
    """수치 미분: 손실 L = mean(출력 · 고정 난수) 에 대해 무작위 방향 d 로
        (L(θ+εd) − L(θ−εd)) / 2ε   ==   <∇L, d>   (autograd)
    인지 확인한다. 원본이 추론 전용이라 ② 기울기 비교를 못 하는 모델에만 쓴다 (그 외에는 ②가 더 엄밀).

    - 모델 고유 dtype(fp32)으로 계산한다: 내부에서 fp32 로 고정해 계산하는 모델이 많아 float64 로 바꿔도 정밀도가 오르지 않는다.
    - 기울기를 받는 파라미터만 움직인다: indexer 처럼 일부러 기울기를 끊은 경로는 수치 미분과 다를 수밖에 없다.
    - fp32 잡음 때문에 ε 을 몇 개 시도하고 가장 잘 맞는 결과를 쓴다 (허용 상대 오차 5e-2).
    """
    m = copy.deepcopy(mine).to(device).float().train()
    x = _to(_mine_inputs(pair, _to(inputs, device)), device)
    out_fn = pair.get("out_fn_mine", _first_tensor)
    with torch.no_grad():
        shape = out_fn(_call(m, x)).shape
    g = torch.randn(shape, generator=torch.Generator().manual_seed(11)).to(device)

    def L():
        return (out_fn(_call(m, x)).float() * g).mean()

    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(L(), [p for _, p in named], allow_unused=True)
    live = [(p, gr) for (_, p), gr in zip(named, grads) if gr is not None and gr.abs().sum() > 0]
    if not live:
        return FDReport(False, [], 0.0, "기울기를 받는 파라미터가 없음")
    gen = torch.Generator().manual_seed(13)
    dirs = []
    for _ in range(n_dirs):
        d = [torch.randn(p.shape, generator=gen).to(device) for p, _ in live]
        norm = torch.sqrt(sum((v * v).sum() for v in d))
        dirs.append([v / norm for v in d])
    best = None
    for eps in eps_list:
        rels = []
        for d in dirs:
            analytic = sum((gr * v).sum() for (_, gr), v in zip(live, d)).item()
            with torch.no_grad():
                for (p, _), v in zip(live, d):
                    p.add_(eps * v)
                lp = L().item()
                for (p, _), v in zip(live, d):
                    p.sub_(2 * eps * v)
                lm = L().item()
                for (p, _), v in zip(live, d):
                    p.add_(eps * v)
            numeric = (lp - lm) / (2 * eps)
            rels.append(abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-8))
        rep = FDReport(max(rels) <= rtol, rels, eps, f"움직인 파라미터 {len(live)}개")
        if rep.ok:
            return rep
        best = rep if best is None or max(rep.rel_errors) < max(best.rel_errors) else best
    return best


@dataclass
class BiasReport:
    ok: bool
    n_routers: int
    details: list[str]


def router_bias_check(mine: nn.Module, pair: dict, inputs, device: str, rate: float = 1e-2) -> BiasReport:
    """aux-loss-free 라우터 bias 업데이트 (miniarch.trainer.RouterBalancer) 점검. 라우터가 없으면 n_routers=0.
    - bias 는 optimizer 에서 빠진다 (기울기·weight decay 대상이 아니라 규칙으로만 움직임)
    - 업데이트 방향: 많이 뽑힌 expert 의 bias 는 내려가고 적게 뽑힌 쪽은 올라간다
    - bias 를 바꾸면 선택이 실제로 바뀐다 (bias 가 선택에 쓰인다)
    """
    from ..trainer import RouterBalancer, param_groups
    m = copy.deepcopy(mine).to(device).float().train()
    bal = RouterBalancer(m)
    if not bal.routers:
        return BiasReport(True, 0, ["라우터 bias 없음 (softmax 라우터·aux loss 방식이거나 MoE 없음)"])
    groups = param_groups(m, 0.1, bal.bias_params())
    in_opt = {id(p) for g in groups for p in g["params"]}
    x = _to(_mine_inputs(pair, _to(inputs, device)), device)
    out = _call(m, x)
    (_first_tensor(out).float().pow(2).mean()).backward()
    details, ok = [], True
    loads = {}
    for name, _, b in bal.routers:
        idx = bal._idx.get(name)
        if idx is None:
            details.append(f"{name}: forward 에서 route_probe 가 안 불림"); ok = False; continue
        loads[name] = torch.bincount(idx.flatten().long(), minlength=b.numel()).float()
        if id(b) in in_opt:
            details.append(f"{name}: bias 가 optimizer 에 들어 있음"); ok = False
    before = {name: b.detach().clone() for name, _, b in bal.routers}
    bal.update(rate)
    for name, _, b in bal.routers:
        if name not in loads:
            continue
        load, delta = loads[name], (b.detach() - before[name]).float()
        over, under = load > load.mean(), load < load.mean()
        good = (delta[over] < 0).all().item() and (delta[under] > 0).all().item()
        ok &= good
        details.append(f"{name}: 부하 {load.int().tolist()} → bias 변화 부호 {'정상' if good else '반대/없음'}")
    # bias 를 크게 바꾸면 선택이 바뀌어야 한다
    name, _, b = bal.routers[0]
    with torch.no_grad():
        b.add_(torch.linspace(-10, 10, b.numel(), device=b.device, dtype=b.dtype))
        _call(m, x)
    sel =torch.bincount(bal._idx[name].flatten().long(), minlength=b.numel()).float()
    moved = not torch.equal(sel, loads[name])
    ok &= moved
    details.append(f"{name}: bias 를 크게 바꾸면 선택 {'바뀜' if moved else '안 바뀜'}")
    bal.remove()
    return BiasReport(ok, len(bal.routers), details)
