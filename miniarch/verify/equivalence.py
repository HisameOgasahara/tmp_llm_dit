"""원본(reference) 모듈과 내 구현(mine)의 수치 동치 검사.

흐름:
  1. ref 를 작은 config 로 랜덤 초기화
  2. ref 가중치를 name_map 으로 mine 에 복사
  3. 같은 입력으로 forward → 최종 출력 + (선택) 중간 텐서 비교
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch
from torch import nn


def set_deterministic(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def force_eager(module: nn.Module) -> None:
    """HF 모델은 sdpa/flash 대신 eager 경로를 쓰도록 강제 (커널 차이 제거)."""
    cfg = getattr(module, "config", None)
    if cfg is not None and hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"


@torch.no_grad()
def perturb_params(module: nn.Module, std: float = 0.05, seed: int = 123) -> None:
    """모든 파라미터에 잡음 추가.

    bias=0, norm weight=1 같은 초기값은 복사가 빠져도 양쪽이 우연히 같아 검사를 통과해 버린다.
    잡음을 넣으면 빠진 파라미터가 출력 차이로 드러난다.
    """
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        p.add_(torch.randn(p.shape, generator=g).to(p.device, p.dtype) * std)


# ---------------------------------------------------------------- 가중치 복사

NameMap = Mapping[str, str] | Callable[[str], str | None]


@dataclass
class CopyReport:
    copied: list[str] = field(default_factory=list)
    unmapped_ref: list[str] = field(default_factory=list)      # ref 에만 있음
    untouched_mine: list[str] = field(default_factory=list)    # mine 에 복사 안 된 것
    shape_mismatch: list[tuple[str, str, tuple, tuple]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unmapped_ref or self.untouched_mine or self.shape_mismatch)


def _resolve(name_map: NameMap | None, ref_name: str) -> str | None:
    if name_map is None:
        return ref_name
    if callable(name_map):
        return name_map(ref_name)
    return name_map.get(ref_name)


@torch.no_grad()
def copy_weights(ref: nn.Module, mine: nn.Module, name_map: NameMap | None = None,
                 transforms: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None
                 ) -> CopyReport:
    """ref.state_dict() → mine. name_map: ref 이름 → mine 이름 (None 반환 = 무시).

    transforms: mine 이름 → 텐서 변환 함수 (예: fused qkv 분할, transpose 등).
    """
    rep = CopyReport()
    mine_sd = mine.state_dict()
    touched: set[str] = set()
    for rname, rt in ref.state_dict().items():
        mname = _resolve(name_map, rname)
        if mname is None:
            continue
        if mname not in mine_sd:
            rep.unmapped_ref.append(f"{rname} -> {mname}")
            continue
        t = rt
        if transforms and mname in transforms:
            t = transforms[mname](t)
        if tuple(t.shape) != tuple(mine_sd[mname].shape):
            rep.shape_mismatch.append((rname, mname, tuple(t.shape), tuple(mine_sd[mname].shape)))
            continue
        mine_sd[mname].copy_(t)
        touched.add(mname)
        rep.copied.append(mname)
    # 버퍼(rope cache 등)는 복사 대상이 아니어도 됨 → 파라미터만 검사
    param_names = {n for n, _ in mine.named_parameters()}
    rep.untouched_mine = sorted(param_names - touched)
    return rep


# ---------------------------------------------------------------- 중간 텐서 캡처

def _first_tensor(x: Any) -> torch.Tensor | None:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x:
        return _first_tensor(x[0])
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    if hasattr(x, "sample"):
        return x.sample
    return None


@contextlib.contextmanager
def capture(module: nn.Module, names: list[str]):
    """module 내부 서브모듈 출력을 {name: tensor} 로 캡처."""
    store: dict[str, torch.Tensor] = {}
    handles = []
    subs = dict(module.named_modules())
    for n in names:
        if n not in subs:
            raise KeyError(f"submodule '{n}' 없음. 예: {list(subs)[:10]}")

        def hook(_m, _i, out, n=n):
            t = _first_tensor(out)
            if t is not None:
                store[n] = t.detach()
        handles.append(subs[n].register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------- 비교

@dataclass
class CompareResult:
    name: str
    ok: bool
    max_abs: float
    max_rel: float
    shape_ref: tuple
    shape_mine: tuple
    note: str = ""


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor,
                    atol: float = 1e-5, rtol: float = 1e-4) -> CompareResult:
    if a.shape != b.shape:
        return CompareResult(name, False, float("nan"), float("nan"),
                             tuple(a.shape), tuple(b.shape), "shape 불일치")
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    max_rel = (diff / (a.abs() + 1e-8)).max().item() if diff.numel() else 0.0
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    return CompareResult(name, ok, max_abs, max_rel, tuple(a.shape), tuple(b.shape))


@dataclass
class EquivReport:
    copy: CopyReport
    results: list[CompareResult]

    @property
    def ok(self) -> bool:
        return self.copy.ok and all(r.ok for r in self.results)

    def table(self):
        import pandas as pd
        return pd.DataFrame([r.__dict__ for r in self.results])

    def summary(self) -> str:
        lines = [f"{'✅ PASS' if self.ok else '❌ FAIL'}  (복사 {len(self.copy.copied)}개)"]
        c = self.copy
        if c.unmapped_ref:
            lines.append(f"  ref 에만 있는 가중치 {len(c.unmapped_ref)}: {c.unmapped_ref[:5]}")
        if c.untouched_mine:
            lines.append(f"  복사 안 된 mine 파라미터 {len(c.untouched_mine)}: {c.untouched_mine[:5]}")
        if c.shape_mismatch:
            lines.append(f"  shape 불일치 {len(c.shape_mismatch)}: {c.shape_mismatch[:3]}")
        for r in self.results:
            mark = "✅" if r.ok else "❌"
            lines.append(f"  {mark} {r.name:<40} max_abs={r.max_abs:.2e} {r.note}")
        return "\n".join(lines)


@torch.no_grad()
def check_equivalence(
    ref: nn.Module,
    mine: nn.Module,
    inputs: tuple | dict,
    name_map: NameMap | None = None,
    *,
    mine_inputs: tuple | dict | None = None,
    mine_input_fn: Callable[[Any], tuple | dict] | None = None,
    out_fn_ref: Callable[[Any], torch.Tensor] = _first_tensor,
    out_fn_mine: Callable[[Any], torch.Tensor] = _first_tensor,
    hook_pairs: list[tuple[str, str]] | None = None,
    transforms: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    device: str | torch.device | None = None,
) -> EquivReport:
    """ref 와 mine 이 같은 함수인지 검사.

    hook_pairs: [(ref 서브모듈 이름, mine 서브모듈 이름), ...] → 중간 텐서도 비교해서
                어디서부터 어긋나는지 찾음.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref = ref.to(device).float().eval()
    mine = mine.to(device).float().eval()
    force_eager(ref)
    rep_copy = copy_weights(ref, mine, name_map, transforms)

    def _to(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: _to(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return type(x)(_to(v) for v in x)
        return x

    inputs = _to(inputs)
    if mine_inputs is not None:
        mine_inputs = _to(mine_inputs)
    elif mine_input_fn is not None:  # 입력 이름/순서가 다른 경우 (예: DiT 의 x, t, y)
        mine_inputs = mine_input_fn(inputs)
    else:
        mine_inputs = inputs

    def _call(m, x):
        return m(**x) if isinstance(x, dict) else m(*x)

    pairs = hook_pairs or []
    with capture(ref, [p[0] for p in pairs]) as rs, capture(mine, [p[1] for p in pairs]) as ms:
        out_r = out_fn_ref(_call(ref, inputs))
        out_m = out_fn_mine(_call(mine, mine_inputs))

    results = [compare_tensors(f"{a}  <->  {b}", rs[a], ms[b], atol, rtol)
               for a, b in pairs if a in rs and b in ms]
    results.append(compare_tensors("[final output]", out_r, out_m, atol, rtol))
    return EquivReport(rep_copy, results)
