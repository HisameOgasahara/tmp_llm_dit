"""내 구현을 원본 config 크기로 meta device 에 만들고(메모리 0) 원본 헤더와 shape 대조."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from .hf_header import TensorInfo


def meta_state_shapes(build: Callable[[], nn.Module]) -> dict[str, tuple]:
    with torch.device("meta"):
        m = build()
    return {k: tuple(v.shape) for k, v in m.state_dict().items()}


# 원본 FP8 체크포인트의 스케일 텐서 등 비교 대상이 아닌 것
DEFAULT_IGNORE = [r"weight_scale_inv$", r"\.scale$", r"input_scale$", r"rotary_emb\.inv_freq$"]


@dataclass
class ShapeReport:
    matched: list[str] = field(default_factory=list)
    mismatch: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    missing_in_mine: list[str] = field(default_factory=list)
    extra_in_mine: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.mismatch or self.missing_in_mine or self.extra_in_mine)

    def summary(self, n: int = 8) -> str:
        s = [f"{'✅ PASS' if self.ok else '❌ FAIL'}  일치 {len(self.matched)}개"]
        if self.mismatch:
            s.append(f"  shape 불일치 {len(self.mismatch)}:")
            s += [f"    {k}: 원본 {a} / 내것 {b}" for k, a, b in self.mismatch[:n]]
        if self.missing_in_mine:
            s.append(f"  내 구현에 없음 {len(self.missing_in_mine)}: {self.missing_in_mine[:n]}")
        if self.extra_in_mine:
            s.append(f"  원본에 없음 {len(self.extra_in_mine)}: {self.extra_in_mine[:n]}")
        return "\n".join(s)


def compare_shapes(mine: dict[str, tuple], original: dict[str, TensorInfo],
                   name_map: Callable[[str], str | None] | None = None,
                   scope: str | None = None, ignore: list[str] | None = None) -> ShapeReport:
    """name_map: 원본 이름 → 내 이름. scope: 비교할 원본 텐서 정규식(예: 특정 층)."""
    ign = [re.compile(p) for p in (DEFAULT_IGNORE + (ignore or []))]
    sc = re.compile(scope) if scope else None
    rep = ShapeReport()
    seen: set[str] = set()
    for oname, info in original.items():
        if any(p.search(oname) for p in ign) or (sc and not sc.search(oname)):
            continue
        mname = name_map(oname) if name_map else oname
        if mname is None:
            continue
        if mname not in mine:
            rep.missing_in_mine.append(oname)
            continue
        seen.add(mname)
        if tuple(info.shape) == tuple(mine[mname]):
            rep.matched.append(oname)
        else:
            rep.mismatch.append((oname, tuple(info.shape), tuple(mine[mname])))
    if sc is None:
        rep.extra_in_mine = sorted(set(mine) - seen)
    return rep
