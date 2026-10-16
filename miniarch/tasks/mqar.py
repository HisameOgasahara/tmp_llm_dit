"""MQAR (multi-query associative recall, Arora et al. 2023 "Zoology").

    입력:  k1 v1 k2 v2 ... kN vN | q1 a1 q2 a2 ... qM aM
    정답:  qi 위치에서 다음 토큰 ai (= qi 와 짝지어진 값)

- 키는 어휘 앞 절반에서 중복 없이, 값은 뒤 절반에서 뽑는다 → 키와 값이 섞이지 않는다.
- 질문 qi 는 앞에 나온 키 중에서 **중복을 허용해** 고른다. 중복 없이 모든 키를 한 번씩 물으면
  "아직 답하지 않은 값 중 하나"를 고르는 소거법으로 기억 없이 정확도 (1/8+…+1/1)/8 ≈ 0.34 를 얻을 수 있고,
  실제로 작은 Llama 가 여기에 갇혀 연관 기억을 배우지 않았다 (2026-10-16). 중복 허용이면 소거법이 통하지 않는다.
- ai 는 입력에도 있지만 qi 다음 칸이라 causal 모델은 볼 수 없다 (정답 누수는 causal mask 가 막는다).
- 거리 = 질문 위치 − 그 키의 위치. 멀수록 어렵고, sliding window·선형 attention 의 한계가 여기서 드러난다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import IGNORE, LMTask, model_logits


@dataclass
class MQARConfig:
    vocab_size: int = 128
    n_pairs: int = 8
    n_queries: int | None = None          # None = n_pairs
    distance_edges: tuple = (8, 16, 32)   # 거리 구간 경계 (지표: acc_d<8, acc_d8-16, ...)
    query_replacement: bool = True        # True: 질문 키 중복 허용 (소거법 차단). False: 모든 키를 한 번씩


class MQAR(LMTask):
    name = "mqar"

    def __init__(self, cfg: MQARConfig = MQARConfig()):
        self.cfg = cfg
        self.n_q = cfg.n_queries or cfg.n_pairs
        half = cfg.vocab_size // 2
        assert cfg.n_pairs <= half and (cfg.query_replacement or self.n_q <= cfg.n_pairs)
        self.half = half

    @property
    def seq_len(self) -> int:
        return 2 * self.cfg.n_pairs + 2 * self.n_q

    def sample(self, gen: torch.Generator, batch_size: int):
        c, N, M, H = self.cfg, self.cfg.n_pairs, self.n_q, self.half
        keys = torch.argsort(torch.rand(batch_size, H, generator=gen), dim=1)[:, :N]            # [B, N] 중복 없음
        vals = torch.randint(H, c.vocab_size, (batch_size, N), generator=gen)                   # [B, N]
        if c.query_replacement:
            qi = torch.randint(0, N, (batch_size, M), generator=gen)                             # 질문할 짝 번호 (중복 허용)
        else:
            qi = torch.argsort(torch.rand(batch_size, N, generator=gen), dim=1)[:, :M]           # 중복 없이
        q, a = keys.gather(1, qi), vals.gather(1, qi)
        ctx = torch.stack([keys, vals], 2).reshape(batch_size, 2 * N)
        qa = torch.stack([q, a], 2).reshape(batch_size, 2 * M)
        ids = torch.cat([ctx, qa], 1)
        tgt = torch.full_like(ids, IGNORE)
        qpos = 2 * N + 2 * torch.arange(M)                                                       # 질문 위치
        tgt[:, qpos] = a
        dist = qpos[None] - 2 * qi                                                               # 질문 − 키 위치
        return ids, tgt, dist

    def batch(self, split, gen, batch_size):
        ids, tgt, dist = self.sample(gen, batch_size)
        return {"input_ids": ids, "targets": tgt, "distance": dist}

    @torch.no_grad()
    def metrics(self, model, batch):
        out = super().metrics(model, batch)
        logits = model_logits(model, batch["input_ids"])
        N = self.cfg.n_pairs
        qpos = 2 * N + 2 * torch.arange(self.n_q, device=logits.device)
        correct = (logits[:, qpos].argmax(-1) == batch["targets"][:, qpos]).float()       # [B, M]
        dist = batch["distance"].to(logits.device)
        edges = (0,) + tuple(self.cfg.distance_edges) + (10 ** 9,)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (dist >= lo) & (dist < hi)
            if m.any():
                out[f"acc_d{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = correct[m].mean().item()
        return out
