"""A7a · Qwen3.8-Flash-Next (125B-A6B): Qwen3.6 하이브리드 + Hyper-Connections + QSA indexer + n-gram PLE + MoE.

HF 원본: transformers/models/qwen4_exp (Qwen4ExpTextModel, 텍스트 부분만). 파라미터 이름은 HF 와 같다.
Gated DeltaNet / partial RoPE / zero-centered RMSNorm 은 qwen36.py 의 부품을 그대로 쓴다.

블록 구성 (한 층)
0. residual 을 hc_count(=4) 개 줄기로 들고 다닌다. 상태 h 는 [B, T, n·D] (줄기를 이어 붙인 모양).
1. PLE (일부 선형 층만) — 토큰 n-gram(2-gram, 3-gram) 을 해시해서 거대한 표에서 임베딩을 꺼내고,
   줄기마다 "이 n-gram 정보가 지금 필요한가" 를 key·query 내적 게이트로 정해 h 에 더한다 (+ dilated conv).
   EOS 를 만나면 n-gram 문맥이 끊긴다 (문서 경계 너머를 섞지 않음).
2. Gated Residual (Hyper-Connection) — 층 앞에서
      입력 = mean_j( sigmoid(저랭크 MLP(norm(h)))_j ⊙ norm(h)_j )     (줄기 n 개 → 블록 입력 하나)
      h ← h + 2·sigmoid(W_inj · norm(h) / n)_j · 블록출력              (블록 출력을 줄기마다 다른 세기로 주입)
   별도의 input_layernorm 이 없고 이 group RMSNorm 이 pre-norm 역할을 한다. 줄기 사이를 섞는 행렬(comb)은 없다.
3. 토큰 섞기 — 선형(Gated DeltaNet) 3 : softmax 1.
   softmax 층은 QSA indexer: 지나간 토큰을 r 개씩 묶어 평균 key 하나로 요약하고, 작은 질의(head 여러 개)로
   점수 Σ_h ReLU(q_h·k) 를 매겨 상위 budget/r 개 묶음 + 아직 덜 찬 마지막 묶음의 토큰만 attention 한다.
4. MoE — softmax 라우터 top-k (확률 재정규화) + sigmoid 게이트가 붙은 shared expert. 2 와 같은 HC 로 감싼다.
마지막: 줄기 n 개를 같은 Gated Residual(주입 없음)로 하나로 합친 것이 최종 hidden (별도 final norm 없음).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from . import qwen36 as Q
from .common import LMOutput

LINEAR, INDEXED = "linear_attention", "indexed_attention"


@dataclass
class Qwen38FlashNextConfig:
    vocab_size: int = 128
    d_model: int = 256
    n_layer: int = 4
    layer_types: list[str] | None = None        # None → [선형, 선형, 선형, indexed] 반복
    full_attention_interval: int = 4
    # softmax attention (Gated Attention + QSA)
    n_head: int = 4
    n_kv_head: int = 2
    head_dim: int = 64
    partial_rotary: float = 0.25                # head_dim 앞 25% 에만 RoPE (rotate_half 방식)
    rope_theta: float = 10_000_000.0
    index_heads: int = 4
    index_dim: int = 64                         # RoPE 폭(head_dim·partial) 이하여야 함
    index_budget: int = 16                      # 묶음에서 고르는 최대 토큰 수 (budget/ratio = 고르는 묶음 수)
    index_ratio: int = 4                        # 토큰 4 개 → 요약 key 1 개
    # Gated DeltaNet
    lin_k_heads: int = 2
    lin_v_heads: int = 4
    lin_k_dim: int = 64
    lin_v_dim: int = 64
    conv_kernel: int = 4
    delta_mode: str = "chunk"
    chunk_size: int = 64
    output_gate: str = "sigmoid"                # DeltaNet 출력 게이트 활성함수 (실제 체크포인트 = sigmoid)
    # MoE
    n_experts: int = 8
    top_k: int = 2
    moe_inter: int = 128
    shared_inter: int = 128
    norm_topk: bool = True
    # Hyper-Connections
    hc_count: int = 4
    hc_lowrank: int = 32
    # PLE (n-gram 임베딩)
    ple_layer_ids: list[int] = field(default_factory=lambda: [2])   # 1 부터 세는 층 번호, 선형 층만
    ple_embed_dim: int | None = None            # None → d_model
    ple_conv_kernel: int = 4
    ngram_size: int = 3                         # 2-gram ~ ngram_size-gram
    heads_per_ngram: int = 2                    # n-gram 차수마다 서로 다른 해시 표 개수
    ngram_vocab_base: int = 1000                # 해시 표 크기 = 이 값 이후의 소수들
    ngram_vocab_div: int = 128
    seed: int = 1234
    eos_token_id: int = 1
    # 기타
    rms_eps: float = 1e-6
    tie_embeddings: bool = True
    # ablation 스위치 (기본값 = 원본)
    use_hc: bool = True                         # False: 줄기 1 개 (게이트·주입 구조는 유지)
    use_indexer: bool = True                    # False: 일반 causal full attention
    use_ple: bool = True                        # False: n-gram 임베딩 없음
    use_moe: bool = True                        # False: 게이트 달린 shared expert 만

    def __post_init__(self):
        if self.layer_types is None:
            k = self.full_attention_interval
            self.layer_types = [LINEAR if (i + 1) % k else INDEXED for i in range(self.n_layer)]
        self.layer_types = [INDEXED if t == "full_attention" else t for t in self.layer_types]
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.d_model
        assert len(self.layer_types) == self.n_layer
        assert all(self.layer_types[i - 1] == LINEAR for i in self.ple_layer_ids), "PLE 는 선형 층에만"

    @property
    def n_streams(self) -> int:
        return self.hc_count if self.use_hc else 1


# ---------------------------------------------------------------- norms

class ZCRMSNorm(nn.Module):
    """zero-centered RMSNorm: norm(x)·(1+w). group 을 주면 마지막 축을 group 폭 조각마다 따로 정규화 (줄기별)."""

    def __init__(self, dim: int, group: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps, self.group = eps, group
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        h = x.float()
        if self.group is not None:
            h = h.reshape(*h.shape[:-1], -1, self.group)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group is not None:
            h = h.flatten(-2)
        return (h * (1.0 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """norm(x)·w·act(z). act = silu 또는 sigmoid."""

    def __init__(self, dim: int, eps: float = 1e-6, act: str = "sigmoid"):
        super().__init__()
        self.eps, self.act = eps, act
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, z):
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(dtype)
        z = z.float()
        return (h * (torch.sigmoid(z) if self.act == "sigmoid" else F.silu(z))).to(dtype)


# ---------------------------------------------------------------- Gated DeltaNet (qwen36 재사용)

class GatedDeltaNet(Q.GatedDeltaNet):
    """qwen36 의 Gated DeltaNet 과 같고 출력 게이트 활성함수만 설정 가능."""

    def __init__(self, c: Qwen38FlashNextConfig):
        super().__init__(Q.Qwen36Config(
            vocab_size=c.vocab_size, d_model=c.d_model, n_layer=1, layer_types=[LINEAR],
            lin_k_heads=c.lin_k_heads, lin_v_heads=c.lin_v_heads, lin_k_dim=c.lin_k_dim, lin_v_dim=c.lin_v_dim,
            conv_kernel=c.conv_kernel, delta_mode=c.delta_mode, chunk_size=c.chunk_size, rms_eps=c.rms_eps))
        self.norm = GatedRMSNorm(c.lin_v_dim, c.rms_eps, c.output_gate)


# ---------------------------------------------------------------- QSA indexer + Gated Attention

class Indexer(nn.Module):
    """QSA: 지난 토큰을 r 개씩 묶어 평균 key 로 요약 → 점수 상위 budget/r 묶음 + 덜 찬 꼬리 묶음 토큰만 허용."""

    def __init__(self, c: Qwen38FlashNextConfig):
        super().__init__()
        self.h, self.d, self.r = c.index_heads, c.index_dim, c.index_ratio
        self.topk = c.index_budget // c.index_ratio
        self.index_qk_proj = nn.Linear(c.d_model, (self.h + 1) * self.d, bias=False)   # [질의 h 개 | key 1 개]
        self.q_layernorm = ZCRMSNorm(self.d, eps=c.rms_eps)
        self.k_layernorm = ZCRMSNorm(self.d, eps=c.rms_eps)
        self.score_probe = nn.Identity()    # hook 지점: 묶음 점수 [B, T, W] (아직 못 보는 묶음 = -inf)
        self.select_probe = nn.Identity()   # hook 지점: 허용된 토큰 [B, T, T] (1 = 선택)

    def forward(self, x, cos, sin):
        """→ 허용 mask [B, T, T] (True = attention 가능). cos/sin [T, rot]."""
        B, T, _ = x.shape
        q, k = self.index_qk_proj(x).split([self.h * self.d, self.d], dim=-1)
        q = self.q_layernorm(q.reshape(B, T, self.h, self.d))
        q = Q.apply_partial_rope(q, cos[:, None], sin[:, None])                         # [B, T, h, d]
        k = k.reshape(B, T, self.d)
        W = T // self.r

        allow = torch.zeros(B, T, T, dtype=torch.bool, device=x.device)
        scores_all = torch.full((B, T, W), float("-inf"), device=x.device)
        # 원본과 같은 질의별 루프. 묶음 요약 key(토큰 평균 → norm → 묶음 첫 토큰 위치로 RoPE)도 원본처럼 질의마다
        # 같은 모양으로 다시 계산한다 → 점수가 비트 단위로 같아 top-k 동점·근접값 처리까지 일치.
        for b in range(B):
            for i in range(T):
                nb = (i + 1) // self.r                       # 질의 i 가 볼 수 있는 완성된 묶음 수
                if nb > 0:
                    groups = k[b, : nb * self.r].reshape(nb, self.r, self.d)
                    pooled = self.k_layernorm(groups.float().mean(dim=1).to(k.dtype))
                    start = torch.arange(nb, device=x.device) * self.r
                    keys = Q.apply_partial_rope(pooled.unsqueeze(1), cos[start][:, None], sin[start][:, None]).squeeze(1)
                    s = torch.matmul(q[b, i].float(), keys.float().transpose(-1, -2)).transpose(-1, -2)
                    s = torch.relu(s).sum(dim=-1) / math.sqrt(self.d)                   # [nb]
                    sel = s.topk(min(self.topk, nb), dim=0).indices
                    tok = (sel[:, None] * self.r + torch.arange(self.r, device=x.device)).flatten()
                    allow[b, i, tok] = True
                    scores_all[b, i, :nb] = s
                allow[b, i, nb * self.r: i + 1] = True       # 덜 찬 꼬리 묶음은 전부 봄
        self.score_probe(scores_all)
        return self.select_probe(allow.float()) > 0.5


class Attention(nn.Module):
    """Gated Attention (QK-norm, partial RoPE, head 별 sigmoid 출력 게이트) + QSA 로 고른 토큰만 봄."""

    def __init__(self, c: Qwen38FlashNextConfig):
        super().__init__()
        self.c = c
        self.h, self.kv, self.d = c.n_head, c.n_kv_head, c.head_dim
        self.q_proj = nn.Linear(c.d_model, self.h * self.d * 2, bias=False)   # head 마다 [query | gate]
        self.k_proj = nn.Linear(c.d_model, self.kv * self.d, bias=False)
        self.v_proj = nn.Linear(c.d_model, self.kv * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, c.d_model, bias=False)
        self.q_norm = ZCRMSNorm(self.d, eps=c.rms_eps)
        self.k_norm = ZCRMSNorm(self.d, eps=c.rms_eps)
        self.indexer = Indexer(c)          # use_indexer=False 여도 파라미터는 둔다 (가중치 호환)
        self.attn_probs = nn.Identity()    # hook 지점: [B, H, T, T]
        self.gate_probe = nn.Identity()    # hook 지점: sigmoid 게이트 [B, T, H*D]

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        if self.c.use_indexer:
            allow = self.indexer(x, cos, sin)
        else:
            allow = torch.ones(T, T, dtype=torch.bool, device=x.device).tril().expand(B, T, T)
        q, gate = self.q_proj(x).view(B, T, self.h, 2 * self.d).chunk(2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.kv, self.d)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv, self.d).transpose(1, 2)
        q, k = Q.apply_partial_rope(q, cos, sin), Q.apply_partial_rope(k, cos, sin)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, k.transpose(2, 3)) * self.d ** -0.5
        scores = scores.masked_fill(~allow[:, None], float("-inf"))
        probs = self.attn_probs(scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype))
        y = (probs @ v).transpose(1, 2).reshape(B, T, -1)
        y = y * self.gate_probe(torch.sigmoid(gate))
        return self.o_proj(y)


# ---------------------------------------------------------------- MoE

class MLP(nn.Module):
    def __init__(self, d, inter):
        super().__init__()
        self.gate_proj = nn.Linear(d, inter, bias=False)
        self.up_proj = nn.Linear(d, inter, bias=False)
        self.down_proj = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    def __init__(self, c: Qwen38FlashNextConfig):
        super().__init__()
        self.c = c
        self.weight = nn.Parameter(torch.empty(c.n_experts, c.d_model))
        nn.init.normal_(self.weight, std=0.02)
        self.route_probe = nn.Identity()   # hook 지점: 선택된 expert [N, k]
        self.prob_probe = nn.Identity()    # hook 지점: 라우터 softmax 확률 [N, E]

    def forward(self, x):
        logits = F.linear(x, self.weight)
        probs = self.prob_probe(F.softmax(logits, dtype=torch.float, dim=-1))
        top, idx = torch.topk(probs, self.c.top_k, dim=-1)
        if self.c.norm_topk:
            top = top / top.sum(dim=-1, keepdim=True)
        return self.route_probe(idx), top.to(logits.dtype)


class Experts(nn.Module):
    def __init__(self, c: Qwen38FlashNextConfig):
        super().__init__()
        self.c = c
        self.gate_up_proj = nn.Parameter(torch.randn(c.n_experts, 2 * c.moe_inter, c.d_model) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(c.n_experts, c.d_model, c.moe_inter) * 0.02)

    def forward(self, x, idx, w):
        out = torch.zeros_like(x)
        for e in range(self.c.n_experts):
            slot, tok = torch.where(idx.T == e)              # 원본과 같은 순서 (slot 먼저)
            if tok.numel() == 0:
                continue
            g, u = F.linear(x[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            h = F.linear(F.silu(g) * u, self.down_proj[e]) * w[tok, slot, None]
            out.index_add_(0, tok, h.to(out.dtype))
        return out


class MoE(nn.Module):
    def __init__(self, c: Qwen38FlashNextConfig):
        super().__init__()
        self.c = c
        if c.use_moe:
            self.gate = Router(c)
            self.experts = Experts(c)
        self.shared_expert = MLP(c.d_model, c.shared_inter)
        self.shared_expert_gate = nn.Linear(c.d_model, 1, bias=False)
        self.shared_gate_probe = nn.Identity()   # hook 지점: shared expert 게이트 [N, 1]

    def forward(self, x):
        B, T, D = x.shape
        flat = x.view(-1, D)
        shared = self.shared_expert(flat)
        shared = self.shared_gate_probe(torch.sigmoid(self.shared_expert_gate(flat))) * shared
        if not self.c.use_moe:
            return shared.view(B, T, D)
        idx, w = self.gate(flat)
        return (self.experts(flat, idx, w) + shared).view(B, T, D)


# ---------------------------------------------------------------- Hyper-Connections (Gated Residual)

class GatedResidual(nn.Module):
    """줄기 n 개 [B,T,n·D] → 블록 입력 [B,T,D] (+ 주입 세기 [B,T,n]). combine=False 면 마지막 합치기용."""

    def __init__(self, c: Qwen38FlashNextConfig, combine: bool = True):
        super().__init__()
        n, D = c.n_streams, c.d_model
        self.n, self.D = n, D
        self.hc_norm = ZCRMSNorm(n * D, group=D, eps=c.rms_eps)
        self.input_mix_weight_down = nn.Linear(n * D, c.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(c.hc_lowrank, n * D, bias=False)
        self.block_inject_weight = nn.Linear(n * D, n, bias=False) if combine else None
        self.mix_probe = nn.Identity()      # hook 지점: 줄기·채널별 입력 가중치 [B, T, n, D]
        self.inject_probe = nn.Identity()   # hook 지점: 줄기별 주입 세기 [B, T, n] (0~2)

    def forward(self, h):
        hn = self.hc_norm(h)
        m = F.silu(self.input_mix_weight_down(hn) / self.n)
        m = self.mix_probe(torch.sigmoid(self.input_mix_weight_up(m)).unflatten(-1, (self.n, self.D)))
        x = (m * hn.unflatten(-1, (self.n, self.D))).mean(dim=-2)
        if self.block_inject_weight is None:
            return x
        inj = self.inject_probe(2 * torch.sigmoid(self.block_inject_weight(hn) / self.n))
        return x, inj

    @staticmethod
    def inject(h, out, inj):
        """h ← h + inj_j · out (줄기 j 마다 같은 블록 출력을 다른 세기로)."""
        return h + (out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)


# ---------------------------------------------------------------- PLE (n-gram 임베딩)

_MASK64 = (1 << 64) - 1
_GAMMA, _M1, _M2, _PRIME_1 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 10007


def _splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def layer_multipliers(vocab: int, ngram_size: int, ple_index: int, seed: int) -> torch.Tensor:
    """n-gram 위치마다 쓰는 홀수 곱셈 상수 (토큰 id × 상수 가 int64 를 넘지 않게 상한)."""
    half = max(1, ((1 << 63) - 1) // max(vocab, 1) // 2)
    base = seed + _PRIME_1 * ple_index
    return torch.tensor([2 * (_splitmix64((base + _GAMMA * (i + 1)) & _MASK64) % half) + 1
                         for i in range(ngram_size)], dtype=torch.long)


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


class NGramEmbedding(nn.Module):
    """2~N-gram 을 해시해서 표에서 꺼낸다. 차수마다 heads_per_ngram 개의 서로 다른 소수 크기 표 (충돌 분산)."""

    def __init__(self, c: Qwen38FlashNextConfig, ple_index: int):
        super().__init__()
        self.c, self.N = c, c.ngram_size
        self.heads = (c.ngram_size - 1) * c.heads_per_ngram
        sizes, offsets, total = [], [], 0
        for j in range(self.heads):
            s = _nth_prime_after(c.ngram_vocab_base - 1, ple_index * self.heads + j + 1)
            sizes.append(s)
            offsets.append(total)
            total += s
        # 원본과 같은 이름의 persistent buffer (state_dict 에 들어 있음)
        self.register_buffer("layer_multipliers", layer_multipliers(c.vocab_size, c.ngram_size, ple_index, c.seed))
        self.register_buffer("ngram_heads_vocab_sizes", torch.tensor(sizes, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets", torch.tensor(offsets, dtype=torch.long))
        padded = math.ceil(total / c.ngram_vocab_div) * c.ngram_vocab_div
        self.ngram_embedding = nn.Embedding(padded, c.ple_embed_dim // self.heads)

    def shift_right(self, ids, s: int):
        """s 칸 앞 토큰. 직전 EOS 이후 구간(문서) 밖이면 EOS 로 채운다."""
        if s == 0:
            return ids
        B, T = ids.shape
        eos = self.c.eos_token_id
        pos = torch.arange(T, device=ids.device)
        eos_pos = torch.where(ids == eos, pos, -1)
        prev_incl = torch.cummax(eos_pos, dim=1).values
        prev = torch.cat([eos_pos.new_full((B, 1), -1), prev_incl[:, :-1]], dim=1)   # 자기 자신 제외 직전 EOS
        in_seg = pos[None] - (prev + 1)
        src = pos - s
        shifted = ids.gather(1, src.clamp_min(0)[None].expand(B, -1))
        valid = (in_seg >= s) & (src[None] >= 0)
        return torch.where(valid, shifted, ids.new_full((), eos))

    def forward(self, input_ids):
        ids = input_ids.long()
        T = ids.shape[1]
        hist = torch.cat([ids.new_full((ids.shape[0], self.N - 1), self.c.eos_token_id), ids], dim=-1)
        sh = [self.shift_right(hist, s) for s in range(self.N)]
        blocks, hp = [], self.c.heads_per_ngram
        for n in range(2, self.N + 1):
            a = (n - 2) * hp
            mixed = sh[0] * self.layer_multipliers[0]
            for p in range(1, n):
                mixed = torch.bitwise_xor(mixed, sh[p] * self.layer_multipliers[p])
            ng = torch.remainder(mixed.unsqueeze(-1), self.ngram_heads_vocab_sizes[a:a + hp].view(1, 1, -1))
            blocks.append(ng + self.ngram_heads_offsets[a:a + hp].view(1, 1, -1))
        ng_ids = torch.cat(blocks, dim=-1)[:, -T:]                                     # [B, T, heads]
        return self.ngram_embedding(ng_ids).flatten(-2)                                # [B, T, ple_dim]


class PLE(nn.Module):
    """n-gram 값 하나를 줄기마다 게이트(key·query)로 걸러 더하고, dilated depthwise conv 로 이웃 문맥을 보탠다."""

    def __init__(self, c: Qwen38FlashNextConfig, ple_index: int):
        super().__init__()
        n, D, E = c.n_streams, c.d_model, c.ple_embed_dim
        self.n, self.D = n, D
        self.ple_embedding = NGramEmbedding(c, ple_index)
        self.key_proj = nn.Linear(E, n * D, bias=False)
        self.value_proj = nn.Linear(E, D, bias=False)
        self.norm_key = ZCRMSNorm(n * D, group=D, eps=c.rms_eps)
        self.norm_query = ZCRMSNorm(n * D, group=D, eps=c.rms_eps)
        self.norm_conv = ZCRMSNorm(n * D, group=D, eps=c.rms_eps)
        self.pad = (c.ple_conv_kernel - 1) * c.ngram_size
        self.conv1d = nn.Conv1d(n * D, n * D, kernel_size=c.ple_conv_kernel, groups=n * D,
                                dilation=c.ngram_size, bias=False)
        self.gate_probe = nn.Identity()   # hook 지점: 줄기별 n-gram 게이트 sigmoid [B, T, n, 1]

    def forward(self, h, input_ids):
        e = self.ple_embedding(input_ids)
        key = self.norm_key(self.key_proj(e)).unflatten(-1, (self.n, self.D))
        value = self.value_proj(e)
        query = self.norm_query(h).unflatten(-1, (self.n, self.D))
        g = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.D)
        g = g.abs().clamp_min(1e-6).sqrt() * g.sign()                                 # 부호 유지 sqrt (폭 압축)
        gv = self.gate_probe(torch.sigmoid(g)) * value.unsqueeze(-2)                   # [B, T, n, D]
        gv_normed = self.norm_conv(gv.flatten(-2))
        gv = gv.flatten(-2)
        conv = F.pad(gv_normed.transpose(1, 2), (self.pad, 0))                         # 앞쪽 padding → causal
        return gv + F.silu(self.conv1d(conv)).transpose(1, 2)


# ---------------------------------------------------------------- 층 / 모델

class Layer(nn.Module):
    def __init__(self, c: Qwen38FlashNextConfig, i: int):
        super().__init__()
        self.c, self.layer_type = c, c.layer_types[i]
        if self.layer_type == LINEAR:
            self.linear_attn = GatedDeltaNet(c)
        else:
            self.self_attn = Attention(c)
        self.mlp = MoE(c)
        self.ple = PLE(c, c.ple_layer_ids.index(i + 1)) if c.use_ple and (i + 1) in c.ple_layer_ids else None
        self.attn_hyper_connection = GatedResidual(c)
        self.mlp_hyper_connection = GatedResidual(c)

    def forward(self, h, input_ids, cos, sin):
        if self.ple is not None:
            h = h + self.ple(h, input_ids)
        x, inj = self.attn_hyper_connection(h)
        a = self.linear_attn(x) if self.layer_type == LINEAR else self.self_attn(x, cos, sin)
        h = GatedResidual.inject(h, a, inj)
        x, inj = self.mlp_hyper_connection(h)
        return GatedResidual.inject(h, self.mlp(x), inj)


class Qwen38FlashNext(nn.Module):
    def __init__(self, c: Qwen38FlashNextConfig):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.n_layer))
        self.hyper_connection_mixer = GatedResidual(c, combine=False)
        self.lm_head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        rot = int(c.head_dim * c.partial_rotary)
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, rot, 2, dtype=torch.float) / rot))
        self.register_buffer("inv_freq", inv, persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> LMOutput:
        T = input_ids.shape[1]
        # 텍스트만이면 M-RoPE 세 축 위치가 같아 일반 RoPE 와 같다
        freqs = torch.arange(T, device=input_ids.device).float()[:, None] * self.inv_freq[None]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        x = self.embed_tokens(input_ids)
        h = x.repeat(1, 1, self.config.n_streams)            # 줄기 n 개를 같은 값으로 시작
        for layer in self.layers:
            h = layer(h, input_ids, cos, sin)
        h = self.hyper_connection_mixer(h)
        logits = self.lm_head(h) if self.lm_head is not None else h @ self.embed_tokens.weight.T
        return LMOutput(h, logits)


# ---------------------------------------------------------------- 검증

def config_from_hf(c) -> Qwen38FlashNextConfig:
    """HF Qwen4ExpTextConfig → 내 config. TextModel 에는 lm_head 가 없으므로 tie_embeddings 기본값(True) 유지."""
    rp = getattr(c, "rope_parameters", None) or {}
    eos = c.eos_token_id[0] if isinstance(c.eos_token_id, list) else c.eos_token_id
    return Qwen38FlashNextConfig(
        vocab_size=c.vocab_size, d_model=c.hidden_size, n_layer=c.num_hidden_layers, layer_types=list(c.layer_types),
        n_head=c.num_attention_heads, n_kv_head=c.num_key_value_heads, head_dim=c.head_dim,
        partial_rotary=rp.get("partial_rotary_factor", 1.0), rope_theta=rp["rope_theta"],
        index_heads=c.indexer_n_heads, index_dim=c.indexer_head_dim, index_budget=c.indexer_budget,
        index_ratio=c.indexer_compress_ratio,
        lin_k_heads=c.linear_num_key_heads, lin_v_heads=c.linear_num_value_heads,
        lin_k_dim=c.linear_key_head_dim, lin_v_dim=c.linear_value_head_dim, conv_kernel=c.linear_conv_kernel_dim,
        output_gate=c.output_gate_type or c.hidden_act,
        n_experts=c.num_experts, top_k=c.num_experts_per_tok, moe_inter=c.moe_intermediate_size,
        shared_inter=c.shared_expert_intermediate_size, norm_topk=c.norm_topk_prob,
        hc_count=c.hc_count, hc_lowrank=c.hc_lowrank,
        ple_layer_ids=list(c.ple_layer_ids), ple_embed_dim=c.ple_embed_dim, ple_conv_kernel=c.ple_conv_kernel_size,
        ngram_size=c.ngram_size, heads_per_ngram=c.heads_per_ngram, ngram_vocab_base=c.ngram_vocab_size_base,
        ngram_vocab_div=c.make_ngram_vocab_size_divisible_by, seed=c.seed,
        eos_token_id=eos if eos is not None else 0, rms_eps=c.rms_norm_eps)


def verify_pair(ref) -> dict:
    """ref = transformers Qwen4ExpTextModel (이름이 같아 name_map 불필요)"""
    mine = Qwen38FlashNext(config_from_hf(ref.config))
    hooks = []
    for i, t in enumerate(ref.config.layer_types):
        if ref.layers[i].ple is not None:
            hooks.append((f"layers.{i}.ple", f"layers.{i}.ple"))
        mixer = "linear_attn" if t == LINEAR else "self_attn"
        hooks += [(f"layers.{i}.{mixer}", f"layers.{i}.{mixer}"), (f"layers.{i}.mlp", f"layers.{i}.mlp"),
                  (f"layers.{i}", f"layers.{i}")]
    hooks.append(("hyper_connection_mixer", "hyper_connection_mixer"))
    return dict(mine=mine, name_map=None, hook_pairs=hooks)
