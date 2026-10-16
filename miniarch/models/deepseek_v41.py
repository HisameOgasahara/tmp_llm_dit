"""A8 · DeepSeek-V4.1 블록 (V4.1-Flash).

원본: HF deepseek-ai/DeepSeek-V4.1-Flash `inference/model.py` + `engram.py` (transformers main 에는 아직 없음).
파라미터 이름은 원본(= 실제 체크포인트 키)과 같다. V4 와 공통인 부품(RMSNorm, interleaved RoPE, 그룹 출력 투영)은
deepseek_v4.py 에서 가져온다. 단, V4.1 원본은 V4(HF) 와 세부가 달라 아래처럼 따로 구현했다.

블록 구성 (한 층)
1. Engram (일부 층, 블록 앞) — 직전 2..n 토큰의 n-gram 을 해시해 큰 표에서 벡터를 꺼내고,
   residual 줄기와 얼마나 맞는지(정규화 내적 → signed sqrt → sigmoid)로 gate 를 걸어 줄기마다 더한다.
2. Attention (CSA2) — 모든 층이 sliding window 를 보고, compress_ratio > 0 인 층은 "압축 KV" 를 추가로 본다.
   층 모드가 넷이다.
   - sliding : ratio 0. 최근 window 개 토큰만.
   - full    : KV 출처 층. 토큰 r 개를 softmax gate 가중합으로 1 개로 압축(r=1 이면 투영만)하고,
               그 압축 latent 로 indexer key 도 만든 뒤 자기 indexer 로 top-k 를 고른다.
   - reindex : 압축 KV 와 indexer key 는 출처 층 것을 재사용, indexer 질의/가중치만 자기 것으로 top-k 를 다시 고름.
   - reuse   : 압축 KV 와 top-k 선택까지 직전 출처 층 것을 그대로 씀 (추가 파라미터 없음).
   ratio 1 묶음에는 2단계 top-k 가 있다: candidate 출처 층이 압축 칸을 block 단위로 먼저 추려 두면,
   뒤의 reindex 층은 그 block 안에서만 top-k 를 고른다.
   공통: K = V 단일 head, 저랭크 q, head 별 sink, head 뒤쪽 rope 채널만 RoPE(압축 층은 큰 theta + YaRN),
        출력에서 RoPE 를 되돌린 뒤 그룹 저랭크 투영(wo_a → wo_b).
3. MoE — sqrt(softplus) 점수 + bias 로 top-k 선택(가중치는 bias 없는 점수) + shared expert. V4.1 에는 hash 라우팅이 없다.
4. mHC — residual 을 hc_mult 줄기로 들고 다님. 계수(pre/post/comb)는 sublayer 입력에서 계산하지만
   pre(줄기 → 입력 합치기)는 한 칸 늦게 쓰인다: attention 은 직전 FFN 이 만든 pre, FFN 은 이번 attention 이 만든 pre.
   첫 층은 0 번 줄기만 쓰는 one-hot, 마지막은 마지막 FFN 의 pre 로 합친다 (별도 head 파라미터 없음).

MTP 자리의 DSpark draft head(mtp.*)는 추측 디코딩 전용이라 forward 에 쓰이지 않아 구현하지 않았다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput
from .deepseek_v4 import GroupedLinear, RMSNorm, apply_rope_tail

SLIDING, FULL, REINDEX, REUSE = "sliding", "full", "reindex", "reuse"


# ---------------------------------------------------------------- Engram 해시 배치 (학습 없음)

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def engram_primes(layer_ids, max_ngram: int, heads: int, vocab: int) -> list:
    """[층][n-gram 크기][head] 마다 겹치지 않는 소수 크기 bucket. 원본 EngramLayout.from_args 와 같은 순서."""
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram - 1):
            sizes, cur = [], vocab - 1
            for _ in range(heads):
                cur += 1
                while not _is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return primes


def engram_multipliers(layer_ids, max_ngram: int, compressed_vocab: int) -> torch.Tensor:
    """층마다 고정 seed 의 홀수 곱셈 계수 [층, max_ngram] (원본 compute_hash_multipliers)."""
    import numpy as np
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
    rows = [torch.tensor(np.random.default_rng(10007 * i).integers(0, bound, size=(max_ngram,), dtype=np.int64) * 2 + 1)
            for i in layer_ids]
    return torch.stack(rows)


@dataclass
class DeepseekV41Config:
    vocab_size: int = 128
    d_model: int = 256
    n_layer: int = 7
    # CSA2 층 배치. 층마다 0 = sliding 만, r = 압축 r→1
    compress_ratios: list[int] | None = None     # None → [0, 2, 2, 2, 1, 1, 1]
    kv_source_layers: list[int] | None = None    # 압축 KV 를 직접 만드는 층 (None → ratio 묶음마다 첫 층)
    index_source_layers: list[int] | None = None  # 자기 indexer 를 가진 층 (None → kv 출처 + 그 다음 층)
    candidate_source_layer: int = -1             # 2단계 top-k 의 1단계(block 추림) 층. -1 = 끔
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    # attention
    n_head: int = 4
    head_dim: int = 64
    rope_dim: int = 16                           # head 뒤쪽 rope_dim 채널에만 RoPE (interleaved)
    q_lora_rank: int = 64
    o_groups: int = 2
    o_lora_rank: int = 64
    window: int = 16                             # sliding window (자기 자신 포함)
    rope_theta: float = 10000.0                  # sliding 층 (YaRN 없음)
    compress_rope_theta: float = 160000.0        # 압축 층 (+ YaRN)
    original_seq_len: int = 0                    # > 0 이면 압축 층에 YaRN
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_heads: int = 4
    index_dim: int = 32
    index_topk: int = 8
    # mHC
    hc_mult: int = 4
    hc_iters: int = 20
    hc_eps: float = 1e-6
    # MoE
    n_experts: int = 8
    top_k: int = 2
    moe_inter: int = 128
    score_func: str = "sqrtsoftplus"            # softmax / sigmoid / sqrtsoftplus
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0                   # 0 = clamp 안 함
    # Engram
    engram_layer_ids: list[int] | None = None    # None → [1]
    engram_max_ngram: int = 3                    # 2-gram .. max_ngram-gram
    engram_heads: int = 2
    engram_dim: int = 32                         # 32 의 배수 (dequant scale block)
    engram_vocab: int = 50                       # bucket 소수 탐색 시작점
    engram_pad_id: int = 2                       # 앞쪽 이력이 없는 칸을 채우는 토큰
    engram_compressed_vocab: int = 0             # 0 → vocab_size (토큰 정규화 표 크기, 곱셈 계수 범위에 쓰임)
    engram_num_embeddings: list[int] | None = None  # None → 소수 합
    # 기타
    rms_eps: float = 1e-20
    # ablation 스위치 (기본값 = 원본)
    use_engram: bool = True
    use_mhc: bool = True                         # False: 일반 pre-norm residual
    use_compressed_kv: bool = True               # False: 모든 층 sliding 만
    use_indexer: bool = True                     # False: 볼 수 있는 압축 칸 전부 (top-k 없음)
    use_candidates: bool = True                  # False: 2단계 top-k 의 block 추림 끔
    use_sink: bool = True                        # False: attention sink 없음
    use_moe: bool = True                         # False: shared expert 만

    def __post_init__(self):
        n = self.n_layer
        if self.compress_ratios is None:
            base = [0, 2, 2, 2, 1, 1, 1]
            self.compress_ratios = (base + [1] * n)[:n]
        self.compress_ratios = list(self.compress_ratios)[:n]
        if self.kv_source_layers is None:
            self.kv_source_layers = [i for i, r in enumerate(self.compress_ratios)
                                     if r and (i == 0 or self.compress_ratios[i - 1] != r)]
        if self.index_source_layers is None:
            self.index_source_layers = sorted({j for i in self.kv_source_layers for j in (i, i + 1)
                                               if j < n and self.compress_ratios[j] == self.compress_ratios[i]})
        if self.engram_layer_ids is None:
            self.engram_layer_ids = [1] if n > 1 else []
        self.engram_layer_ids = list(self.engram_layer_ids)
        if not self.engram_compressed_vocab:
            self.engram_compressed_vocab = self.vocab_size
        if self.engram_num_embeddings is None:
            ps = engram_primes(self.engram_layer_ids, self.engram_max_ngram, self.engram_heads, self.engram_vocab)
            self.engram_num_embeddings = [sum(sum(p) for p in layer) for layer in ps]

    def mode(self, i: int) -> str:
        r = self.compress_ratios[i] if self.use_compressed_kv else 0
        if not r:
            return SLIDING
        if i in self.kv_source_layers:
            return FULL
        return REINDEX if i in self.index_source_layers else REUSE


# ---------------------------------------------------------------- RoPE (YaRN)

def yarn_inv_freq(dim, base, original_seq_len, factor, beta_fast, beta_slow) -> torch.Tensor:
    """원본 precompute_freqs_cis 의 주파수 부분. original_seq_len > 0 이면 YaRN(긴 파장 채널만 factor 로 나눔)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_table(T: int, inv_freq: torch.Tensor):
    """위치 0..T-1 → cos, sin [T, rope_dim/2].
    원본 precompute_freqs_cis 처럼 CPU fp32 에서 torch.polar 로 만든 뒤 옮긴다 (cos/sin 이 비트 단위로 같아져
    indexer top-k 같은 이산 선택이 ulp 차이로 뒤집힐 여지를 줄임)."""
    f = torch.outer(torch.arange(T, device="cpu"), inv_freq.detach().float().cpu())
    cis = torch.polar(torch.ones_like(f), f)
    return cis.real.contiguous().to(inv_freq.device), cis.imag.contiguous().to(inv_freq.device)


# ---------------------------------------------------------------- Engram

class NgramHash(nn.Module):
    """위치마다 그 위치에서 끝나는 2..n-gram 의 해시 id [B, L, n_engram_층, (n-1)·heads] (원본 NgramHashState)."""

    def __init__(self, c: DeepseekV41Config):
        super().__init__()
        self.n, self.pad_token = c.engram_max_ngram, c.engram_pad_id
        primes = engram_primes(c.engram_layer_ids, c.engram_max_ngram, c.engram_heads, c.engram_vocab)
        flat = [[p for per in layer for p in per] for layer in primes]
        offsets = [[sum(f[:k]) for k in range(len(f))] for f in flat]
        self.register_buffer("primes", torch.tensor(primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer("multipliers", engram_multipliers(c.engram_layer_ids, c.engram_max_ngram,
                                                               c.engram_compressed_vocab), persistent=False)
        # 토큰 → 정규화 토큰 표 (대소문자 등 합침). 원본은 tokenizer 로 만들고, 검증 때 원본 것을 복사해 넣는다.
        self.register_buffer("token_map", torch.arange(c.vocab_size), persistent=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        comp = self.token_map[input_ids]
        pad = self.token_map[self.pad_token]
        pos = torch.arange(L, device=input_ids.device).expand(B, L)
        tokens, blocked = [], torch.zeros_like(pos, dtype=torch.bool)
        for shift in range(self.n):
            src = comp.gather(1, (pos - shift).clamp_min(0))
            blocked = blocked | (pos < shift)                     # 시퀀스 시작 앞은 pad
            tokens.append(torch.where(blocked, pad, src))
        prods = torch.stack(tokens, dim=-1).unsqueeze(2) * self.multipliers   # [B, L, 층, n]
        rolling, hashes = prods[..., 0], []
        for i in range(1, self.n):                                # XOR 누적 → (i+1)-gram 해시
            rolling = torch.bitwise_xor(rolling, prods[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class EngramEmbed(nn.Module):
    """n-gram 해시 표. 원본은 fp8 값 + 32 칸마다 scale 로 저장 → 꺼낼 때 곱해서 복원."""

    def __init__(self, num: int, dim: int, block: int = 32):
        super().__init__()
        self.num, self.block = num, block
        self.weight = nn.Parameter(torch.randn(num, dim))
        self.scale = nn.Parameter(torch.ones(num, dim // block))

    def forward(self, idx):
        mask = idx >= self.num
        idx = idx.masked_fill(mask, 0)
        v = F.embedding(idx, self.weight).float().unflatten(-1, (-1, self.block)) * \
            F.embedding(idx, self.scale).float().unsqueeze(-1)
        return v.flatten(-2).masked_fill(mask.unsqueeze(-1), 0)


class Engram(nn.Module):
    """해시로 꺼낸 벡터 → (줄기별 key, 공유 value). gate = sigmoid(signed sqrt(정규화 <줄기, key>))."""

    def __init__(self, c: DeepseekV41Config, layer_id: int):
        super().__init__()
        self.c, self.index = c, c.engram_layer_ids.index(layer_id)
        cols = (c.engram_max_ngram - 1) * c.engram_heads
        self.embed = EngramEmbed(c.engram_num_embeddings[self.index], c.engram_dim)
        self.wkv = nn.Linear(cols * c.engram_dim, c.d_model * (c.hc_mult + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(c.hc_mult, c.d_model))
        self.k_weight = nn.Parameter(torch.ones(c.hc_mult, c.d_model))
        self.gate_probe = nn.Identity()   # hook 지점: gate [B, L, hc]

    def forward(self, x, hash_ids):
        if x.dim() == 3:                  # mHC 끔 (ablation): 줄기를 복제해 계산 후 평균
            return self.forward(x.unsqueeze(2).expand(-1, -1, self.c.hc_mult, -1), hash_ids).mean(2)
        c = self.c
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([c.hc_mult * c.d_model, c.d_model], dim=-1)
        key = key.float().unflatten(-1, (c.hc_mult, c.d_model))
        w = self.q_weight.float() * self.k_weight.float()
        h = x.float()
        # (토큰, 줄기) 마다 d_model 축으로 정규화 (줄기끼리 묶지 않음)
        rstd = torch.rsqrt(h.square().mean(-1) + c.rms_eps) * torch.rsqrt(key.square().mean(-1) + c.rms_eps)
        dot = (h * w * key).sum(-1) * rstd * c.d_model ** -0.5
        gate = self.gate_probe(torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)))
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


# ---------------------------------------------------------------- 압축 KV / Indexer

class Compressor(nn.Module):
    """토큰 r 개 → softmax(gate) 가중합 1 개 (겹침/위치 bias 없음). r = 1 이면 투영 + norm 만. 출력은 RoPE 전."""

    def __init__(self, c: DeepseekV41Config, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.norm = RMSNorm(c.head_dim, c.rms_eps)
        self.wkv = nn.Linear(c.d_model, c.head_dim, bias=False)
        if ratio > 1:
            self.wgate = nn.Linear(c.d_model, c.head_dim, bias=False)
        self.pool_probe = nn.Identity()   # hook 지점: 묶음 안 softmax 가중치 [B, W, r, D]

    def forward(self, x):
        r = self.ratio
        if r == 1:
            return self.norm(self.wkv(x))
        dtype, x = x.dtype, x.float()
        cut = x.shape[1] - x.shape[1] % r          # 덜 찬 마지막 묶음은 버림 (원본은 decode 용 state 로 보관)
        if cut == 0:
            return x.new_zeros(x.shape[0], 0, self.norm.weight.shape[0]).to(dtype)
        kv = self.wkv(x)[:, :cut].unflatten(1, (-1, r))
        score = self.wgate(x)[:, :cut].unflatten(1, (-1, r))
        return self.norm((kv * self.pool_probe(score.softmax(dim=2))).sum(dim=2).to(dtype))


def select_candidate_blocks(logits, compress_lens, topk_blocks: int, block_size: int):
    """2단계 top-k 의 1단계: 압축 칸을 block_size 씩 묶어 block 최고점 기준 top 블록만 남긴 bool mask."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    nb = scores.size(-1)
    last = (compress_lens - 1) // block_size             # 가장 최근(덜 찬) block 은 항상 포함
    scores = scores.masked_fill(torch.arange(nb, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, nb), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(nn.Module):
    """Lightning Indexer: 점수 Σ_h w_h·ReLU(q_h·k) 로 압축 칸 top-k. key 는 KV 출처 층만 (압축 latent 에서) 만든다."""

    def __init__(self, c: DeepseekV41Config, layer_id: int):
        super().__init__()
        self.c = c
        self.ratio = c.compress_ratios[layer_id]
        self.heads, self.dim = c.index_heads, c.index_dim
        self.owns_k = layer_id in c.kv_source_layers
        self.is_candidate_source = c.use_candidates and layer_id == c.candidate_source_layer
        self.uses_candidates = c.use_candidates and 0 <= c.candidate_source_layer < layer_id
        self.wq_b = nn.Linear(c.q_lora_rank, self.heads * self.dim, bias=False)
        self.weights_proj = nn.Linear(c.d_model, self.heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(c.head_dim, self.dim, bias=False)
            self.k_norm = RMSNorm(self.dim, c.rms_eps)
        self.score_probe = nn.Identity()       # hook 지점: indexer 점수 [B, T, W] (mask 전)
        self.candidate_probe = nn.Identity()   # hook 지점: candidate block mask [B, T, W]
        self.select_probe = nn.Identity()      # hook 지점: 고른 압축 칸 [B, T, k] (-1 = 없음)

    def make_keys(self, latent, cos, sin):
        return apply_rope_tail(self.k_norm(self.wk(latent)), cos, sin)

    def forward(self, x, qr, index_k, cos, sin, shared):
        B, T, _ = x.shape
        W = index_k.shape[1]
        q = apply_rope_tail(self.wq_b(qr).unflatten(-1, (self.heads, self.dim)), cos[:, None], sin[:, None])
        w = self.weights_proj(x) * (self.dim ** -0.5 * self.heads ** -0.5)
        s = torch.einsum("bshd,btd->bsht", q, index_k)
        s = self.score_probe((s.relu() * w.unsqueeze(-1)).sum(dim=2))
        lens = (torch.arange(1, T + 1, device=x.device) // self.ratio).unsqueeze(-1)    # 질의마다 볼 수 있는 칸 수
        s = s.masked_fill(torch.arange(W, device=x.device) >= lens, -torch.inf)
        if self.is_candidate_source:
            shared["candidates"] = self.candidate_probe(
                select_candidate_blocks(s, lens, self.c.candidate_topk_blocks, self.c.candidate_block_size))
        elif self.uses_candidates:
            s = s.masked_fill(~shared["candidates"], -torch.inf)
        idx = s.topk(min(self.c.index_topk, W), dim=-1, sorted=False).indices.sort(dim=-1).values
        return self.select_probe(torch.where(idx < lens, idx, -1))


# ---------------------------------------------------------------- Attention

class Attention(nn.Module):
    def __init__(self, c: DeepseekV41Config, layer_id: int):
        super().__init__()
        self.c = c
        self.h, self.d = c.n_head, c.head_dim
        self.mode = c.mode(layer_id)
        self.ratio = c.compress_ratios[layer_id] if self.mode != SLIDING else 0
        self.attn_sink = nn.Parameter(torch.zeros(self.h))
        self.wq_a = nn.Linear(c.d_model, c.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(c.q_lora_rank, c.rms_eps)
        self.wq_b = nn.Linear(c.q_lora_rank, self.h * self.d, bias=False)
        self.wkv = nn.Linear(c.d_model, self.d, bias=False)                 # K = V, head 1개
        self.kv_norm = RMSNorm(self.d, c.rms_eps)
        self.wo_a = GroupedLinear(self.h * self.d // c.o_groups, c.o_groups * c.o_lora_rank, c.o_groups)
        self.wo_b = nn.Linear(c.o_groups * c.o_lora_rank, c.d_model, bias=False)
        self.compressor = Compressor(c, self.ratio) if self.mode == FULL else None
        self.indexer = Indexer(c, layer_id) if (self.ratio and layer_id in c.index_source_layers) else None
        self.attn_probs = nn.Identity()   # hook 지점: [B, H, T, T + W] (앞 T = window 토큰, 뒤 W = 압축 칸)
        self.sink_probe = nn.Identity()   # hook 지점: sink 로 간 확률 [B, H, T]

    def forward(self, x, cos, sin, shared):
        B, T, _ = x.shape
        qr = self.q_norm(self.wq_a(x))
        q = apply_rope_tail(self.wq_b(qr).unflatten(-1, (self.h, self.d)), cos[:, None], sin[:, None])   # [B, T, H, D]
        kv = apply_rope_tail(self.kv_norm(self.wkv(x)), cos, sin)                                       # [B, T, D]
        i, j = torch.arange(T, device=x.device)[:, None], torch.arange(T, device=x.device)[None, :]
        allow = ((j <= i) & (i - j < self.c.window)).expand(B, T, T)

        if self.ratio:
            r, W = self.ratio, T // self.ratio
            if self.mode == FULL:     # 압축 latent → (indexer key, RoPE 후 압축 KV) 를 뒤 층과 공유
                latent = self.compressor(x)
                pc, ps = cos[: W * r: r], sin[: W * r: r]                   # 압축 칸 j 는 위치 j·r
                if self.indexer is not None and self.indexer.owns_k:
                    shared["index_k"] = self.indexer.make_keys(latent, pc, ps)
                shared["compress_kv"] = apply_rope_tail(latent, pc, ps)
            ckv = shared["compress_kv"][:, :W]
            if self.indexer is not None:
                idx = (torch.empty(B, T, 0, dtype=torch.long, device=x.device) if W == 0 else
                       self.indexer(x, qr, shared["index_k"][:, :W], cos, sin, shared))
                shared["topk_idxs"] = idx
            else:                     # reuse: 직전 indexer 의 선택 그대로
                idx = shared["topk_idxs"]
            if self.c.use_indexer:
                sel = torch.zeros(B, T, W + 1, dtype=torch.bool, device=x.device)
                cmask = sel.scatter_(-1, torch.where(idx >= 0, idx, W), True)[..., :W]
            else:
                ready = torch.arange(W, device=x.device)[None, :] < ((torch.arange(T, device=x.device) + 1) // r)[:, None]
                cmask = ready.expand(B, T, W)
            kv = torch.cat([kv, ckv], dim=1)
            allow = torch.cat([allow, cmask], dim=-1)

        logits = torch.einsum("bthd,bsd->bhts", q, kv) * self.d ** -0.5
        logits = logits.masked_fill(~allow[:, None], float("-inf"))
        if self.c.use_sink:
            logits = torch.cat([logits, self.attn_sink.view(1, -1, 1, 1).expand(B, -1, T, 1).to(logits.dtype)], dim=-1)
            probs = logits.softmax(-1)
            self.sink_probe(probs[..., -1])
            probs = probs[..., :-1]
        else:
            probs = logits.softmax(-1)
        probs = self.attn_probs(probs)
        o = torch.einsum("bhts,bsd->bthd", probs.to(kv.dtype), kv)
        o = apply_rope_tail(o, cos[:, None], -sin[:, None])     # V 에도 RoPE 가 들어갔으므로 질의 위치로 되돌림
        return self.wo_b(self.wo_a(o.reshape(B, T, self.c.o_groups, -1)).flatten(2))


# ---------------------------------------------------------------- MoE

class Expert(nn.Module):
    """SwiGLU. up 은 양쪽, gate 는 위쪽만 clamp. 라우팅 가중치는 w2 앞에서 곱한다."""

    def __init__(self, d, inter, limit):
        super().__init__()
        self.limit = limit
        self.w1 = nn.Linear(d, inter, bias=False)
        self.w2 = nn.Linear(inter, d, bias=False)
        self.w3 = nn.Linear(d, inter, bias=False)

    def forward(self, x, weights=None):
        dtype = x.dtype
        g, u = self.w1(x).float(), self.w3(x).float()
        if self.limit > 0:
            u = u.clamp(-self.limit, self.limit)
            g = g.clamp(max=self.limit)
        h = F.silu(g) * u
        if weights is not None:
            h = weights * h
        return self.w2(h.to(dtype))


class Gate(nn.Module):
    def __init__(self, c: DeepseekV41Config):
        super().__init__()
        self.c = c
        self.weight = nn.Parameter(torch.randn(c.n_experts, c.d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(c.n_experts))   # 선택에만 쓰는 부하 분산 bias
        self.route_probe = nn.Identity()   # hook 지점: 선택된 expert [N, k]

    def forward(self, x):
        c = self.c
        s = F.linear(x.float(), self.weight.float()) / c.gate_temp
        s = s.softmax(-1) if c.score_func == "softmax" else s.sigmoid() if c.score_func == "sigmoid" else F.softplus(s).sqrt()
        idx = self.route_probe((s + self.bias).topk(c.top_k, dim=-1)[1])
        w = s.gather(1, idx)
        if c.norm_topk_prob and c.top_k > 1:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return w * c.route_scale, idx


class MoE(nn.Module):
    def __init__(self, c: DeepseekV41Config):
        super().__init__()
        self.c = c
        if c.use_moe:
            self.gate = Gate(c)
            self.experts = nn.ModuleList(Expert(c.d_model, c.moe_inter, c.swiglu_limit) for _ in range(c.n_experts))
        self.shared_experts = Expert(c.d_model, c.moe_inter, c.swiglu_limit)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        y = torch.zeros_like(x, dtype=torch.float32)
        if self.c.use_moe:
            w, idx = self.gate(x)
            for e, expert in enumerate(self.experts):
                tok, slot = torch.where(idx == e)
                if tok.numel() == 0:
                    continue
                y[tok] += expert(x[tok], w[tok, slot, None])
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


# ---------------------------------------------------------------- 층 / 모델

class Layer(nn.Module):
    def __init__(self, c: DeepseekV41Config, i: int):
        super().__init__()
        self.c = c
        self.rope_kind = "comp" if c.compress_ratios[i] else "main"
        self.attn = Attention(c, i)
        self.ffn = MoE(c)
        self.engram = Engram(c, i) if (c.use_engram and i in c.engram_layer_ids) else None
        self.attn_norm = RMSNorm(c.d_model, c.rms_eps)
        self.ffn_norm = RMSNorm(c.d_model, c.rms_eps)
        if c.use_mhc:
            n = c.hc_mult
            mix, hd = (2 + n) * n, n * c.d_model
            self.hc_attn_fn = nn.Parameter(torch.randn(mix, hd) * 0.02)
            self.hc_ffn_fn = nn.Parameter(torch.randn(mix, hd) * 0.02)
            self.hc_attn_base = nn.Parameter(torch.zeros(mix))
            self.hc_ffn_base = nn.Parameter(torch.zeros(mix))
            self.hc_attn_scale = nn.Parameter(torch.ones(3))
            self.hc_ffn_scale = nn.Parameter(torch.ones(3))
            self.attn_hc_probe = nn.Identity()   # hook 지점: attention 쪽 Sinkhorn 혼합 행렬 [B, T, n, n]
            self.ffn_hc_probe = nn.Identity()

    def hc_mixes(self, h, fn, scale, base, probe):
        """줄기 [B,T,n,D] → pre [B,T,n], post [B,T,n], comb [B,T,n,n] (원본 hc_mixes + hc_split_sinkhorn)."""
        c, n = self.c, self.c.hc_mult
        x = h.flatten(2).float()
        m = F.linear(x, fn) * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.rms_eps)
        pre = torch.sigmoid(m[..., :n] * scale[0] + base[:n]) + c.hc_eps
        post = 2 * torch.sigmoid(m[..., n:2 * n] * scale[1] + base[n:2 * n])
        comb = (m[..., 2 * n:] * scale[2] + base[2 * n:]).unflatten(-1, (n, n))
        comb = comb.softmax(-1) + c.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + c.hc_eps)              # Sinkhorn: 열/행 번갈아 정규화
        for _ in range(c.hc_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + c.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + c.hc_eps)
        return pre, post, probe(comb)

    @staticmethod
    def hc_pre(h, pre):
        return torch.sum(pre.unsqueeze(-1) * h.float(), dim=2).to(h.dtype)

    @staticmethod
    def hc_post(x, res, post, comb):
        # 새 줄기 k = post_k·(블록 출력) + Σ_j comb[j,k]·(기존 줄기 j)
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * res.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, h, pre_mix, rope, shared):
        cos, sin = rope[self.rope_kind]
        if not self.c.use_mhc:                         # 일반 pre-norm residual (ablation)
            h = h + self.attn(self.attn_norm(h), cos, sin, shared)
            return h + self.ffn(self.ffn_norm(h)), None
        res = h
        a_pre, a_post, a_comb = self.hc_mixes(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.attn_hc_probe)
        x = self.attn(self.attn_norm(self.hc_pre(h, pre_mix)), cos, sin, shared)
        h = self.hc_post(x, res, a_post, a_comb)
        res = h
        f_pre, f_post, f_comb = self.hc_mixes(h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self.ffn_hc_probe)
        x = self.ffn(self.ffn_norm(self.hc_pre(h, a_pre)))   # FFN 은 attention 입력에서 계산한 pre 를 씀
        return self.hc_post(x, res, f_post, f_comb), f_pre   # f_pre 는 다음 층 attention 이 씀


class DeepseekV41(nn.Module):
    def __init__(self, c: DeepseekV41Config):
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.n_layer))
        self.norm = RMSNorm(c.d_model, c.rms_eps)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.engram_hash = NgramHash(c) if (c.use_engram and c.engram_layer_ids) else None
        self.register_buffer("inv_main", yarn_inv_freq(c.rope_dim, c.rope_theta, 0, 1.0, 0, 0), persistent=False)
        self.register_buffer("inv_comp", yarn_inv_freq(c.rope_dim, c.compress_rope_theta, c.original_seq_len,
                                                       c.rope_factor, c.beta_fast, c.beta_slow), persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, GroupedLinear)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> LMOutput:
        c = self.config
        T = input_ids.shape[1]
        hashes = self.engram_hash(input_ids) if self.engram_hash is not None else None
        h = self.embed(input_ids)
        pre_mix = None
        if c.use_mhc:
            h = h.unsqueeze(2).repeat(1, 1, c.hc_mult, 1)
            pre_mix = h.new_zeros(*h.shape[:3], dtype=torch.float32)
            pre_mix[:, :, 0] = 1.0                     # 첫 층 attention 은 0 번 줄기만
        rope = {"main": rope_table(T, self.inv_main), "comp": rope_table(T, self.inv_comp)}
        shared = {}                                    # 층 간 공유: compress_kv, index_k, topk_idxs, candidates
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(h, hashes[:, :, layer.engram.index, :])
            h, pre_mix = layer(h, pre_mix, rope, shared)
        if c.use_mhc:
            h = Layer.hc_pre(h, pre_mix)
        h = self.norm(h)
        return LMOutput(h, self.head(h))


# ---------------------------------------------------------------- 검증

def _get(c, *names, default=None):
    for n in names:
        v = c.get(n) if isinstance(c, dict) else getattr(c, n, None)
        if v is not None:
            return v
    return default


def config_from_hf(c) -> DeepseekV41Config:
    """원본 inference ModelArgs(inference/config.json 키) 또는 HF config.json 의 text_config 를 받는다."""
    c = _get(c, "text_config", default=c)
    rs = _get(c, "rope_scaling", default={}) or {}
    n = _get(c, "n_layers", "num_hidden_layers")
    return DeepseekV41Config(
        vocab_size=_get(c, "vocab_size"), d_model=_get(c, "dim", "hidden_size"), n_layer=n,
        compress_ratios=list(_get(c, "compress_ratios"))[:n],
        kv_source_layers=list(_get(c, "kv_source_layers", "kv_source_layer_ids")),
        index_source_layers=list(_get(c, "index_source_layers", "index_source_layer_ids")),
        candidate_source_layer=_get(c, "candidate_source_layer", "candidate_source_layer_id", default=-1),
        candidate_topk_blocks=_get(c, "candidate_topk_blocks", default=0),
        candidate_block_size=_get(c, "candidate_block_size", default=0),
        n_head=_get(c, "n_heads", "num_attention_heads"), head_dim=_get(c, "head_dim"),
        rope_dim=_get(c, "rope_head_dim", "qk_rope_head_dim"), q_lora_rank=_get(c, "q_lora_rank"),
        o_groups=_get(c, "o_groups"), o_lora_rank=_get(c, "o_lora_rank"),
        window=_get(c, "window_size", "sliding_window"),
        rope_theta=float(_get(c, "rope_theta")), compress_rope_theta=float(_get(c, "compress_rope_theta")),
        original_seq_len=_get(c, "original_seq_len", default=rs.get("original_max_position_embeddings", 0)),
        rope_factor=float(_get(c, "rope_factor", default=rs.get("factor", 1.0))),
        beta_fast=_get(c, "beta_fast", default=rs.get("beta_fast", 32)),
        beta_slow=_get(c, "beta_slow", default=rs.get("beta_slow", 1)),
        index_heads=_get(c, "index_n_heads"), index_dim=_get(c, "index_head_dim"), index_topk=_get(c, "index_topk"),
        hc_mult=_get(c, "hc_mult"), hc_iters=_get(c, "hc_sinkhorn_iters"), hc_eps=_get(c, "hc_eps"),
        n_experts=_get(c, "n_routed_experts"), top_k=_get(c, "n_activated_experts", "num_experts_per_tok"),
        moe_inter=_get(c, "moe_inter_dim", "moe_intermediate_size"),
        score_func=_get(c, "score_func", "scoring_func", default="sqrtsoftplus"),
        gate_temp=_get(c, "gate_temp", default=1.0), norm_topk_prob=_get(c, "norm_topk_prob", default=True),
        route_scale=_get(c, "route_scale", "routed_scaling_factor"), swiglu_limit=_get(c, "swiglu_limit", default=0.0),
        engram_layer_ids=list(_get(c, "engram_layer_ids", default=[])),
        engram_max_ngram=_get(c, "engram_max_ngram_size", default=1),
        engram_heads=_get(c, "engram_n_heads", default=0), engram_dim=_get(c, "engram_head_dim", default=0),
        engram_vocab=_get(c, "engram_vocab_size", default=0),
        engram_pad_id=_get(c, "engram_pad_id", "engram_pad_token_id", default=2),
        engram_compressed_vocab=_get(c, "engram_compressed_vocab_size", default=0),
        engram_num_embeddings=list(_get(c, "engram_num_embeddings", default=[])),
        rms_eps=_get(c, "norm_eps", "rms_norm_eps"))


def verify_pair(ref) -> dict:
    """ref = 원본 inference Transformer (stage 모듈이 ref.args 로 ModelArgs 를 붙여 둠). 이름이 같아 name_map 불필요.

    원본 forward 는 (sample 한 id, 마지막 위치 logits, MTP 용 hidden) 을 돌려주므로 마지막 위치 logits 끼리 비교하고,
    전 위치는 층별 hook (+ 최종 norm) 으로 비교한다.
    """
    mine = DeepseekV41(config_from_hf(ref.args))
    if mine.engram_hash is not None:   # 토큰 정규화 표는 tokenizer 에서 만들어지므로 원본 것을 복사
        mine.engram_hash.token_map = ref.engram_hash.token_map.clone()
    pairs = []
    for i, layer in enumerate(mine.layers):
        names = ([f"layers.{i}.engram"] if layer.engram is not None else []) + \
                [f"layers.{i}.attn", f"layers.{i}.ffn", f"layers.{i}"]
        pairs += [(n, n) for n in names]
    pairs.append(("norm", "norm"))
    return dict(mine=mine, name_map=None, hook_pairs=pairs,
                out_fn_ref=lambda o: o[1], out_fn_mine=lambda o: o.logits[:, -1])
