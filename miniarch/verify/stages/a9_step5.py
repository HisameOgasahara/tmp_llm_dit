"""A9 · StepFun Step-5-Preview (sliding/DSA 하이브리드 · 층별 partial RoPE · head gate · MoE) 검증 케이스.

원본 = HF stepfun-ai/Step-5-Preview-BF16 `inference/model.py` + `kernel.py` + `vision_encoder.py` (커밋 고정).
transformers main 에는 step5 가 없다 (stepfun 계열은 step3p7 뿐). 원본 kernel.py 를 그대로 import 하고
Triton 커널을 부르는 함수만 같은 수식의 fp32 PyTorch 구현으로 바꿔 끼운다 (kernel.py 안의 순수 PyTorch 진행 코드
— Step5SparseIndexer.project, update_summaries_prefill, score_regions, prefill_metadata, DSALayerCache — 는 원본 그대로).
  - indexer_norm_rope   : q RMSNorm(1+w) / k LayerNorm(E[x²]-mean²) + 앞 rope_dim 채널 NEOX 회전 → 같은 식
  - csa_compress_regions: region 안 채널별 softmax(z) 가중평균 (shift = region 전체 최댓값) → 같은 식, fp8 저장 대신 fp32
  - indexer_logits      : Σ_h w_h·ReLU(q_h·k), e4m3 활성값 반올림은 항등 (fp32 기준값이므로 양자화 오차 제거)
  - prefill_sparse_meta : 질의 region 이전 region 중 top-k (동점은 번호 작은 쪽) + 현재 region 덧붙이기 → 같은 packed 형식
  - sparse_attention_prefill: packed region 의 토큰을 모아 softmax → gather + softmax
  - clamped_swiglu / weighted_topk_gather: fp32 식 그대로 (출력 bf16 고정 대신 입력 dtype)
  - linear_fp8_or_bf16  : bf16 경로의 bf16 cast 제거 (fp8 경로는 쓰지 않음, 호출되면 에러)
  - build_rope_cache    : 같은 식, cos/sin 표를 bf16 대신 fp32 로 저장
  - fused_qknorm_rope   : head_dim 192 전용 → 축소 config(head_dim 24)에서는 호출되지 않음 (호출되면 에러)
커널 외에 바꾼 것 (dtype 저장 형식만): DSA cache 의 region 요약 텐서를 fp8 → fp32. 나머지 bf16 가중치는 .float().
원본 코드를 고치지 않고 붙인 것: forward 를 감싸 [B, T] 입력(+ seq_lens)을 packed 입력으로 바꾸고, 매번 새 cache 를 만들어
prefill 한 번을 돌린 뒤 logits 를 [B, T, V] (padding 위치 0) 로 되돌린다.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from ..cases import BATCH, VOCAB, Case

STAGE = "A9"
SLUG = "step5"
TITLE = "StepFun Step-5-Preview (sliding + DSA · MoE)"
INTRO = """
원본: HF `stepfun-ai/Step-5-Preview-BF16`의 `inference/model.py` + `kernel.py` (커밋 `7734b7f` 고정).
transformers main에는 아직 step5가 없어서 StepFun 공식 추론 코드를 받아 씁니다.
이 코드는 Triton 커널을 쓰므로 **커널을 부르는 함수만** 같은 수식의 fp32 PyTorch 구현으로 바꿔 끼웁니다
(fp8/e4m3 반올림은 항등, bf16 저장은 fp32. 목록은 `miniarch/verify/stages/a9_step5.py` 맨 위 설명).
vision encoder와 MTP 층은 비교하지 않습니다 (텍스트 backbone만).

내 구현: `miniarch/models/step5.py`. 가중치 이름은 실제 체크포인트 키(`model.layers.N...`)와 같습니다.

부품
- sliding 3 : full 1 하이브리드 (92층). sliding 층은 window 512, full 층은 DSA(sparse) attention
- 층마다 다른 RoPE: sliding은 head 전체 회전 · theta 1e4, full은 앞 1/3만 회전 · theta 1e7 + llama3 scaling(full 층만)
- GQA(q 64 / kv 4), q·k에 zero-centered RMSNorm `x·(1+w)`, 출력에 head별 sigmoid gate
- DSA indexer: 토큰 8개 region을 채널별 softmax(z) 가중평균으로 요약하고, kv group마다 `Σ w·ReLU(q·요약)`으로
  이전 region top-512 + 현재 region만 봄
- MoE: expert 352개 중 8개 (fp32 sigmoid + 선택용 bias, 정규화 × 3.0) + shared expert, 앞 3층은 dense.
  SwiGLU는 gate·up 모두 clamp(±7). residual은 fp32로 누적

축소 config (모든 경로가 한 번씩 쓰이도록 고름)
- 5층: [sliding+dense, full(DSA)+dense, sliding+MoE, sliding+MoE, full(DSA)+MoE]
- head_dim 24: full 층 회전 비율 1/3 → 8채널, 0.3 → 7 → 짝수로 6채널 (반올림 경로). theta 1e4 / 1e6
- llama3 scaling: original 512, factor 4, 대역 1~4 → 회전 주파수가 그대로 / 보간 / 1/4 세 구간에 모두 걸림
- DSA: region 4토큰, top-3, indexer head 4개(kv group 2 × 2), proxy 32, indexer RoPE 8채널
- 시퀀스 45 / 38 (오른쪽 padding): region이 최대 11개라 top-3이 실제로 고르고 버림, 덜 찬 마지막 region,
  요청마다 다른 길이(packed 경계), window 12 밖 토큰까지 지나감
- SwiGLU limit을 0.6~1.0으로 작게 둬서 clamp가 실제로 걸림. expert 8개 중 3개
- 원본이 `torch.empty`/0으로 만드는 가중치는 0이 아닌 값으로 채웁니다.
- 층별(attention / routed MoE / shared / dense MLP / 층 출력)과 최종 norm, 전 위치 logits를 비교합니다.
"""
PIP = []
HEADER = [("stepfun-ai/Step-5-Preview-BF16", r"^model\.layers\.(2|3)\.", "")]
CAUSAL = True
SELF_CHECKS = [
    ("""
### 커널 대체 확인

원본 비교의 기준값이 커널 대체에 달려 있으므로, 대체한 region 선택(`prefill_sparse_meta`)이 정의
(질의 region 이전 region 중 점수 top-k, 동점은 번호 작은 쪽 + 현재 region)와 같은지,
`sparse_attention_prefill`이 고른 토큰에 대한 softmax와 같은지 따로 확인합니다. 0 동점을 일부러 많이 넣습니다.
GPU에서 원본 Triton 커널이 돌아가면 (GPU 세대에 따라 안 될 수 있음) 원본과도 직접 비교합니다.
""", """
V.set_deterministic(0)
rows_, W_, R_, topk_ = 40, 9, 4, 3
pos_ = torch.randint(0, W_ * R_ + R_, (rows_,), dtype=torch.int32)
sc_ = (torch.randint(-2, 3, (rows_, W_)).float() * 0.5).clamp_min(0)       # ReLU 처럼 0 동점이 많음
table_ = torch.arange(8, dtype=torch.int32).view(1, -1)                     # 모든 요청이 쓰는 page 표 하나
packed_, cnt_ = S.prefill_sparse_meta(sc_, pos_, table_, None, topk=topk_, region_size=R_, regions_per_page=4)
ok_ = True
for r in range(rows_):
    p = int(pos_[r]); vis = min(p // R_, W_)
    want = sorted(sorted(range(vis), key=lambda j: (-float(sc_[r, j]), j))[:topk_]) + [p // R_]
    got = [x & 0xFFFFFF for x in packed_[r, :int(cnt_[r])].tolist() if x >= 0]
    ok_ = ok_ and got == want
report("region 선택 대체 == 정의 (동점 포함)", ok_, 0.0)

G_, H_, D_ = 2, 4, 16
Tq_ = rows_ // G_
q_, kc_, vc_ = torch.randn(Tq_, H_, D_), torch.randn(64, G_, D_), torch.randn(64, G_, D_)
o1, _ = S.sparse_attention_prefill(q_, kc_, vc_, packed_, cnt_, num_kv_groups=G_, region_size=R_, softmax_scale=0.3)
o2 = torch.zeros_like(q_)
for g in range(G_):
    for t in range(Tq_):
        toks = []
        for x in packed_[g * Tq_ + t, :int(cnt_[g * Tq_ + t])].tolist():
            if x >= 0:
                toks += [(x & 0xFFFFFF) * R_ + o for o in range(x >> 24)]
        idx = torch.tensor(toks)
        for h in range(g * (H_ // G_), (g + 1) * (H_ // G_)):
            o2[t, h] = ((kc_[idx, g] @ q_[t, h]) * 0.3).softmax(0) @ vc_[idx, g]
err = (o1 - o2).abs().max().item()
report("sparse_attention_prefill 대체 == 정의", err < 1e-5, err)

S.load_reference()
O_ = S._LOADED["orig"]
if dev == "cuda":
    try:
        p2, c2 = O_["prefill_sparse_meta"](sc_.to(dev), pos_.to(dev), table_.to(dev), None,
                                            topk=topk_, region_size=R_, regions_per_page=4)
        same = torch.equal(p2.cpu(), packed_) and torch.equal(c2.cpu(), cnt_)
        report("region 선택: 원본 Triton == 대체", same, 0.0)
        o3, _ = O_["sparse_attention_prefill"](q_.to(dev), kc_.to(dev), vc_.to(dev), p2, c2,
                                               num_kv_groups=G_, region_size=R_, softmax_scale=0.3)
        err = (o3.cpu() - o1).abs().max().item()
        report("sparse attention: 원본 Triton == 대체 (tf32 허용)", err < 5e-3, err)
    except Exception as e:
        print("⚠️ 원본 Triton 커널을 이 GPU/버전에서 실행하지 못해 건너뜀:", type(e).__name__, str(e)[:200])
"""),
    ("""
### DSA 선택 · attention map 확인 (내 구현)

- full(DSA) 층: 질의 위치 p 마다 고른 과거 region 수 = min(top-k, p // region) 이어야 합니다 (kv group 마다).
- sliding 층: 미래 칸과 window 밖 칸의 확률은 정확히 0.
- 모든 층: 확률 행 합 = 1.
""", """
from miniarch.models import step5 as S5
ref, inp = S.ref()
m = S5.verify_pair(ref)["mine"].to(dev).eval()
c = m.config
names = [n for n, _ in m.named_modules() if n.endswith(("attn_probs", "select_probe"))]
x = inp["input_ids"].to(dev)
T_ = x.shape[1]
pos = torch.arange(T_, device=dev)
with V.capture(m, names) as st, torch.no_grad():
    m(x)
for i, layer in enumerate(m.model.layers):
    a = layer.self_attn
    p = st[f"model.layers.{i}.self_attn.attn_probs"]
    err = (p.sum(-1) - 1).abs().max().item()
    if a.sparse:
        sel = st[f"model.layers.{i}.self_attn.select_probe"]
        n_sel = sel.sum(-1).float()
        want = (pos // c.region).clamp_max(c.index_topk).float()
        seen = (p > 0).sum(-1).float().mean().item()
        print(f"layer {i} DSA  rope {a.span}채널  질의당 고른 region 평균 {n_sel.mean():.2f}개  평균 본 토큰 {seen:.1f}")
        report(f"layer {i} 고른 region 수 = min(top-k, p//R)", bool((n_sel == want).all()), 0.0)
    else:
        outside = (pos[None] > pos[:, None]) | (pos[:, None] - pos[None] >= c.window)
        leak = p[..., outside].abs().max().item()
        print(f"layer {i} sliding  rope {a.span}채널  window {c.window}")
        report(f"layer {i} window 밖·미래 확률 0", leak == 0, leak)
    report(f"layer {i} 확률 행 합 = 1", err < 1e-5, err)
"""),
    ("""
### padding 무관성

원본은 요청을 이어 붙인 packed 입력이라 padding이 없습니다. 내 구현은 [B, T]에 오른쪽 padding을 두는데,
causal이라 padding이 앞쪽 출력에 영향을 주지 않아야 합니다. 묶음으로 돌린 결과와 요청 하나씩 돌린 결과를 비교합니다.
""", """
from miniarch.models import step5 as S5
ref, inp = S.ref()
m = S5.verify_pair(ref)["mine"].to(dev).eval()
x, L = inp["input_ids"].to(dev), inp["seq_lens"].to(dev)
with torch.no_grad():
    full = m(x, L).logits
    for b in range(x.shape[0]):
        l = int(L[b])
        solo = m(x[b:b + 1, :l]).logits
        err = (full[b, :l] - solo[0]).abs().max().item()
        report(f"요청 {b} (길이 {l}) 단독 == 묶음", err < 1e-4, err)
"""),
    ("""
### ablation 스위치

나중에 부품별 비교 실험에 쓸 스위치를 하나씩 끄고 forward가 정상(유한값)인지 확인합니다.
""", """
import dataclasses
from miniarch.models import step5 as S5
ref, inp = S.ref()
base = S5.config_from_hf(ref.config)
x = inp["input_ids"].to(dev)
for sw in ["use_sparse", "use_sliding", "use_attn_gate", "use_partial_rope", "use_rope_scaling",
           "use_moe", "use_shared_expert", "use_router_bias", "use_swiglu_limit"]:
    mm = S5.Step5(dataclasses.replace(base, **{sw: False})).to(dev).eval()
    with torch.no_grad():
        out = mm(x).logits
    report(f"{sw}=False forward", bool(torch.isfinite(out).all()), 0.0)
"""),
]

REPO = "stepfun-ai/Step-5-Preview-BF16"
REVISION = "7734b7ffac118d70d8398bfa3eb0fab007eb45b1"
_LOADED: dict = {}


# ---------------------------------------------------------------- kernel.py 대체 (fp32 PyTorch)

def _norm_rope(x, w, b, cos, sin, positions, *, num_heads, head_dim, rotary_dim, eps, weight_bias, subtract_mean):
    """_indexer_norm_rope_kernel: head 별 정규화((x-mean)·rstd·(w+wb)[+b]) 후 앞 2·rotary_dim 채널 회전."""
    T = x.shape[0]
    v = x.view(T, num_heads, head_dim).float()
    ss = (v * v).sum(-1, keepdim=True)
    if subtract_mean:
        mean = v.sum(-1, keepdim=True) / head_dim
        var = ss / head_dim - mean * mean
    else:
        mean, var = 0.0, ss / head_dim
    y = (v - mean) * torch.rsqrt(var + eps) * (w.float() + weight_bias)
    if b is not None:
        y = y + b.float()
    y = y.to(x.dtype).float()                      # 커널: 회전 전 출력 dtype 으로 한 번 반올림
    c = cos[positions.long()].float()[:, None]
    s = sin[positions.long()].float()[:, None]
    re, im = y[..., :rotary_dim], y[..., rotary_dim:2 * rotary_dim]
    out = torch.cat((re * c - im * s, re * s + im * c, y[..., 2 * rotary_dim:]), -1)
    return out.to(x.dtype).reshape(x.shape)


def indexer_norm_rope(index_q, index_k, index_z, q_norm_weight, k_norm_weight, k_norm_bias, cos, sin, positions, *,
                      head_dim, num_q_heads, num_k_heads, rotary_dim, eps=1e-6, q_norm_weight_bias=1.0):
    q = _norm_rope(index_q, q_norm_weight, None, cos, sin, positions, num_heads=num_q_heads, head_dim=head_dim,
                   rotary_dim=rotary_dim, eps=eps, weight_bias=q_norm_weight_bias, subtract_mean=False)
    k = _norm_rope(index_k, k_norm_weight, k_norm_bias, cos, sin, positions, num_heads=num_k_heads, head_dim=head_dim,
                   rotary_dim=rotary_dim, eps=eps, weight_bias=0.0, subtract_mean=True)
    return q, k, index_z.contiguous()


def csa_compress_regions(index_k, index_z, token_start, token_count, *, region_size=8):
    """[regions, heads, P]: s_d = Σ_t exp(z_td - m) k_td / Σ_t exp(z_td - m), m = (region, head) 전체 최댓값."""
    off = torch.arange(region_size, device=index_k.device)
    ok = off[None] < token_count.long()[:, None]                                   # [regions, R]
    idx = torch.where(ok, token_start.long()[:, None] + off[None], torch.zeros_like(off)[None])
    k = index_k[idx].float()                                                       # [regions, R, heads, P]
    z = index_z[idx].float().masked_fill(~ok[:, :, None, None], float("-inf"))
    m = z.amax(dim=(1, 3), keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    w = torch.where(ok[:, :, None, None], torch.exp(z - m), torch.zeros_like(z))
    den, num = w.sum(1), (w * k).sum(1)
    summary = torch.where(den > 0, num / den.clamp_min(1e-20), torch.zeros_like(num))
    return summary, summary                                                        # fp8 사본 대신 fp32 그대로


def round_activations_e4m3(tensor):
    return tensor                                                                  # e4m3 반올림 → 항등


def indexer_logits(index_q, weights, index_k, *, block_q=64, block_k=128):
    """[groups * seq_q, seq_k] (group-major): Σ_h relu(q_h · k) · w_h, head 순서대로 fp32 누적."""
    seq_q, groups, hpg, _ = index_q.shape
    q, k, w = index_q.float(), index_k[:, 0].float(), weights.float()
    acc = torch.zeros(groups, seq_q, k.shape[0], device=index_q.device)
    for h in range(hpg):
        s = torch.einsum("qgp,kp->gqk", q[:, :, h], k)
        acc = acc + s.clamp_min(0) * w[:, :, h].t()[:, :, None]
    return acc.reshape(groups * seq_q, -1)


def prefill_sparse_meta(logits, q_positions, block_table, request_indices=None, *, topk, region_size=8,
                        regions_per_page):
    """[rows, topk+1] int32 (phys | valid_tokens << 24, 빈칸 -1), counts = min(현재 region, topk) + 1.

    history = 현재 region 이전 region 중 점수 top-k (동점은 region 번호 오름차순), 오름차순으로 기록 후
    slot min(현재 region, topk) 에 현재 region (causal 로 보이는 토큰 수와 함께) 을 넣는다."""
    rows, W = logits.shape
    dev = logits.device
    pos = q_positions.long()
    req = torch.zeros(rows, dtype=torch.long, device=dev) if request_indices is None else request_indices.long()
    table = block_table.long()
    if table.shape[0] == 1:          # 한 줄짜리 표 = 모든 요청 공유
        req = torch.zeros_like(req)
    visible = (pos // region_size).clamp(0, W)
    ar = torch.arange(W, device=dev)
    valid = ar[None] < visible[:, None]
    order = torch.sort(logits.float().masked_fill(~valid, float("-inf")), dim=-1, descending=True, stable=True).indices
    rank = torch.empty_like(order).scatter_(1, order, ar.expand(rows, W).contiguous())
    sel = valid & (rank < visible.clamp_max(topk)[:, None])
    slot = sel.long().cumsum(-1) - 1

    def pack(r, region):
        page = table[req[r], region // regions_per_page]
        phys = page * regions_per_page + region % regions_per_page
        vt = (pos[r] + 1 - region * region_size).clamp(0, region_size)
        return torch.where(page >= 0, phys | (vt << 24), torch.full_like(phys, -1))

    packed = torch.full((rows, topk + 1), -1, dtype=torch.long, device=dev)
    r_i, c_i = sel.nonzero(as_tuple=True)
    packed[r_i, slot[r_i, c_i]] = pack(r_i, c_i)
    cur = (pos // region_size).clamp_min(0)
    hc = cur.clamp_max(topk)
    rr = torch.arange(rows, device=dev)
    packed[rr, hc] = pack(rr, cur)
    return packed.int(), (hc + 1).int()


def sparse_attention_prefill(query, key_cache, value_cache, packed_regions, region_counts, *, num_kv_groups,
                             region_size=8, softmax_scale=None, block_regions=16):
    """질의마다 (kv group 별) packed region 의 토큰만 모아 softmax. 고른 칸이 없으면 0, lse=-inf."""
    Tq, H, D = query.shape
    G = num_kv_groups
    S_ = packed_regions.shape[1]
    scale = softmax_scale if softmax_scale is not None else D ** -0.5
    meta = packed_regions.view(G, Tq, S_).long()
    slot_ok = torch.arange(S_, device=query.device)[None, None] < region_counts.view(G, Tq, 1).long()
    phys, valid = meta & ((1 << 24) - 1), meta >> 24
    live = slot_ok & (meta >= 0) & (valid > 0)
    off = torch.arange(region_size, device=query.device)
    tok_ok = (live[..., None] & (off < valid[..., None])).flatten(2)                   # [G, Tq, S*R]
    tok = torch.where(tok_ok, (phys[..., None] * region_size + off).flatten(2), torch.zeros_like(tok_ok, dtype=torch.long))
    kc, vc = key_cache.transpose(0, 1), value_cache.transpose(0, 1)                   # [G, N, D]
    gi = torch.arange(G, device=query.device)[:, None, None]
    k, v = kc[gi, tok].float(), vc[gi, tok].float()                                   # [G, Tq, L, D]
    q = query.view(Tq, G, H // G, D).permute(1, 0, 2, 3).float()                      # [G, Tq, h, D]
    s = torch.einsum("gqhd,gqld->gqhl", q, k) * scale
    s = s.masked_fill(~tok_ok[:, :, None, :], float("-inf"))
    mx = s.amax(-1, keepdim=True)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    p = torch.exp(s - mx)
    den = p.sum(-1, keepdim=True)
    o = torch.einsum("gqhl,gqld->gqhd", p, v)
    o = torch.where(den > 0, o / den.clamp_min(1e-30), torch.zeros_like(o))
    lse = torch.where(den[..., 0] > 0, mx[..., 0] + torch.log(den[..., 0].clamp_min(1e-30)),
                      torch.full_like(den[..., 0], float("-inf")))
    out = o.permute(1, 0, 2, 3).reshape(Tq, H, D).to(query.dtype)
    return out, lse.permute(1, 0, 2).reshape(Tq, H)


def clamped_swiglu(gate, up, limit=7.0):
    return (F.silu(gate.float()).clamp(max=float(limit)) * up.float().clamp(-float(limit), float(limit))).to(gate.dtype)


def weighted_topk_gather(contributions, weights):
    acc = torch.zeros(contributions.shape[0], contributions.shape[2], device=contributions.device)
    for k in range(contributions.shape[1]):
        acc = acc + contributions[:, k].float() * weights[:, k].float().unsqueeze(-1)
    return acc.to(contributions.dtype)


def linear_fp8_or_bf16(x, weight, weight_scale_inv):
    if weight_scale_inv is not None:
        raise NotImplementedError("fp8 expert 경로는 쓰지 않음 (양자화 없는 config)")
    return F.linear(x, weight.to(x.dtype))


def fused_qknorm_rope(*a, **kw):
    raise NotImplementedError("fused_qknorm_rope 는 head_dim 192 전용 (축소 config 에서는 호출되지 않아야 함)")


_REPLACED = {f.__name__: f for f in (indexer_norm_rope, csa_compress_regions, round_activations_e4m3, indexer_logits,
                                     prefill_sparse_meta, sparse_attention_prefill, clamped_swiglu,
                                     weighted_topk_gather, linear_fp8_or_bf16, fused_qknorm_rope)}


def load_reference():
    """원본 kernel.py / vision_encoder.py / model.py 를 받아, kernel 의 Triton 호출 함수만 바꾼 뒤 model 을 import."""
    if _LOADED:
        return _LOADED["model"], _LOADED["kernel"]
    import importlib.util
    from huggingface_hub import hf_hub_download
    paths = {n: hf_hub_download(REPO, f"inference/{n}.py", revision=REVISION)
             for n in ("kernel", "vision_encoder", "model")}

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    bare = ("kernel", "vision_encoder")
    saved = {n: sys.modules.get(n) for n in bare}
    try:
        K = load("kernel", paths["kernel"])
        orig = {n: getattr(K, n) for n in list(_REPLACED) + ["build_rope_cache"]}
        for n, f in _REPLACED.items():
            setattr(K, n, f)
        build = orig["build_rope_cache"]

        def build_rope_cache(**kw):          # 같은 식, 표만 bf16 대신 fp32 로 저장
            kw["dtype"] = torch.float32
            return build(**kw)
        K.build_rope_cache = build_rope_cache
        load("vision_encoder", paths["vision_encoder"])
        M = load("step5_inference_model", paths["model"])   # from kernel import ... 가 바꾼 함수를 잡는다
    finally:   # 흔한 이름이라 전역 sys.modules 는 원래대로 (model 모듈은 이미 객체를 잡고 있음)
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v
    _LOADED.update(model=M, kernel=K, orig=orig)
    return M, K


def _wrap_forward(m, M, K):
    """원본 forward(packed ids, cache, seq_lens) 를 forward(input_ids [B, T], seq_lens=None) 로 감싼다.
    매 호출마다 새 cache 로 prefill 한 번. 반환: ([B, T, V] logits, padding 위치 0)."""
    orig = m.forward

    def forward(input_ids, seq_lens=None):
        B, T = input_ids.shape
        dev = input_ids.device
        lens = torch.full((B,), T, device=dev) if seq_lens is None else seq_lens.to(dev)
        lens = lens.to(torch.int32)
        mask = torch.arange(T, device=dev)[None] < lens[:, None]
        cache = M.Step5Cache(m.config, batch=B, max_tokens=T, device=dev, dtype=m.model.embed_tokens.weight.dtype)
        for lc in cache.layers:   # region 요약 저장: fp8 → fp32
            if isinstance(lc, K.DSALayerCache):
                lc.summary = torch.zeros(lc.summary.shape, device=dev, dtype=torch.float32)
        m.model.build_rope(dev)
        logits = orig(input_ids[mask], cache, lens, return_all_logits=True).logits
        full = logits.new_zeros(B, T, logits.shape[-1])
        full[mask] = logits
        return (full,)

    m.forward = forward


def ref():
    M, K = load_reference()
    L = 5
    sparse = dict(enabled=True, proxy_dim=32, sparse_indexer_rope_dim=8, sparse_indexer_use_rope=True,
                  sparse_indexer_num_heads=4, sparse_indexer_num_k_heads=1, sparse_indexer_q_norm_type="rmsnorm",
                  sparse_indexer_k_norm_type="layernorm", sparse_indexer_csa_z_norm_type="none",
                  sparse_indexer_softmax_variant="ssmax", sparse_indexer_ssmax_s_granularity="q_head",
                  num_provider_groups=2, topk=3, region_block_size=4, compression_method="csa_block_compress",
                  attention_impl="sparse_gqa", apply_to_layer_types=["full_attention"])
    cfg = M.Step5Config(
        hidden_size=64, intermediate_size=96, num_hidden_layers=L, vocab_size=VOCAB,
        num_attention_heads=4, num_attention_groups=2, head_dim=24, rms_norm_eps=1e-5, norm_dtype="float32",
        fp32_residual_connection=True, use_head_wise_attn_gate=True,
        rope_theta=[1e4, 1e6, 1e4, 1e4, 1e6], partial_rotary_factors=[1.0, 1 / 3, 1.0, 1.0, 0.3],
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=12,
        rope_scaling={"rope_type": "llama3", "factor": 4.0, "original_max_position_embeddings": 512,
                      "low_freq_factor": 1.0, "high_freq_factor": 4.0},
        yarn_only_types=["full_attention"], max_position_embeddings=128,
        use_moe=True, moe_intermediate_size=32, share_expert_dim=48, moe_num_experts=8, moe_top_k=3,
        moe_layer_list=[2, 3, 4], use_moe_router_bias=True, need_fp32_gate=True, norm_expert_weight=True,
        moe_router_scaling_factor=3.0, swiglu_limits=[1.0, 0.8, 1.0, 0.6, 0.8],
        swiglu_limits_shared=[0.9, 0.9, 0.7, 1.0, 0.6], num_nextn_predict_layers=0, sparse_config=sparse,
        tie_word_embeddings=False, initializer_range=0.02)
    m = M.Step5ForCausalLM(cfg)
    m.float()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():    # 원본은 torch.empty / 0 으로 만들기만 하므로 직접 초기화 (활성값이 O(1) 이 되도록)
        for name, p in m.named_parameters():
            r = torch.randn(p.shape, generator=g)
            if name.endswith("sparse_indexer_k_norm.weight"):      # 일반 LayerNorm
                p.copy_(1 + 0.1 * r)
            elif name.endswith("sparse_indexer_k_norm.bias"):
                p.copy_(0.1 * r)
            elif name.endswith(("norm.weight", "layernorm.weight")):   # zero-centered RMSNorm (x·(1+w))
                p.copy_(0.1 * r)
            elif name.endswith("router_bias"):
                p.copy_(0.1 * r)
            elif name.endswith("embed_tokens.weight"):
                p.copy_(r)
            else:
                p.copy_(r * p.shape[-1] ** -0.5)
    _wrap_forward(m, M, K)
    T = 45
    ids = torch.randint(0, VOCAB, (BATCH, T), generator=torch.Generator().manual_seed(0))
    lens = torch.tensor([T, T - 7] + [T] * (BATCH - 2), dtype=torch.int32)
    return m, {"input_ids": ids, "seq_lens": lens}


CASE = Case(STAGE, "Step-5 전체: sliding/DSA 하이브리드 + 층별 partial RoPE(llama3) + head gate + clamped MoE",
            "Step-5-Preview", "HF stepfun-ai/Step-5-Preview-BF16 inference/model.py@7734b7f",
            ref, "miniarch.models.step5", atol=1e-4)
