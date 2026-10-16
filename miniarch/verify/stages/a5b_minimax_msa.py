"""A5b · MiniMax-M3 MSA (블록 top-k sparse attention) 검증 케이스.

원본 전체 모델 코드가 없어서 attention 층 단위로 비교한다.
RefMSAAttention 의 핵심 계산은 MiniMax-AI/MSA 저장소(고정 commit)의 순수 PyTorch reference 함수를
소스 그대로 받아 실행하고, 그 앞뒤(투영·norm·RoPE·top-k 규칙)는 sglang minimax_m3 코드를 옮겨 적었다.
"""
from __future__ import annotations

import functools
from types import SimpleNamespace

import torch
from torch import nn

from ..cases import BATCH, VOCAB, Case

STAGE = "A5b"
SLUG = "minimax_msa"
TITLE = "MiniMax-M3 MSA (블록 top-k sparse attention)"
INTRO = """
원본
- 정답 함수: github `MiniMax-AI/MSA` @ `80434d7` 의 순수 PyTorch reference
  - `sparse_attention_ref` (`python/fmha_sm100/cute/test_sparse_atten.py`): 고른 블록만 보는 GQA attention. 커널 테스트의 golden.
  - `ref_tile_max_score` (`tests/regression/test_correctness.py`): 128 토큰 블록별 최대 q·k 점수 (index 가지의 블록 점수).
  - 두 파일은 커널 패키지를 import 하므로 import 하지 않고, 함수 소스만 잘라 실행합니다.
- 층 연결: sglang `models/minimax_m3.py` (투영 이름, gemma RMSNorm, partial RoPE, index 가지)와
  `minimax_sparse/prefill/flash_with_topk_idx.py` 의 top-k 규칙(local 블록 강제, 오름차순, -1 채움)을 옮겨 적은 `RefMSAAttention`.
- HF `MiniMaxAI/MiniMax-M3` 는 config 만 있고 modeling 코드가 없어서, **모델 전체가 아니라 attention 층 단위로 비교**합니다.

내 구현: `miniarch/models/minimax_msa.py` (`MSAAttention`, 작은 LM `MiniMaxMSA`).

부품
- Main 가지: GQA + head 별 gemma RMSNorm + partial RoPE
- Index 가지: 그룹당 index query 1개 + 공유 index key 1개 → 블록 max 점수 → 그룹별 top-k (local 블록 강제)
- Main 가지는 고른 블록 안의 causal 토큰만 attention

축소 config 선택 이유
- `ref_tile_max_score` 가 블록 크기 128을 고정해 두어서 block 128, 시퀀스 512 → 블록 4개.
- topk 2 (= local 1 + 고르는 블록 1): 블록 0 질의는 볼 블록이 topk보다 적어 -1 채움 경로, 블록 1은 전부 선택,
  블록 2·3은 실제 선택이 일어납니다. KV head 2개 → 그룹 2개가 서로 다른 블록을 고르는지도 비교합니다.
- hook 비교: 블록 점수(`index_score`), 고른 블록(`block_select`), 최종 출력.
- 내 구현은 attention 층이라 자동 causal 검사 대신, 자체 점검에서 LM 전체의 causal 누수를 검사합니다.
"""
PIP = []
HEADER = [("MiniMaxAI/MiniMax-M3", r"language_model\.model\.layers\.(0|3)\.(self_attn|input_layernorm)", "")]
CAUSAL = False
SELF_CHECKS = [
    ("""
### 내 LM 전체 causal 누수

dense 1층 + MSA 1층 LM. 시퀀스 800 (블록 7개)의 뒤쪽 절반(400~)을 바꿔도 앞쪽 출력이 그대로여야 합니다.
앞쪽 질의 블록 2·3은 실제로 블록을 골라야 하고, 블록 3(384~511)은 과거·미래 토큰이 섞여 있어
블록 점수가 질의 위치까지의 키만 써야 통과합니다.
""", """
from miniarch.models import minimax_msa as M
V.set_deterministic(0)
c = M.config_from_hf(S.hf_config())
lm = M.MiniMaxMSA(c).to(dev).eval()
r = V.causal_leak_test(lambda x: lm(x).last_hidden_state, vocab_size=c.vocab_size, seq_len=800, device=dev, atol=1e-4)
report("내 LM " + r.name, r.ok, r.max_abs)
"""),
    ("""
### topk ≥ 블록 수 → dense attention 과 같음 / topk 작으면 달라짐

모든 블록을 고르면 MSA는 보통 causal GQA와 같아야 합니다 (`use_sparse=False` 와 비교).
반대로 topk 2면 결과가 달라져야 sparse 경로가 실제로 쓰인 것입니다.
""", """
import dataclasses
from miniarch.models import minimax_msa as M
V.set_deterministic(0)
c = M.config_from_hf(S.hf_config())
full = M.MiniMaxMSA(dataclasses.replace(c, topk=16)).to(dev).eval()
dense = M.MiniMaxMSA(dataclasses.replace(c, use_sparse=False)).to(dev).eval()
small = M.MiniMaxMSA(c).to(dev).eval()
dense.load_state_dict(full.state_dict()); small.load_state_dict(full.state_dict())
x = torch.randint(0, c.vocab_size, (2, 400), device=dev)
r = V.same_function_test("topk 16 (블록 4개) vs dense", lambda: full(x).logits, lambda: dense(x).logits, atol=1e-5)
report(r.name, r.ok, r.max_abs)
with torch.no_grad():
    d = (small(x).logits - dense(x).logits).abs().max().item()
report("topk 2 는 dense 와 달라야 함", d > 1e-4, d)
"""),
    ("""
### 블록 선택 · attention map hook

init 블록 1개, topk 3, 시퀀스 600 (블록 5개)으로 바꿔 init 강제 경로까지 봅니다.
- 고른 블록 수 = min(topk, 질의 블록 + 1), init 블록(0)과 local 블록(질의가 속한 블록)은 항상 포함
- `attn_probs` 는 고른 블록 밖에서 0, 행 합 1
""", """
import dataclasses
from miniarch.models import minimax_msa as M
V.set_deterministic(0)
c = dataclasses.replace(M.config_from_hf(S.hf_config()), init_blocks=1, topk=3)
lm = M.MiniMaxMSA(c).to(dev).eval()
i = c.layer_types.index("sparse_attention")
pre = f"layers.{i}.self_attn."
x = torch.randint(0, c.vocab_size, (2, 600), device=dev)
with V.capture(lm, [pre + "block_select", pre + "attn_probs", pre + "index_score"]) as st, torch.no_grad():
    lm(x)
sel, p = st[pre + "block_select"], st[pre + "attn_probs"]          # [B, G, T, NB], [B, H, T, T]
T, bs = x.shape[1], c.block_size
qb = torch.arange(T, device=dev) // bs
print("block_select", tuple(sel.shape), " attn_probs", tuple(p.shape), " index_score", tuple(st[pre + "index_score"].shape))
report("고른 블록 수 = min(topk, 질의 블록+1)", bool((sel.sum(-1) == (qb + 1).clamp(max=c.topk)).all()))
report("init 블록 항상 포함", bool(sel[..., 0].all()))
report("local 블록 항상 포함", bool(sel.gather(-1, qb.view(1, 1, T, 1).expand(*sel.shape[:3], 1)).all()))
tok = sel.repeat_interleave(bs, -1)[..., :T].repeat_interleave(c.n_head // c.n_kv_head, 1)
leak = p.masked_fill(tok, 0).abs().max().item()
report("고른 블록 밖 확률 = 0", leak == 0, leak)
row = (p.sum(-1) - 1).abs().max().item()
report("attention 행 합 = 1", row < 1e-5, row)
last = int(qb.max())
for g in range(sel.shape[1]):   # 그룹마다 고르는 블록이 다른지 (마지막 질의 블록 기준, 블록별 선택 비율)
    print(f"그룹 {g}:", [round(v, 2) for v in sel[:, g, qb == last].float().mean((0, 1)).tolist()])
"""),
    ("""
### index 가지 KL 손실 · 기울기 격리

`index_kl=True` 로 학습 모드 forward → `kl_loss.backward()`.
teacher(main 확률)는 detach, index 입력은 stopgrad(x) 이므로 기울기는 index 투영·norm 에만 가야 합니다 (논문 식 11).
""", """
import dataclasses
from miniarch.models import minimax_msa as M
V.set_deterministic(0)
c = dataclasses.replace(M.config_from_hf(S.hf_config()), index_kl=True)
lm = M.MiniMaxMSA(c).to(dev).train()
x = torch.randint(0, c.vocab_size, (2, 400), device=dev)
lm(x)
kl = lm.kl_loss
kl.backward()
leak = [n for n, q in lm.named_parameters() if ".index_" not in n and q.grad is not None and q.grad.abs().max() > 0]
got = [n for n, q in lm.named_parameters() if ".index_" in n and q.grad is not None and q.grad.abs().max() > 0]
print("KL =", kl.item(), " 기울기 받은 index 파라미터:", got)
report("KL ≥ 0 이고 유한", bool(torch.isfinite(kl)) and kl.item() >= -1e-6, kl.item())
report("KL 기울기는 index 가지에만", len(leak) == 0 and len(got) > 0, float(len(leak)))
"""),
]

# ---------------------------------------------------------------- 원본 reference 함수 (MSA 저장소, 고정 commit)

MSA_COMMIT = "80434d7f67877c6570ca19cac444b84bc9855dac"
_REF_SOURCES = [
    ("python/fmha_sm100/cute/test_sparse_atten.py", "sparse_attention_ref"),
    ("tests/regression/test_correctness.py", "ref_tile_max_score"),
]


@functools.lru_cache(maxsize=1)
def ref_functions() -> dict:
    """MSA 저장소 파일을 받아 reference 함수 정의만 잘라 실행 → {이름: 함수}.

    파일 전체는 커널 패키지(interface, fmha_sm100, pytest)를 import 해서 Colab T4 에서 못 불러오므로
    ast 로 해당 FunctionDef 만 꺼낸다. 함수 본문은 한 글자도 바꾸지 않는다.
    """
    import __future__
    import ast
    import math
    import os
    import urllib.request
    from typing import Optional

    cache = os.path.join(os.path.expanduser("~"), ".cache", "miniarch", f"msa_{MSA_COMMIT[:12]}")
    os.makedirs(cache, exist_ok=True)
    ns = {"torch": torch, "math": math, "Optional": Optional, "BLK_KV": 128}   # BLK_KV: 기본 인자값
    for path, name in _REF_SOURCES:
        local = os.path.join(cache, path.replace("/", "__"))
        if not os.path.exists(local):
            url = f"https://raw.githubusercontent.com/MiniMax-AI/MSA/{MSA_COMMIT}/{path}"
            urllib.request.urlretrieve(url, local)
        src = open(local, encoding="utf-8").read()
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
        code = compile(ast.Module(body=[fn], type_ignores=[]), f"MSA@{MSA_COMMIT[:7]}/{path}", "exec",
                       flags=__future__.annotations.compiler_flag, dont_inherit=True)
        exec(code, ns)
    return {name: ns[name] for _, name in _REF_SOURCES}


# ---------------------------------------------------------------- sglang 층 연결 (옮겨 적음)

class _GemmaRMSNorm(nn.Module):
    """sglang layers/layernorm.py GemmaRMSNorm.forward_native"""

    def __init__(self, dim, eps):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        return x.to(orig_dtype)


def _neox_rope(x, positions, inv_freq, rotary_dim):
    """sglang RotaryEmbedding (is_neox_style=True): 앞 rotary_dim 채널을 절반씩 나눠 회전. x [B, T, h, D]."""
    freqs = positions.float()[:, None] * inv_freq[None, :]
    cos, sin = freqs.cos()[:, None, :].to(x.dtype), freqs.sin()[:, None, :].to(x.dtype)
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = torch.chunk(x_rot, 2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, x_pass), dim=-1)


def _topk_select(score, block_size, topk, init_blocks, local_blocks):
    """sglang prefill _topk_index_kernel (MASK_INIT/MASK_LOCAL=False) 의 규칙.

    score [B, G, T, NB] → kv_block_indexes [B, G, T, topk] int32, 오름차순, 모자라면 -1.
    """
    B, G, T, NB = score.shape
    pos = torch.arange(T, device=score.device)
    valid = (pos + block_size) // block_size                         # 질의 위치까지 보이는 블록 수
    j = torch.arange(NB, device=score.device)
    causal = j[None, :] < valid[:, None]
    s = score.float().masked_fill(~causal, -1e30)
    s = torch.where(causal & (j[None, :] < init_blocks), torch.full_like(s, 1e30), s)
    local = j[None, :] >= (valid - local_blocks).clamp(min=0)[:, None]
    s = torch.where(causal & local, torch.full_like(s, 1e29), s)
    k = min(topk, NB)
    idx = s.topk(k, dim=-1).indices                                  # 점수 순 → 앞 min(topk, valid) 개가 유효
    n = valid.clamp(max=topk)
    keep = torch.arange(k, device=score.device)[None, :] < n[:, None]
    idx = torch.where(keep, idx, torch.full_like(idx, NB)).sort(dim=-1).values
    idx = torch.where(idx < NB, idx, torch.full_like(idx, -1)).to(torch.int32)
    if k < topk:
        idx = torch.cat([idx, idx.new_full((B, G, T, topk - k), -1)], dim=-1)
    return idx.contiguous()


class RefMSAAttention(nn.Module):
    """원본 MSA attention 층: 투영/norm/RoPE (sglang) + ref_tile_max_score + top-k 규칙 + sparse_attention_ref."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        sp = cfg.sparse_attention_config
        H, G, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        Gi, Di = sp["sparse_num_index_heads"], sp["sparse_index_dim"]
        assert sp["sparse_block_size"] == 128, "ref_tile_max_score 는 블록 128 고정"
        assert sp["sparse_score_type"] == "max" and Gi == G
        self.q_proj = nn.Linear(cfg.hidden_size, H * D, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, G * D, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, G * D, bias=False)
        self.o_proj = nn.Linear(H * D, cfg.hidden_size, bias=False)
        self.q_norm = _GemmaRMSNorm(D, cfg.rms_norm_eps)
        self.k_norm = _GemmaRMSNorm(D, cfg.rms_norm_eps)
        self.index_q_proj = nn.Linear(cfg.hidden_size, Gi * Di, bias=False)
        self.index_k_proj = nn.Linear(cfg.hidden_size, Di, bias=False)
        self.index_q_norm = _GemmaRMSNorm(Di, cfg.rms_norm_eps)
        self.index_k_norm = _GemmaRMSNorm(Di, cfg.rms_norm_eps)
        r = cfg.rotary_dim
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, r, 2, dtype=torch.float) / r))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.index_score = nn.Identity()
        self.block_select = nn.Identity()

    def forward(self, hidden_states):
        fns = ref_functions()
        cfg, sp, x = self.config, self.config.sparse_attention_config, hidden_states
        B, T, _ = x.shape
        H, G, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        Gi, Di, bs = sp["sparse_num_index_heads"], sp["sparse_index_dim"], sp["sparse_block_size"]
        NB = T // bs
        assert T % bs == 0, "sparse_attention_ref 는 seqlen_k 가 블록 크기의 배수여야 함"
        pos = torch.arange(T, device=x.device)
        rope = lambda t: _neox_rope(t, pos, self.inv_freq, cfg.rotary_dim)
        # main 가지
        q = rope(self.q_norm(self.q_proj(x).view(B, T, H, D)))
        k = rope(self.k_norm(self.k_proj(x).view(B, T, G, D)))
        v = self.v_proj(x).view(B, T, G, D)
        # index 가지 → 블록 점수 (원본 함수, 출력 [head, tile, token])
        iq = rope(self.index_q_norm(self.index_q_proj(x).view(B, T, Gi, Di)))
        ik = rope(self.index_k_norm(self.index_k_proj(x).view(B, T, 1, Di)))
        lens = torch.full((B,), T, dtype=torch.int32)
        score = fns["ref_tile_max_score"](iq.reshape(B * T, Gi, Di), ik.reshape(B * T, 1, Di), lens, lens,
                                          1, NB, causal=True)
        score = score.view(Gi, NB, B, T).permute(2, 0, 3, 1) * Di ** -0.5            # [B, G, T, NB]
        score = self.index_score(score.clamp(min=-1e30))
        q2k = _topk_select(score, bs, sp["sparse_topk_blocks"], sp["sparse_init_block"], sp["sparse_local_block"])
        sel = torch.zeros(B, Gi, T, NB + 1, dtype=torch.bool, device=x.device)
        sel.scatter_(-1, torch.where(q2k >= 0, q2k, NB).long(), True)
        self.block_select(sel[..., :NB])
        # main 가지 sparse attention (원본 함수)
        out, _ = fns["sparse_attention_ref"](q, k, v, q2k, blk_kv=bs, causal=True, softmax_scale=D ** -0.5)
        return self.o_proj(out.to(x.dtype).reshape(B, T, H * D)), q2k


def hf_config():
    """MiniMax-M3 text_config 와 같은 키의 축소 config (ref 와 내 LM 자체 점검이 함께 쓴다)."""
    return SimpleNamespace(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rotary_dim=8, rope_theta=10000.0,
        rms_norm_eps=1e-6, use_gemma_norm=True, use_qk_norm=True, qk_norm_type="per_head",
        tie_word_embeddings=False,
        sparse_attention_config={
            "use_sparse_attention": True, "sparse_index_dim": 16, "sparse_num_index_heads": 2,
            "sparse_topk_blocks": 2, "sparse_block_size": 128, "sparse_score_type": "max",
            "sparse_init_block": 0, "sparse_local_block": 1,
            "sparse_attention_freq": [0, 1], "sparse_disable_index_value": [0, 1]})


def ref():
    """MSA attention 층 1개, 시퀀스 512 (블록 4개, topk 2 = local + 1)."""
    m = RefMSAAttention(hf_config())
    x = torch.randn(BATCH, 512, 64, generator=torch.Generator().manual_seed(0))
    return m, {"hidden_states": x}


CASE = Case(STAGE, "MSA attention 층: index 가지 블록 max 점수 + 그룹별 top-k (local 강제) + 블록 sparse GQA",
            "MiniMax-M3", "github MiniMax-AI/MSA@80434d7 (sparse_attention_ref, ref_tile_max_score) + sglang minimax_m3",
            ref, "miniarch.models.minimax_msa", atol=1e-4)
