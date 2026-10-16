"""A6b · Kimi-K3 (KDA + gated MLA + Attention Residuals(A7b) + latent MoE) 검증 케이스.

원본은 HF 원격 코드(moonshotai/Kimi-K3 의 modeling_kimi_linear.py)이고 KDA 부분이 fla Triton 커널을 부른다.
T4 / fp32 에서 Triton 커널이 안 돌거나 tf32 로 오차가 커질 수 있어서, fla 의 네 이름만 아래 순수 PyTorch 판으로 바꿔 끼운다.
  ShortConvolution   → depthwise causal Conv1d + SiLU            (fla.modules.conv.short_conv 와 같은 파라미터)
  FusedRMSNormGated  → RMSNorm · w · sigmoid(g)                  (fla.modules.fused_norm_gate 커널 식)
  chunk_kda / fused_recurrent_kda → naive_kda
        커널 안 전처리(q/k L2norm, sigmoid(β), 게이트 lower_bound·sigmoid(exp(A_log)(g+dt_bias)))
        + fla.ops.kda.naive.naive_recurrent_kda 의 토큰별 루프 그대로
나머지(MLA, MoE, AttnRes, 층 배치)는 원본 코드를 그대로 실행한다.
"""
from __future__ import annotations

import re
import sys
import types

import torch
import torch.nn.functional as F
from torch import nn

from ..cases import BATCH, VOCAB, Case

STAGE = "A6b"
SLUG = "kimi_k3"
TITLE = "Kimi-K3 (KDA + gated MLA + Attention Residuals + latent MoE)"
INTRO = """
원본: HF 원격 코드 `moonshotai/Kimi-K3` @ `f831ab6` 의 `modeling_kimi_linear.py` (텍스트 모델 `KimiLinearModel`).
내 구현: `miniarch/models/kimi_k3.py`. A7b(Attention Residuals)도 같은 모델 안에 있어서 함께 검증합니다.

- KDA (Kimi Delta Attention): Gated DeltaNet과 같은 delta rule인데 망각 게이트가 key 채널마다 다릅니다. short conv, 저랭크 게이트, lower bound -5
- gated MLA: latent KV 압축, RoPE 없음(NoPE), 출력 sigmoid 게이트
- Attention Residuals: residual을 더하는 대신 이전 블록 출력들을 softmax로 섞어 다음 층 입력을 만듭니다
- latent MoE: sigmoid 라우터 + bias, routed expert는 작은 latent 차원에서 계산, shared expert, SiTU 활성함수

**fla 커널 대체**: 원본 KDA는 fla Triton 커널(`chunk_kda`)을 부릅니다. T4/fp32에서 안 돌 수 있어 원본 builder가
같은 수식의 순수 PyTorch 판(fla의 `naive_recurrent_kda` 루프 + 커널 안 전처리)으로 바꿔 끼웁니다. 나머지는 원본 코드 그대로입니다.

축소 config: 4층 = [KDA, KDA, KDA, MLA], AttnRes 블록 2층(블록 경계 2번), 시퀀스 100 → 내 구현 chunk(64) 2개 + padding.
첫 층 dense, 나머지 MoE(expert 8, top-2, latent 32). 그룹 라우팅 분기(그룹 2개 중 1개)까지 쓰도록 했고(K3 실제는 그룹 1개),
SiTU β를 작게(1.5 / 2.5) 잡아 tanh 포화가 출력에 드러나게 했습니다.
"""
PIP = ["einops"]
HEADER = [("moonshotai/Kimi-K3", r"language_model\.model\.layers\.(0|3)\.", "")]
CAUSAL = True
SELF_CHECKS = [
    ("""
### KDA recurrent == chunk == fla naive

토큰 하나씩 메모리를 갱신하는 정의 그대로의 구현, 학습용 chunk 구현, 원본 builder가 쓰는 fla naive 루프가 같은 값을 내는지 확인합니다.
망각은 채널마다 다르고 [-5, 0) 범위 전체(아주 강한 망각 포함)를 쓰며, 길이 150 · chunk 32로 경계를 여러 번 넘깁니다.
""", """
from miniarch.models import kimi_k3 as K

V.set_deterministic(0)
B_, H_, T_, Dk_, Dv_ = 2, 3, 150, 16, 8
q_ = K.l2norm(torch.randn(B_, H_, T_, Dk_, device=dev)) * Dk_ ** -0.5
k_ = K.l2norm(torch.randn(B_, H_, T_, Dk_, device=dev))
v_ = torch.randn(B_, H_, T_, Dv_, device=dev)
g_ = -5 * torch.sigmoid(3 * torch.randn(B_, H_, T_, Dk_, device=dev))
b_ = torch.rand(B_, H_, T_, device=dev)
rec = K.kda_recurrent(q_, k_, v_, g_, b_)[0]
r = V.same_function_test("KDA recurrent vs chunk(32)", lambda: rec,
                         lambda: K.kda_chunk(q_, k_, v_, g_, b_, chunk_size=32)[0])
report(r.name, r.ok, r.max_abs)
tr = lambda t: t.transpose(1, 2)
r = V.same_function_test("KDA recurrent vs fla naive (원본 builder)", lambda: rec,
                         lambda: tr(S.naive_kda(tr(q_), tr(k_), tr(v_), tr(g_), tr(b_), scale=1.0)[0]))
report(r.name, r.ok, r.max_abs)
"""),
    ("""
### 모델 전체: chunk 모드 == recurrent 모드

같은 가중치로 `delta_mode="chunk"`와 `"recurrent"`의 최종 출력이 같아야 합니다.
""", """
import dataclasses
from miniarch.models import kimi_k3 as K

V.set_deterministic(0)
ref, inp = S.ref()
m = K.verify_pair(ref)["mine"].to(dev).eval()
m2 = K.KimiK3(dataclasses.replace(m.config, delta_mode="recurrent")).to(dev).eval()
m2.load_state_dict(m.state_dict())
x = inp["input_ids"].to(dev)
with torch.no_grad():
    err = (m(x).last_hidden_state - m2(x).last_hidden_state).abs().max().item()
report("모델 chunk vs recurrent", err < 1e-4, err)
"""),
    ("""
### hook 확인: AttnRes 깊이 방향 확률 · MLA attention map · 라우팅 · KDA 망각

- `*_res_probe`: [B, T, 블록 수+1]. 각 층 입력이 이전 블록들(첫 칸 = embedding)과 현재 부분합을 어떤 비율로 섞는지. 합 = 1.
- `attn_probs`: MLA 층 [B, H, T, T], 행 합 = 1. `route_probe`: 토큰마다 고른 expert. `decay_probe`: 채널별 log 망각.
""", """
names = [n for n, _ in m.named_modules() if n.endswith(("res_probe", "attn_probs", "route_probe", "decay_probe"))]
with V.capture(m, names) as st, torch.no_grad():
    m(x)
for n in names:
    if n not in st:   # 0층 attention 입력은 이전 블록이 없어 섞지 않음
        continue
    t = st[n]
    if n.endswith(("res_probe", "attn_probs")):
        err = (t.sum(-1) - 1).abs().max().item()
        extra = f"평균 확률={t.float().mean((0, 1)).cpu().numpy().round(3)}" if n.endswith("res_probe") else ""
        print(f"{n:<42} {tuple(t.shape)}  합 오차={err:.1e} {extra}")
        report(f"{n} 확률 합", err < 1e-5, err)
    elif n.endswith("route_probe"):
        print(f"{n:<42} expert 사용 횟수={torch.bincount(t.flatten(), minlength=m.config.n_experts).tolist()}")
    else:
        print(f"{n:<42} 한 step 보존율 exp(g) 평균={t.exp().mean():.3f}  최소={t.exp().min():.4f}")
"""),
    ("""
### ablation 스위치 동작

스위치를 하나씩 끈 모델이 정상적으로 forward 되는지 확인합니다 (값 비교가 아니라 shape · 유한값 확인).
""", """
for sw in ["use_attn_res", "use_kda", "use_moe", "use_latent_moe", "use_output_gate"]:
    cfg = dataclasses.replace(m.config, **{sw: False})
    with torch.no_grad():
        out = K.KimiK3(cfg).to(dev).eval()(x).logits
    ok = bool(torch.isfinite(out).all()) and tuple(out.shape) == (x.shape[0], x.shape[1], cfg.vocab_size)
    report(f"ablation {sw}=False forward", ok, 0.0)
"""),
]

REPO, REVISION = "moonshotai/Kimi-K3", "f831ab66814297da540d832a5235f8e904f29d06"
_PKG = "_kimi_k3_ref"


# ---------------------------------------------------------------- fla 대체 (순수 PyTorch)

class ShortConvolution(nn.Conv1d):
    """fla.modules.ShortConvolution 대체: depthwise causal conv (weight [D, 1, W]) + SiLU."""

    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", backend="triton",
                 device=None, dtype=None, **kwargs):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias,
                         padding=kernel_size - 1, device=device, dtype=dtype)
        self.hidden_size, self.activation = hidden_size, activation

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False, cu_seqlens=None, **kwargs):
        assert mask is None and cache is None and cu_seqlens is None, "검증용 대체: cache/varlen 미지원"
        T = x.shape[1]
        y = super().forward(x.transpose(1, 2))[..., :T].transpose(1, 2)
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        if residual is not None:
            y = y + residual
        return y, None


class FusedRMSNormGated(nn.Module):
    """fla.modules.FusedRMSNormGated 대체: fp32 로 x·rstd·w 후 sigmoid(g) (swish 면 g·sigmoid(g)) 곱."""

    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", device=None, dtype=None):
        super().__init__()
        self.hidden_size, self.eps, self.activation = hidden_size, eps, activation
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        assert residual is None
        xf = x.float()
        y = xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        gf = g.float()
        y = y * gf * torch.sigmoid(gf) if self.activation in ("swish", "silu") else y * torch.sigmoid(gf)
        return y.to(x.dtype)


def _l2norm(x, eps=1e-6):
    """fla.modules.l2norm.l2norm_fwd: x / sqrt(Σx² + eps)."""
    xf = x.float()
    return (xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) + eps)).to(x.dtype)


def naive_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
              allow_neg_eigval=False, safe_gate=False, lower_bound=None, cu_seqlens=None, **kwargs):
    """fla chunk_kda / fused_recurrent_kda 대체. 입력 [B, T, H, K] 레이아웃 (fla 와 같음).

    1) 커널 안 전처리 (fla.ops.kda.chunk.ChunkKDAFunction.forward, gate.kda_gate_chunk_cumsum 과 같은 식)
    2) fla.ops.kda.naive.naive_recurrent_kda 의 토큰별 루프 그대로
    """
    assert cu_seqlens is None, "검증용 대체: varlen 미지원"
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q), _l2norm(k)
    if use_beta_sigmoid_in_kernel:
        beta = beta.float().sigmoid() * (2.0 if allow_neg_eigval else 1.0)
    if use_gate_in_kernel:
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")
        H = g.shape[-2]
        g = g.float()
        if dt_bias is not None:
            g = g + dt_bias.float().view(H, -1)
        A = A_log.float().view(H, 1).exp()
        g = lower_bound * torch.sigmoid(A * g) if lower_bound is not None else -A * F.softplus(g)

    # ---- naive_recurrent_kda (fla v0.5.2) ----
    dtype = v.dtype
    B, T, H, K, HV, Vd = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = (t.to(torch.float) for t in (q, k, v, g, beta))
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    S = k.new_zeros(B, HV, K, Vd)
    if initial_state is not None:
        S = S + initial_state
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum("bhk,bhv->bhkv", b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)
    return o.to(dtype), (S if output_final_state else None)


def _causal_mask(config=None, input_embeds=None, inputs_embeds=None, attention_mask=None, **kwargs):
    """transformers.masking_utils.create_causal_mask 대체 (eager, padding 없음): 4D 덧셈 mask [B, 1, T, T].

    masking_utils 의 인자 이름이 버전마다 달라서(input_embeds/inputs_embeds, cache_position) 같은 의미로 직접 만든다.
    """
    assert attention_mask is None, "검증용 대체: padding mask 미지원"
    e = input_embeds if input_embeds is not None else inputs_embeds
    B, T = e.shape[:2]
    m = torch.full((T, T), torch.finfo(e.dtype).min, device=e.device, dtype=e.dtype).triu(1)
    return m[None, None].expand(B, 1, T, T)


def _fallbacks():
    """transformers 버전에 따라 없어진 이름의 대체값 (import 가 성공하면 원래 것으로 덮어써진다)."""
    import typing

    import transformers
    from transformers.utils import logging as hf_logging
    ident = lambda f: f  # noqa: E731
    return {
        "PretrainedConfig": getattr(transformers, "PretrainedConfig", None) or getattr(transformers, "PreTrainedConfig"),
        "check_model_inputs": ident, "OutputRecorder": lambda *a, **k: None, "ALL_LAYERNORM_LAYERS": [],
        "TransformersKwargs": dict, "FlashAttentionKwargs": dict, "Unpack": getattr(typing, "Unpack", typing.Optional),
        "auto_docstring": ident, "can_return_tuple": ident, "logging": hf_logging, "create_causal_mask": _causal_mask,
        # fla 대체
        "ShortConvolution": ShortConvolution, "FusedRMSNormGated": FusedRMSNormGated,
        "chunk_kda": naive_kda, "fused_recurrent_kda": naive_kda, "tensor_cache": ident,
        "prepare_lens_from_mask": lambda m: m.sum(-1, dtype=torch.int32),
        "prepare_cu_seqlens_from_mask": lambda m: F.pad(m.sum(-1, dtype=torch.int32).cumsum(0), (1, 0)).int(),
    }


def _patch_source(name: str, src: str) -> str:
    if name == "modeling_kimi_linear":
        # fla import 블록 제거 (위 대체값을 쓴다)
        src, n = re.subn(r"try:\n    from fla\.modules import.*?except ImportError:\n    raise ImportError\([^\n]*\)\n",
                         "", src, flags=re.S)
        assert n == 1, "fla import 블록을 못 찾음 (원본 revision 확인)"
        # 입력 기본값 처리 decorator: 버전마다 시그니처가 달라 제거 (계산과 무관)
        src, n = re.subn(r"\n    @check_model_inputs\n", "\n", src)
        assert n == 1, "@check_model_inputs 를 못 찾음"
    # 한 줄짜리 transformers import 는 실패해도 넘어가게 (_fallbacks 값을 쓴다)
    return re.sub(r"^(from transformers[\w.]* import [^\n()]+)$", r"try:\n    \1\nexcept ImportError:\n    pass",
                  src, flags=re.M)


def _load_reference():
    """원격 코드 두 파일을 pinned revision 으로 받아 가짜 패키지 `_kimi_k3_ref` 로 실행 (trust_remote_code 불필요)."""
    key = f"{_PKG}.modeling_kimi_linear"
    if key in sys.modules:
        return sys.modules[key]
    from huggingface_hub import hf_hub_download
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = []
    sys.modules[_PKG] = pkg
    for name in ("configuration_kimi_k3", "modeling_kimi_linear"):
        path = hf_hub_download(REPO, f"{name}.py", revision=REVISION)
        src = _patch_source(name, open(path, encoding="utf-8").read())
        mod = types.ModuleType(f"{_PKG}.{name}")
        mod.__file__, mod.__package__ = path, _PKG
        mod.__dict__.update(_fallbacks())
        sys.modules[mod.__name__] = mod
        try:
            exec(compile(src, path, "exec"), mod.__dict__)
        except Exception:
            sys.modules.pop(mod.__name__, None)
            raise
        setattr(pkg, name, mod)
    mod = sys.modules[key]
    mod.create_causal_mask = _causal_mask
    return mod


# ---------------------------------------------------------------- 원본 builder

def ref():
    """KDA 3층 + MLA 1층, AttnRes 블록 2층, 시퀀스 100 (내 구현 chunk 64 → chunk 2개 + padding)."""
    M = _load_reference()
    cfg = M.KimiLinearConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, hidden_act="situ",
        activation_situ_beta=1.5, activation_situ_linear_beta=2.5, rms_norm_eps=1e-5,
        q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        mla_use_nope=True, mla_use_output_gate=True,
        linear_attn_config={"kda_layers": [1, 2, 3], "full_attn_layers": [4], "num_heads": 4, "head_dim": 16,
                            "short_conv_kernel_size": 4, "gate_lower_bound": -5.0, "use_full_rank_gate": True},
        num_experts=8, num_experts_per_token=2, moe_intermediate_size=32, num_shared_experts=1,
        routed_expert_hidden_size=32, latent_moe_use_norm=True, moe_router_activation_func="sigmoid",
        moe_renormalize=True, routed_scaling_factor=2.5, num_expert_group=2, topk_group=1,
        first_k_dense_replace=1, moe_layer_freq=1, attn_res_block_size=2,
        max_position_embeddings=256, use_cache=False, tie_word_embeddings=False,
        pad_token_id=None, bos_token_id=None, eos_token_id=None)
    m = M.KimiLinearModel(cfg)
    m.config._attn_implementation = "eager"      # 원본 __init__ 이 flash_attention_2 로 고정 → T4 용 eager
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # torch.empty 로 만들어지는 파라미터를 0 이 아닌 값으로 (망각 세기·라우팅 bias 가 다양하게)
        for layer in m.layers:
            a = layer.self_attn
            if hasattr(a, "dt_bias"):
                a.dt_bias.copy_(torch.randn(a.dt_bias.shape, generator=g) * 0.3 - 0.3)
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is not None:
                moe.gate.e_score_correction_bias.copy_(torch.randn(cfg.num_experts, generator=g) * 0.1)
    m.eval()
    ids = torch.randint(0, VOCAB, (BATCH, 100), generator=torch.Generator().manual_seed(0))
    return m, {"input_ids": ids}


CASE = Case(STAGE, "Kimi-K3 전체: KDA(채널별 망각 delta rule) + gated MLA(NoPE) + Attention Residuals + latent MoE",
            "Kimi-K3", "HF moonshotai/Kimi-K3 modeling_kimi_linear.py", ref, "miniarch.models.kimi_k3", atol=1e-4)
