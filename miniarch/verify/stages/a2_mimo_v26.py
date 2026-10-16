"""A2 · MiMo-V2.6 (SWA + full 하이브리드, attention sink, partial RoPE, sigmoid 그룹 MoE) 검증 케이스."""
from __future__ import annotations

import inspect
import sys

import torch

from ..cases import BATCH, VOCAB, Case

STAGE = "A2"
SLUG = "mimo_v26"
TITLE = "MiMo-V2.6 (SWA + full · sink · partial RoPE · MoE)"
INTRO = """
원본: HF 원격 코드 `XiaomiMiMo/MiMo-V2.6-Flash-RL` 의 `modeling_mimo_v2.py` (`MiMoV2Model`, commit `dcd3fd8` 고정).
Flash / Pro 저장소의 모델 코드는 같은 파일입니다. 내 구현: `miniarch/models/mimo_v26.py`.
텍스트 backbone만 비교합니다 (vision/audio 인코더, MTP 층, DFlash draft 제외).

부품
- sliding window attention + full attention 하이브리드 (Flash 1:5, Pro 1:7), 층 종류마다 head 수·KV head·head_dim·RoPE theta가 다름
- q/k head_dim ≠ v head_dim (192 / 128), value에 상수 스케일(0.707) 곱함
- partial RoPE: head 앞쪽 33.4% 채널만 회전
- head별 attention sink (gpt-oss 방식, softmax 분모에만 추가). 원본 설정은 sliding 층에만 있음. QK-norm은 없음
- MoE: sigmoid 점수 + 보정 bias, 그룹 top-k → expert top-k, 첫 층만 dense

축소 config: 4층 = [full+dense, sliding+MoE, sliding+MoE, full+MoE], window 8, 시퀀스 40.
- full / sliding 층의 head_dim(24 / 30), KV head(1 / 2), theta(1e6 / 1e4)를 일부러 다르게 해서 설정이 뒤바뀌면 바로 드러나게 했습니다.
- rope 채널 = int(24×0.334)=8, int(30×0.334)=10 → 실제(192 → 64)처럼 앞쪽 일부만 회전합니다.
- expert 8개를 4그룹으로 나누고 2그룹만 고르게 해서 그룹 제한 경로를 씁니다 (실제 설정은 그룹 1개).
  routed_scaling_factor=2.5, eps 1e-5 도 기본값과 다르게 둬서 누락을 잡습니다.
- 원본 코드가 `torch.empty`로 만드는 sink·router 가중치·보정 bias는 0이 아닌 값으로 채웁니다.
- 실제 체크포인트와 같은 fused `qkv_proj`로 비교하고, 분리형(q/k/v_proj)은 자체 점검에서 한 번 더 비교합니다.
"""
PIP = []
HEADER = [("XiaomiMiMo/MiMo-V2.6-Flash-RL", r"^model\.layers\.(0|1)\.", "")]
CAUSAL = True
SELF_CHECKS = [
    ("""
### attention map · sink · window 확인

- `attn_probs`: [B, H, T, T] (sink 칸 제외). 행 합 + sink 확률 = 1 이어야 합니다 (sink 없는 full 층은 행 합 = 1).
- 미래 칸과 sliding window 밖 칸의 확률은 정확히 0 이어야 합니다.
""", """
from miniarch.models import mimo_v26 as M
ref, _ = S.ref()
m = M.verify_pair(ref)["mine"].to(dev).eval()
names = [n for n, _ in m.named_modules() if n.endswith(("attn_probs", "sink_probe"))]
T_ = 40
x = torch.randint(0, V.cases.VOCAB, (2, T_), device=dev)
with V.capture(m, names) as st, torch.no_grad():
    m(x)
i_, j_ = torch.arange(T_, device=dev)[:, None], torch.arange(T_, device=dev)[None, :]
for i in range(m.config.n_layer):
    kind = m.config.layer_types[i]
    p = st[f"layers.{i}.self_attn.attn_probs"]
    s = st.get(f"layers.{i}.self_attn.sink_probe", torch.zeros_like(p[..., 0]))
    err = (p.sum(-1) + s - 1).abs().max().item()
    outside = j_ > i_
    if kind == "sliding_attention":
        outside = outside | (i_ - j_ >= m.config.window)
    leak = p[..., outside].abs().max().item()
    print(f"layer {i} {kind:<18} probs={tuple(p.shape)}  행합+sink 오차={err:.1e}  마스크 밖 최대={leak:.1e}  평균 sink={s.mean():.3f}")
    report(f"layer {i} 확률 합(+sink)", err < 1e-5, err)
    report(f"layer {i} 마스크 밖 확률 0", leak == 0, leak)
"""),
    ("""
### MoE 라우팅: 그룹 제한

토큰마다 고른 expert가 `topk_group`개 이하의 그룹 안에만 있어야 합니다. expert별 부하도 함께 봅니다.
""", """
from miniarch.models import mimo_v26 as M
ref, _ = S.ref()
m = M.verify_pair(ref)["mine"].to(dev).eval()
names = [n for n, _ in m.named_modules() if n.endswith("route_probe")]
x = torch.randint(0, V.cases.VOCAB, (2, 40), device=dev)
with V.capture(m, names) as st, torch.no_grad():
    m(x)
c = m.config
per = c.n_experts // c.n_group
for n in names:
    idx = st[n]
    ng = max(len(set(row)) for row in (idx // per).tolist())
    load = torch.bincount(idx.flatten(), minlength=c.n_experts).tolist()
    print(f"{n:<28} 토큰당 그룹 수 최대={ng}  expert 부하={load}")
    report(f"{n} 그룹 수 <= {c.topk_group}", ng <= c.topk_group, float(ng))
"""),
    ("""
### 분리형 q/k/v_proj 배치도 원본과 같은지

원본은 `attention_projection_layout`이 `"split"`이면 q/k/v를 따로 투영합니다 (계산은 같고 가중치 이름만 다름).
""", """
from miniarch.models import mimo_v26 as M
V.set_deterministic(0)
ref_s, inp_s = S.ref(layout="split")
V.perturb_params(ref_s)
pair = M.verify_pair(ref_s)
mine_s = pair.pop("mine")
rep = V.check_equivalence(ref_s, mine_s, inp_s, atol=S.CASE.atol, device=dev, **pair)
print(rep.summary())
report("split q/k/v 배치 == 원본", rep.ok, max(r.max_abs for r in rep.results))
"""),
]

REPO = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
REVISION = "dcd3fd83fb181622fbc606fbb1709901982a62c2"


def _compat_mask(fn):
    """원격 코드는 transformers 5.3 기준 인자(input_embeds, cache_position)로 mask 함수를 부른다.
    최신 버전 이름(inputs_embeds)으로 바꾸고 없는 인자는 버린다. mask 내용은 그대로."""
    if getattr(fn, "_mimo_compat", False):
        return fn
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()) or \
            ("input_embeds" in params and "cache_position" in params):
        return fn

    def wrap(**kw):
        if "input_embeds" in kw and "input_embeds" not in params:
            kw["inputs_embeds"] = kw.pop("input_embeds")
        return fn(**{k: v for k, v in kw.items() if k in params})
    wrap._mimo_compat = True
    return wrap


def _remote_classes():
    """고정 commit 의 configuration/modeling .py 만 받아서 import (가중치 다운로드 없음)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg_cls = get_class_from_dynamic_module("configuration_mimo_v2.MiMoV2Config", REPO, revision=REVISION)
    model_cls = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2Model", REPO, revision=REVISION)
    mod = sys.modules[model_cls.__module__]
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        setattr(mod, name, _compat_mask(getattr(mod, name)))
    return cfg_cls, model_cls


def ref(layout: str = "fused_qkv"):
    """4층 = [full+dense, sliding+MoE, sliding+MoE, full+MoE], window 8, 시퀀스 40."""
    cfg_cls, model_cls = _remote_classes()
    cfg = cfg_cls(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=1, head_dim=24, v_head_dim=16,
        swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=30, swa_v_head_dim=12,
        rope_theta=1e6, swa_rope_theta=1e4, partial_rotary_factor=0.334,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 0.334},
        sliding_window=8, add_full_attention_sink_bias=False, add_swa_attention_sink_bias=True,
        hybrid_layer_pattern=[0, 1, 1, 0], attention_value_scale=0.707,
        n_routed_experts=8, moe_intermediate_size=32, num_experts_per_tok=2, routed_scaling_factor=2.5,
        scoring_func="sigmoid", topk_method="noaux_tc", n_group=4, topk_group=2, norm_topk_prob=True,
        moe_layer_freq=[0, 1, 1, 1], attention_projection_layout=layout, layernorm_epsilon=1e-5,
        max_position_embeddings=256, use_cache=False, tie_word_embeddings=False)
    m = model_cls(cfg)
    m.config._attn_implementation = "eager"   # sink 는 eager 경로에서만 정확
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # 원본이 torch.empty 로 두는 값들을 0 이 아닌 값으로
        for layer in m.layers:
            a = layer.self_attn
            if a.attention_sink_bias is not None:
                a.attention_sink_bias.copy_(torch.randn(a.attention_sink_bias.shape, generator=g))
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None:
                gate.weight.copy_(torch.randn(gate.weight.shape, generator=g) * 0.1)
                gate.e_score_correction_bias.copy_(torch.randn(gate.e_score_correction_bias.shape, generator=g) * 0.1)
    ids = torch.randint(0, VOCAB, (BATCH, 40), generator=torch.Generator().manual_seed(0))
    return m, {"input_ids": ids}


CASE = Case(STAGE, "MiMo-V2.6: SWA+full 하이브리드 + attention sink + partial RoPE + value scale + sigmoid 그룹 MoE",
            "MiMo-V2.6-Flash/Pro", "HF XiaomiMiMo/MiMo-V2.6-Flash-RL/modeling_mimo_v2.py", ref,
            "miniarch.models.mimo_v26", atol=1e-4)
