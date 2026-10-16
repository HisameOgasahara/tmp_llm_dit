"""A10 · Gemma 4 (sliding/global 하이브리드 · K=V · PLE · KV 공유 · dense+MoE 병렬) 검증 케이스."""
from __future__ import annotations

import torch

from ..cases import BATCH, VOCAB, Case

STAGE = "A10"
SLUG = "gemma4"
TITLE = "Gemma 4 (sliding/global · K=V · PLE · KV 공유 · MoE)"
INTRO = """
원본: transformers `gemma4` 의 `Gemma4TextModel` 과 `gemma4_unified` 의 `Gemma4UnifiedTextModel`
(tag `v5.20.0` = commit `48e4bcd7`; 2026-10 `main`(`683b7219`) 의 두 폴더 파일과 글자 단위로 같음).
내 구현: `miniarch/models/gemma4.py`. 텍스트 backbone 만 비교합니다 (vision/audio 인코더 제외).

체크포인트 (config.json 확인): 31B (dense, K=V), 26B-A4B (K=V + dense MLP·MoE 병렬, expert 128 중 8),
E4B / E2B (PLE + 마지막 18/20 층 KV 공유, E2B 는 공유 층 MLP 2배 폭), 12B (`gemma4_unified`, 2026-05).

부품
- sliding 5 : global 1 하이브리드. sliding 층 head_dim 256 / global 층 512, global 층 KV head 수 따로 (`num_global_key_value_heads`)
- QK-norm + V-norm(가중치 없음), softmax 스케일 1.0
- `attention_k_eq_v`: global 층은 v_proj 없이 k_proj 출력을 V 로 사용 (K 는 k_norm+RoPE, V 는 V-norm)
- RoPE 층 종류별: sliding 은 기본(θ=1e4), global 은 proportional(θ=1e6, 주파수 앞 25% 만 회전)
- Gemma sandwich norm (attention·FFN 앞뒤), 임베딩 × √d, 층 끝 `layer_scalar`, GeGLU(gelu_tanh) MLP, logit softcapping(30)
- MoE (26B-A4B): dense MLP 와 expert 가지를 둘 다 계산해 각각 norm 후 더함. router 는 norm·scale·per_expert_scale
- PLE (E 계열): 층마다 토큰별 작은 벡터를 게이트에 곱해 residual 에 주입
- KV 공유 (E 계열): 마지막 층들은 공유 직전 같은 종류 층의 K/V 를 재사용, 2배 폭 MLP 스위치

`gemma4_unified` 텍스트 모델은 PLE·MoE 항목이 빠졌을 뿐 attention·MLP·RoPE·층 계산이 gemma4 와 같은 코드입니다
(생성된 modeling 파일 diff 로 확인). 별도 스위치 없이 같은 구현으로 자체 점검에서 한 번 더 비교합니다.

축소 config (본 비교): 7층 = [S, F, S, S, F, S, F], window 5, 시퀀스 24 — 원본이 허용하는 기능을 한꺼번에 켬
- head_dim sliding 16 / global 32, KV head sliding 2 / global 1, K=V 켬 → 설정이 뒤바뀌면 shape 부터 어긋남
- global RoPE: 32 × 0.25 / 2 = 주파수 4개만 회전
- 마지막 3층 KV 공유 (F, S, F 가 각각 layer 1, 3 의 K/V 사용) + 공유 층 2배 폭 MLP
- MoE expert 8개 중 3개 + dense MLP, PLE dim 8
- 원본 초기값이 1 인 `layer_scalar`, `per_expert_scale` 과 작은 router 가중치는 0/1 이 아닌 값으로 채움

자체 점검: E 계열(K=V 끔), dense 31B 형, `Gemma4UnifiedTextModel`, `Gemma4ForCausalLM`(softcapping 된 logits) 각각 원본과 비교.
"""
PIP = ["transformers>=5.20.0"]
HEADER = [("google/gemma-4-26B-A4B-it", r"language_model\.layers\.(4|5)\.", ""),
          ("google/gemma-4-E4B-it", r"language_model\.(layers\.(5|41)\.|per_layer|embed_tokens_per_layer)", "")]
CAUSAL = True
SELF_CHECKS = [
    ("""
### 다른 변형 config 도 원본과 같은지

- `e_series`: K=V 끔(global 층에도 v_proj, KV head = sliding 과 같음) + PLE + KV 공유 + 2배 폭 MLP (E4B/E2B 형)
- `dense`: 31B 형 (K=V, PLE·공유·MoE 없음, 기본 5:1 층 배치)
- `unified`: `Gemma4UnifiedTextModel` (12B, 텍스트 경로 동일한지)
- `causal_lm`: `Gemma4ForCausalLM` 으로 감싸 logits(softcapping 포함)까지 비교
""", """
from miniarch.models import gemma4 as G

def eq_variant(v):
    V.set_deterministic(0)
    r, inp = S.ref(v)
    V.perturb_params(r)
    pair = G.verify_pair(r)
    mine = pair.pop("mine")
    rep = V.check_equivalence(r, mine, inp, atol=S.CASE.atol, device=dev, **pair)
    print(f"--- {v}: {type(r).__name__}")
    print(rep.summary())
    report(f"{v} == 원본", rep.ok, max(x.max_abs for x in rep.results))

for v in ["e_series", "dense", "unified", "causal_lm"]:
    eq_variant(v)
"""),
    ("""
### attention map · window · KV 공유 · proportional RoPE

- `attn_probs` 행 합 = 1, 미래 칸과 sliding window 밖 칸 확률은 정확히 0.
- KV 공유 층의 K(`k_probe`)는 넘겨준 층의 K 와 비트 단위로 같아야 합니다.
- global 층 inv_freq 중 0 이 아닌(회전하는) 주파수는 head_dim × 0.25 / 2 개.
""", """
from miniarch.models import gemma4 as G
ref, inp = S.ref()
m = G.verify_pair(ref)["mine"].to(dev).eval()
c = m.config
names = [n for n, _ in m.named_modules() if n.endswith(("attn_probs", "k_probe"))]
x = inp["input_ids"].to(dev)
T_ = x.shape[1]
with V.capture(m, names) as st, torch.no_grad():
    m(x)
i_, j_ = torch.arange(T_, device=dev)[:, None], torch.arange(T_, device=dev)[None, :]
for i in range(c.n_layer):
    t, att = c.layer_types[i], m.layers[i].self_attn
    p = st[f"layers.{i}.self_attn.attn_probs"]
    outside = j_ > i_
    if t == "sliding_attention":
        outside = outside | (i_ - j_ >= c.sliding_window)
    err = (p.sum(-1) - 1).abs().max().item()
    leak = p[..., outside].abs().max().item()
    print(f"layer {i} {t:<18} head_dim={att.d:<3} kv={att.kv} 공유={att.is_shared!s:<5} 넘겨줌={att.store_kv!s:<5} "
          f"행합 오차={err:.1e} 마스크 밖={leak:.1e}")
    report(f"layer {i} 확률 합", err < 1e-5, err)
    report(f"layer {i} 마스크 밖 확률 0", leak == 0, leak)
src = {c.layer_types[i]: i for i in range(c.n_layer) if m.layers[i].self_attn.store_kv}
for i in range(c.n_layer):
    if m.layers[i].self_attn.is_shared:
        s = src[c.layer_types[i]]
        d = (st[f"layers.{i}.self_attn.k_probe"] - st[f"layers.{s}.self_attn.k_probe"]).abs().max().item()
        report(f"layer {i} K == layer {s} K (KV 공유)", d == 0, d)
inv = m.full_attention_inv_freq
n_rot = int((inv > 0).sum())
print(f"global inv_freq: 회전 {n_rot} / {inv.numel()}")
report("global RoPE 앞 25% 만 회전", n_rot == int(0.25 * c.global_head_dim // 2), 0.0)
"""),
    ("""
### ablation 스위치

스위치를 끄면 출력이 실제로 바뀌는지 확인합니다 (같은 가중치를 그대로 로드). softcapping 은 logits 가 ±cap 안에 있는지만 봅니다.
""", """
import dataclasses
from miniarch.models import gemma4 as G
ref, inp = S.ref()
m = G.verify_pair(ref)["mine"].to(dev).eval()
x = inp["input_ids"].to(dev)
with torch.no_grad():
    out = m(x)
for sw in ["qk_norm", "v_norm", "sandwich_norm", "scale_embed", "use_dense_branch", "use_moe_branch", "use_ple"]:
    m2 = G.Gemma4(dataclasses.replace(m.config, **{sw: False})).to(dev).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        d = (m2(x).last_hidden_state - out.last_hidden_state).abs().max().item()
    report(f"{sw}=False 이면 출력이 바뀜", d > 1e-4, d)
cap = m.config.final_logit_softcap
mx = out.logits.abs().max().item()
report(f"logits 가 ±{cap} 안", mx < cap, mx)
"""),
]

TF_REV = "48e4bcd72362ff4887651112dbfa8227211a393a"   # transformers v5.20.0
CKPT = {  # config.json 을 확인한 체크포인트 revision
    "google/gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "google/gemma-4-26B-A4B-it": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    "google/gemma-4-E4B-it": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "google/gemma-4-12B-it": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
}

LAYERS = ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
          "full_attention", "sliding_attention", "full_attention"]


def _base(**over):
    kw = dict(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=48, num_hidden_layers=7,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
        layer_types=list(LAYERS), sliding_window=5, max_position_embeddings=256, use_cache=False,
        hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6, final_logit_softcapping=30.0,
        rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                         "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                                            "rope_theta": 1_000_000.0}},
        tie_word_embeddings=True)
    kw.update(over)
    return kw


VARIANTS = {
    # 본 비교: 원본이 허용하는 기능 전부 (K=V + MoE + PLE + KV 공유 + 2배 폭 MLP)
    "main": dict(attention_k_eq_v=True, num_global_key_value_heads=1,
                 enable_moe_block=True, num_experts=8, top_k_experts=3, moe_intermediate_size=24,
                 hidden_size_per_layer_input=8, vocab_size_per_layer_input=VOCAB,
                 num_kv_shared_layers=3, use_double_wide_mlp=True),
    # E4B/E2B 형: K=V 없음 (global 층도 v_proj, KV head = num_key_value_heads)
    "e_series": dict(attention_k_eq_v=False, hidden_size_per_layer_input=8, vocab_size_per_layer_input=VOCAB,
                     num_kv_shared_layers=3, use_double_wide_mlp=True),
    # 31B 형: 기본 5:1 배치 6층, K=V, PLE·공유·MoE 없음
    "dense": dict(num_hidden_layers=6, layer_types=None, attention_k_eq_v=True, num_global_key_value_heads=1,
                  hidden_size_per_layer_input=0, num_kv_shared_layers=0),
}


def _fill(model):
    """원본 초기값 1 인 layer_scalar / per_expert_scale / router.scale 과 작은 router 가중치를 의미 있는 값으로."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for layer in model.layers:
            layer.layer_scalar.copy_(0.5 + torch.rand(layer.layer_scalar.shape, generator=g))
            r = getattr(layer, "router", None)
            if r is not None:
                r.proj.weight.copy_(torch.randn(r.proj.weight.shape, generator=g) * 0.5)
                r.scale.copy_(1 + 0.3 * torch.randn(r.scale.shape, generator=g))
                r.per_expert_scale.copy_(1 + 0.3 * torch.randn(r.per_expert_scale.shape, generator=g))


def ref(variant: str = "main"):
    """variant: main / e_series / dense (Gemma4TextModel), unified (Gemma4UnifiedTextModel),
    causal_lm (main config 의 Gemma4ForCausalLM, logits 비교)."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM, Gemma4TextModel
    if variant == "unified":
        from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextModel
        cfg = Gemma4UnifiedTextConfig(**_base(num_hidden_layers=6, layer_types=None, attention_k_eq_v=True,
                                              num_global_key_value_heads=1))
        m = Gemma4UnifiedTextModel(cfg)
        inner = m
    elif variant == "causal_lm":
        m = Gemma4ForCausalLM(Gemma4TextConfig(**_base(**VARIANTS["main"])))
        inner = m.model
    else:
        m = Gemma4TextModel(Gemma4TextConfig(**_base(**VARIANTS[variant])))
        inner = m
    m.config._attn_implementation = "eager"
    # Gemma4TextExperts 는 @use_experts_implementation → 기본값 grouped_mm (T4 에선 fallback 커널).
    # 내 구현과 같은 expert 루프 순서로 비교하도록 eager 로 고정 (setter 라 sub-config 에도 전파).
    m.config._experts_implementation = "eager"
    _fill(inner)
    ids = torch.randint(0, VOCAB, (BATCH, 24), generator=torch.Generator().manual_seed(0))
    return m, {"input_ids": ids}


CASE = Case(STAGE, "Gemma 4: sliding/global 하이브리드 + K=V + QK/V-norm + proportional RoPE + PLE + KV 공유 + dense·MoE 병렬",
            "Gemma-4-31B/26B-A4B/E4B/12B", "transformers/models/gemma4, gemma4_unified (v5.20.0)", ref,
            "miniarch.models.gemma4", atol=1e-4)
