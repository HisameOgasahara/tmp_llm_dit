"""B2 · Krea-2 (single-stream DiT) 검증 케이스."""
from __future__ import annotations

import torch

from ..cases import BATCH, Case

STAGE = "B2"
SLUG = "krea2"
TITLE = "Krea-2 (Single-Stream DiT)"
INTRO = """
원본: diffusers `Krea2Transformer2DModel` — `src/diffusers/models/transformers/transformer_krea2.py`
(diffusers 0.41.0 릴리스 = main `af4271c3e18f14d2e8a21a0480d3a89e7bf5cdba` 와 같은 파일, 아래 pip 로 0.41.0 고정).
내 구현: `miniarch/models/krea2.py` (파라미터 이름이 원본과 같아 name_map 없음).

Krea-2 12B 는 **single-stream** DiT 입니다. 텍스트와 이미지 토큰을 `[텍스트 | 이미지]` 한 줄로 이어 붙이고
모든 블록이 같은 가중치로 둘을 함께 처리합니다 (B1 MM-DiT 처럼 스트림별 가중치가 따로 있지 않음).

- 텍스트 조건: 텍스트 인코더 12개 층의 hidden state 묶음 → 층 축 attention(layerwise) → Linear 로 층 축 접기 → 토큰 축 refiner → txt_in
- 시간 조건: modulation 벡터 하나를 모든 블록이 공유하고, 블록마다 학습된 표(scale_shift_table)만 더함
- 블록: zero-centered RMSNorm(1+w), GQA, q/k RMSNorm, 3축 RoPE (t, h, w; 텍스트는 좌표 0), sigmoid 출력 gate, SwiGLU
- 출력: 이미지 토큰만 잘라 final_layer (temb + 표 로 scale/shift)

축소 config: hidden 64 (head 16 × 4, kv head 2), 본 블록 2개, 텍스트 폭 32 · 층 3개 · layerwise 2 + refiner 2 (텍스트도 GQA),
RoPE 축 (4, 6, 6), 이미지 4×5 격자(가로·세로가 바뀌면 드러나도록 직사각형, t 좌표 1 로 세 축 모두 회전),
텍스트 6 토큰 중 0번 샘플은 뒤 2개가 padding (마스크 경로). 0 으로 초기화되는 norm·표는 시드 고정 난수로 채웁니다.
"""
PIP = ["diffusers==0.41.0"]
HEADER = [
    ("krea/Krea-2-Raw", r"^transformer_blocks\.0\.", "transformer"),
    ("krea/Krea-2-Raw", r"^(img_in|time_embed|time_mod_proj|txt_in|final_layer|text_fusion\.(projector|refiner_blocks\.0))\.", "transformer"),
]
CAUSAL = False
SELF_CHECKS = [
    ("""
### attention map 확인

- `attn_probs`: [B, H, S, S], S = 텍스트 6 + 이미지 20. 각 행 합 = 1, padding 텍스트 key 로 가는 확률 = 0 이어야 합니다.
- 이미지 query 가 텍스트 key 에 얼마나 주목하는지(text←image), 이미지끼리(image←image) 비율을 층마다 출력합니다.
""", """
from miniarch.models import krea2 as K
ref, inputs = S.ref()
m = K.verify_pair(ref)["mine"]; m.load_state_dict(ref.state_dict()); m = m.to(dev).eval()
x = {k: v.to(dev) for k, v in inputs.items()}
L = x["encoder_hidden_states"].shape[1]
names = [n for n, _ in m.named_modules() if n.endswith("attn_probs")]
with V.capture(m, names) as st, torch.no_grad():
    m(**x)
pad = ~x["encoder_attention_mask"]
for n in names:
    p = st[n]
    err = (p.sum(-1) - 1).abs().max().item()
    if n.startswith("text_fusion.layerwise"):
        report(f"{n} 행 합", err < 1e-5, err); continue
    leak = p[..., :L].masked_select(pad[:, None, None, :].expand(-1, p.shape[1], p.shape[2], -1)).abs().max().item()
    report(f"{n} 행 합", err < 1e-5, err)
    report(f"{n} padding key 확률 0", leak == 0.0, leak)
    if n.startswith("transformer_blocks"):
        img = p[:, :, L:]
        print(f"   text←image {img[..., :L].sum(-1).mean():.3f}   image←image {img[..., L:].sum(-1).mean():.3f}")
"""),
    ("""
### 텍스트 = 위치 없는 집합 (padding · 순서 불변)

텍스트 토큰의 RoPE 좌표는 모두 0 이고 refiner 에는 RoPE 가 없습니다. 그래서
- padding 토큰을 더 붙여도 (좌표 0 행도 같이 추가) 이미지 출력이 그대로여야 하고
- 유효한 텍스트 토큰 순서를 섞어도 이미지 출력이 그대로여야 합니다.
""", """
from miniarch.models import krea2 as K
ref, inputs = S.ref()
m = K.verify_pair(ref)["mine"]; m.load_state_dict(ref.state_dict()); m = m.to(dev).eval()
x = {k: v.to(dev) for k, v in inputs.items()}
with torch.no_grad():
    base = m(**x)
    e, msk, pos = x["encoder_hidden_states"], x["encoder_attention_mask"], x["position_ids"]
    g = torch.Generator(device=dev).manual_seed(5)
    junk = torch.randn(e.shape[0], 3, *e.shape[2:], device=dev, generator=g) * 10
    x2 = dict(x, encoder_hidden_states=torch.cat([e, junk], 1),
              encoder_attention_mask=torch.cat([msk, torch.zeros(e.shape[0], 3, dtype=torch.bool, device=dev)], 1),
              position_ids=torch.cat([torch.zeros(3, 3, device=dev), pos], 0))
    d_pad = (m(**x2) - base).abs().max().item()
    perm = torch.tensor([3, 1, 0, 2, 5, 4], device=dev)   # 유효 토큰 0~3 끼리, padding 4~5 끼리 섞음 (샘플 1 은 전부 유효)
    x3 = dict(x, encoder_hidden_states=e[:, perm], encoder_attention_mask=msk[:, perm])
    d_perm = (m(**x3) - base).abs().max().item()
report("padding 토큰 추가해도 출력 동일", d_pad < 1e-4, d_pad)
report("텍스트 순서 섞어도 출력 동일", d_perm < 1e-4, d_perm)
"""),
    ("""
### modulation · 시간 조건

- `mod_probe`: 블록마다 (prescale, preshift, pregate, postscale, postshift, postgate). 크기를 출력합니다.
- 시간 modulation 은 모든 블록이 공유하므로 두 블록의 modulation 차이는 **시간과 무관하게** 두 블록 표의 차이와 같아야 합니다.
- timestep 을 바꾸면 출력이 바뀌어야 합니다.
""", """
from miniarch.models import krea2 as K
ref, inputs = S.ref()
m = K.verify_pair(ref)["mine"]; m.load_state_dict(ref.state_dict()); m = m.to(dev).eval()
x = {k: v.to(dev) for k, v in inputs.items()}
names = [n for n, _ in m.named_modules() if n.endswith("mod_probe")]
with V.capture(m, names) as st, torch.no_grad():
    out = m(**x)
labels = ["prescale", "preshift", "pregate", "postscale", "postshift", "postgate"]
for n in names:
    mod = st[n]
    labs = labels if mod.ndim == 4 else ["scale", "shift"]   # 블록 [B,1,6,H] / final [B,2,H]
    print(f"{n:<36} " + "  ".join(f"{l}={mod[..., j, :].abs().mean():.3f}" for j, l in enumerate(labs)))
b = m.transformer_blocks
diff = st["transformer_blocks.0.mod_probe"] - st["transformer_blocks.1.mod_probe"]
err = (diff - (b[0].scale_shift_table - b[1].scale_shift_table)).abs().max().item()
report("블록 간 modulation 차이 = 표 차이 (공유 temb)", err < 1e-5, err)
with torch.no_grad():
    d = (m(**dict(x, timestep=x["timestep"].flip(0))) - out).abs().max().item()
report("timestep 바꾸면 출력이 바뀜", d > 1e-4, d)
"""),
    ("""
### CFG 경로 · ablation

- 파이프라인 CFG: `v = cond + s·(cond − neg)`, neg 는 빈 프롬프트 (여기서는 유효 텍스트 1개). s=0 이면 cond 와 같아야 하고 neg 는 cond 와 달라야 합니다.
- 원본 부품을 하나씩 끈 ablation(RoPE / 출력 gate / 블록 표 / layerwise 융합)이 실제로 출력을 바꾸는지 확인합니다.
""", """
import dataclasses
from miniarch.models import krea2 as K
ref, inputs = S.ref()
m = K.verify_pair(ref)["mine"]; m.load_state_dict(ref.state_dict()); m = m.to(dev).eval()
x = {k: v.to(dev) for k, v in inputs.items()}
neg_mask = torch.zeros_like(x["encoder_attention_mask"]); neg_mask[:, 0] = True
with torch.no_grad():
    cond = m(**x)
    neg = m(**dict(x, encoder_attention_mask=neg_mask))
for s in (0.0, 4.5):
    v = cond + s * (cond - neg)
    print(f"guidance {s}: |v - cond| = {(v - cond).abs().max():.3e}")
report("빈 텍스트(neg) ≠ cond", (cond - neg).abs().max().item() > 1e-4, (cond - neg).abs().max().item())
for sw in ["use_rope", "use_attn_gate", "use_block_table", "use_layerwise_fusion", "use_qk_norm"]:
    m2 = K.Krea2(dataclasses.replace(m.config, **{sw: False})).to(dev).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        d = (m2(**x) - cond).abs().max().item()
    report(f"{sw}=False 면 출력이 바뀜", d > 1e-4, d)
"""),
]


def ref():
    from diffusers import Krea2Transformer2DModel
    m = Krea2Transformer2DModel(
        in_channels=16, num_layers=2, attention_head_dim=16, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=96, timestep_embed_dim=32, text_hidden_dim=32, num_text_layers=3,
        text_num_attention_heads=4, text_num_key_value_heads=2, text_intermediate_size=64,
        num_layerwise_text_blocks=2, num_refiner_text_blocks=2, axes_dims_rope=(4, 6, 6),
        rope_theta=1000.0, norm_eps=1e-5)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # zero-centered norm 과 modulation 표는 0 초기화 → 난수로 채워 경로가 실제로 쓰이게
        for n, p in m.named_parameters():
            if n.endswith("scale_shift_table") or (n.endswith(".weight") and ".norm" in "." + n):
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    L, gh, gw = 6, 4, 5
    gi = torch.Generator().manual_seed(0)
    img_ids = torch.zeros(gh, gw, 3)
    img_ids[..., 0] = 1.0   # 파이프라인은 0 이지만 t 축 회전도 검증하려고 1
    img_ids[..., 1] = torch.arange(gh, dtype=torch.float32)[:, None]
    img_ids[..., 2] = torch.arange(gw, dtype=torch.float32)[None, :]
    inputs = {
        "hidden_states": torch.randn(BATCH, gh * gw, 16, generator=gi),
        "encoder_hidden_states": torch.randn(BATCH, L, 3, 32, generator=gi),
        "timestep": torch.tensor([0.25, 0.8]),
        "position_ids": torch.cat([torch.zeros(L, 3), img_ids.reshape(-1, 3)], 0),
        "encoder_attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool),
    }
    return m, inputs


CASE = Case(STAGE, "Single-stream DiT: 텍스트 층 융합 + 공유 modulation + GQA/qk-norm/3축 RoPE/출력 gate", "Krea-2",
            "diffusers transformer_krea2.py", ref, "miniarch.models.krea2", atol=1e-4)
