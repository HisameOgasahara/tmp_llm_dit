"""B1 · Qwen-Image MM-DiT (dual stream · joint attention · adaLN · QK-norm · MSRoPE) 검증 케이스.

원본 = diffusers QwenImageTransformer2DModel (PyPI 0.41.0 = 커밋 086bf95 의 transformer_qwenimage.py).
같은 클래스가 설정만 바꿔 생성(Qwen-Image/2512) · 편집(Edit-2511, zero_cond_t) · Layered 를 모두 맡으므로
ref(variant) 로 셋을 만들 수 있게 했다. CASE 는 경로가 가장 많은 편집(edit), 나머지 둘은 자체 점검에서 비교.
"""
from __future__ import annotations

import torch

from ..cases import BATCH, Case

STAGE = "B1"
SLUG = "qwen_image"
TITLE = "Qwen-Image MM-DiT (dual stream)"
INTRO = """
원본: diffusers `QwenImageTransformer2DModel` (`src/diffusers/models/transformers/transformer_qwenimage.py`,
PyPI `diffusers==0.41.0` = 커밋 `086bf95` 고정. main `af4271c` 는 Neuron 전용 RoPE 분기만 지운 것으로 CPU/CUDA 계산은 같음).
내 구현: `miniarch/models/qwen_image.py`. 가중치 이름은 실제 체크포인트(`Qwen/Qwen-Image` `transformer/`)와 같습니다.
VAE 와 텍스트 인코더(Qwen2.5-VL)는 빼고, transformer 만 비교합니다 (텍스트 입력 = 인코더 폭의 랜덤 hidden).

부품
- dual stream 블록: 이미지·텍스트가 adaLN·q/k/v·출력·MLP 가중치를 따로 갖고, attention 만 [텍스트, 이미지] 를 이어 한 번에 (양방향)
- adaLN: timestep → [shift, scale, gate] × (attn, MLP). 최종층은 [scale, shift] 순서
- QK-norm: head 별 RMSNorm (이미지 `norm_q/k`, 텍스트 `norm_added_q/k`)
- MSRoPE: head_dim 을 [frame | 높이 | 너비] 로 나눈 3축 RoPE. 이미지 위치는 가운데 0 기준, 텍스트는 이미지 대각선 뒤
- 텍스트 padding mask (key 쪽만 가림), 편집용 `zero_cond_t` (참조 이미지 토큰은 t=0 modulation)

축소 config: 2층, d=64 (head 4 × 16), RoPE 축 (4, 6, 6), 텍스트 폭 48, latent 16채널(=4ch × 2×2 patch), 텍스트 7토큰.
- 두 번째 샘플의 텍스트 mask 는 `[1,1,0,1,1,0,0]` (중간 구멍 + 끝 padding)
- edit(CASE): 이미지 2장 (1,3,4)+(1,4,2) → 홀수 높이의 가운데 정렬, frame 위치 0/1, 토큰별 modulation 선택 경로
- t2i / layered 는 자체 점검에서 같은 방식으로 비교 (layered: 4장, 마지막 조건 이미지 frame -1, 추가 t 조건 Embedding)
- 층별로 attention 출력, 두 MLP, 블록 출력(텍스트 쪽), 최종 norm 과 최종 출력을 비교합니다.
"""
PIP = ["diffusers==0.41.0"]
HEADER = [("Qwen/Qwen-Image", r"^transformer_blocks\.0\.", "transformer"),
          ("Qwen/Qwen-Image", r"^(img_in|txt_in|txt_norm|time_text_embed|norm_out|proj_out)\.", "transformer")]
CAUSAL = False
SELF_CHECKS = [
    ("""
### 다른 변형도 원본과 같은지 (t2i · layered)

- t2i: Qwen-Image / 2512 처럼 이미지 1장, `zero_cond_t=False`
- layered: Qwen-Image-Layered 처럼 `use_layer3d_rope=True` + `use_additional_t_cond=True`, 이미지 4장 (마지막 = 조건 이미지)
""", """
import dataclasses
for v in ["t2i", "layered"]:
    row = V.cases.run_case(dataclasses.replace(S.CASE, ref=lambda v=v: S.ref(v)))
    print(f"[{v}]", row["status"]); print(row["detail"])
    report(f"원본 동치 ({v})", row["status"].startswith("✅"))
"""),
    ("""
### joint attention map

`attn_probs` [B, H, T+N, T+N] 는 행·열 모두 [텍스트 T | 이미지 N] 순서입니다.
- 행 합 = 1, padding 텍스트 key 열의 확률 = 0
- 이미지 질의가 이미지/텍스트에, 텍스트 질의가 이미지에 주는 평균 확률 질량을 봅니다.
""", """
from miniarch.models import qwen_image as Q
ref, inp = S.ref()
m = Q.verify_pair(ref)["mine"]; V.copy_weights(ref, m); m = m.to(dev).eval()
x = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inp.items()}
T_ = x["encoder_hidden_states"].shape[1]
mask = x["encoder_hidden_states_mask"].bool()
names = [n for n, _ in m.named_modules() if n.endswith("attn_probs")]
with V.capture(m, names) as st, torch.no_grad():
    m(**x)
for n in names:
    p = st[n]
    err = (p.sum(-1) - 1).abs().max().item()
    pad = max(p[b, :, :, :T_][..., ~mask[b]].abs().max().item() if (~mask[b]).any() else 0.0 for b in range(p.shape[0]))
    ii, it = p[:, :, T_:, T_:].sum(-1).mean().item(), p[:, :, T_:, :T_].sum(-1).mean().item()
    ti = p[:, :, :T_, T_:].sum(-1).mean().item()
    print(f"{n:<36} {tuple(p.shape)}  이미지→이미지 {ii:.3f}  이미지→텍스트 {it:.3f}  텍스트→이미지 {ti:.3f}")
    report(f"{n} 행 합 = 1", err < 1e-5, err)
    report(f"{n} padding key 확률 = 0", pad == 0.0, pad)
"""),
    ("""
### 텍스트 padding · CFG 빈 텍스트 경로

- padding 위치의 텍스트 내용을 바꾸거나, 끝에 padding 토큰을 더 붙여도 이미지 출력이 그대로여야 합니다
  (텍스트 RoPE 위치는 앞에서부터라 유효 토큰 위치가 변하지 않음).
- CFG 의 무조건 경로처럼 텍스트를 전부 가리면 텍스트 내용과 무관해야 하고, 원본도 같은 값을 내야 합니다.
""", """
from miniarch.models import qwen_image as Q
ref, inp = S.ref()
m = Q.verify_pair(ref)["mine"]; V.copy_weights(ref, m); m = m.to(dev).eval(); ref = ref.to(dev).eval()
x = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inp.items()}
e, mk = x["encoder_hidden_states"], x["encoder_hidden_states_mask"]
g = torch.Generator(device=dev).manual_seed(5)
with torch.no_grad():
    y0 = m(**x)
    e2 = e.clone(); pad = ~mk.bool()
    e2[pad] = torch.randn(e2[pad].shape, generator=g, device=dev) * 3
    d1 = (m(**dict(x, encoder_hidden_states=e2)) - y0).abs().max().item()
    e3 = torch.cat([e, torch.randn(e.shape[0], 3, e.shape[2], generator=g, device=dev)], 1)
    mk3 = torch.cat([mk, torch.zeros(mk.shape[0], 3, dtype=mk.dtype, device=dev)], 1)
    d2 = (m(**dict(x, encoder_hidden_states=e3, encoder_hidden_states_mask=mk3)) - y0).abs().max().item()
    xn = dict(x, encoder_hidden_states_mask=torch.zeros_like(mk))
    yn = m(**xn)
    d3 = (m(**dict(xn, encoder_hidden_states=torch.randn(e.shape, generator=g, device=dev))) - yn).abs().max().item()
    d4 = (ref(**xn).sample - yn).abs().max().item()
    d5 = (yn - y0).abs().max().item()
report("padding 내용 바꿔도 출력 같음", d1 < 1e-5, d1)
report("끝 padding 추가해도 출력 같음", d2 < 1e-4, d2)
report("텍스트 전부 mask → 텍스트 내용 무관", d3 < 1e-5 and bool(torch.isfinite(yn).all()), d3)
report("텍스트 전부 mask: 원본 == 내 구현", d4 < 1e-4, d4)
report("조건/무조건 출력이 다름", d5 > 1e-3, d5)
"""),
    ("""
### timestep · adaLN modulation probe

- `img_mod_probe` / `txt_mod_probe`: [shift1, scale1, gate1, shift2, scale2, gate2] (1 = attention, 2 = MLP).
- 편집 경로(`zero_cond_t`): 이미지 modulation 은 [t, 0] 두 벌이고, 뒤 절반이 timestep=0 으로 돌린 결과의 앞 절반과 같아야 합니다.
- timestep 을 바꾸면 출력이 바뀌어야 합니다.
""", """
from miniarch.models import qwen_image as Q
ref, inp = S.ref()
m = Q.verify_pair(ref)["mine"]; V.copy_weights(ref, m); m = m.to(dev).eval()
x = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inp.items()}
B_ = x["timestep"].shape[0]
names = [n for n, _ in m.named_modules() if n.endswith(("img_mod_probe", "txt_mod_probe"))]
x0 = dict(x, timestep=torch.zeros_like(x["timestep"]))
with V.capture(m, names) as st, torch.no_grad():
    y = m(**x)
with V.capture(m, names) as st0, torch.no_grad():
    y0 = m(**x0)
for n in names:
    a = st[n]
    sh1, sc1, g1, sh2, sc2, g2 = a[:B_].chunk(6, dim=-1)
    print(f"{n:<34} {tuple(a.shape)}  |scale1| {sc1.abs().mean():.3f}  |gate1| {g1.abs().mean():.3f}  |gate2| {g2.abs().mean():.3f}")
    if n.endswith("img_mod_probe") and m.config.zero_cond_t:
        err = (a[B_:] - st0[n][:B_]).abs().max().item()
        report(f"{n} 참조 이미지 = t=0 modulation", a.shape[0] == 2 * B_ and err < 1e-5, err)
d = (y - y0).abs().max().item()
report("timestep 바꾸면 출력이 바뀜", d > 1e-3, d)
"""),
    ("""
### ablation 스위치

부품별 비교 실험에 쓸 스위치를 하나씩 끄고, forward 가 유한값이고 출력이 실제로 바뀌는지 확인합니다.
""", """
import dataclasses
from miniarch.models import qwen_image as Q
ref, inp = S.ref()
m = Q.verify_pair(ref)["mine"]; V.copy_weights(ref, m); m = m.to(dev).eval()
x = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inp.items()}
with torch.no_grad():
    y = m(**x)
for sw in ["use_qk_norm", "use_rope", "center_rope", "use_text_mask", "joint_attention", "use_txt_norm"]:
    m2 = Q.QwenImage(dataclasses.replace(m.config, **{sw: False})).to(dev).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        out = m2(**x)
    d = (out - y).abs().max().item()
    report(f"{sw}=False forward (출력 변화)", bool(torch.isfinite(out).all()) and d > 1e-5, d)
"""),
]

DIFFUSERS_VERSION = "0.41.0"
DIFFUSERS_COMMIT = "086bf9578c0f4acbc66e48cd1e7cc26befd9e10f"
TXT_LEN, TXT_DIM = 7, 48
SHAPES = {
    "t2i": [(1, 3, 5)],
    "edit": [(1, 3, 4), (1, 4, 2)],             # 생성 대상 + 참조 이미지
    "layered": [(1, 2, 4)] * 4,                 # layer 3장 + 조건 이미지 (frame -1)
}


def ref(variant: str = "edit"):
    import diffusers
    from diffusers import QwenImageTransformer2DModel
    if diffusers.__version__ != DIFFUSERS_VERSION:
        print(f"⚠️ diffusers {diffusers.__version__} (검토한 원본은 {DIFFUSERS_VERSION})")
    m = QwenImageTransformer2DModel(
        patch_size=2, in_channels=16, out_channels=4, num_layers=2,
        attention_head_dim=16, num_attention_heads=4, joint_attention_dim=TXT_DIM,
        guidance_embeds=False, axes_dims_rope=(4, 6, 6),
        zero_cond_t=variant == "edit",
        use_additional_t_cond=variant == "layered", use_layer3d_rope=variant == "layered")
    shapes = SHAPES[variant]
    n = sum(f * h * w for f, h, w in shapes)
    g = torch.Generator().manual_seed(0)
    inputs = {
        "hidden_states": torch.randn(BATCH, n, 16, generator=g),
        "encoder_hidden_states": torch.randn(BATCH, TXT_LEN, TXT_DIM, generator=g),
        "encoder_hidden_states_mask": torch.tensor([[1] * TXT_LEN] + [[1, 1, 0, 1, 1, 0, 0]] * (BATCH - 1)),
        "timestep": torch.linspace(0.3, 0.85, BATCH),
        "img_shapes": [list(shapes) for _ in range(BATCH)],
    }
    if variant == "layered":
        inputs["additional_t_cond"] = torch.arange(BATCH) % 2
    return m, inputs


CASE = Case(STAGE, "MM-DiT dual stream: joint attention + adaLN + QK-norm + MSRoPE (+ 편집 zero_cond_t)",
            "Qwen-Image (Edit-2511)", f"diffusers transformer_qwenimage.py@{DIFFUSERS_COMMIT[:7]} (v0.41.0)",
            ref, "miniarch.models.qwen_image", atol=1e-4)
