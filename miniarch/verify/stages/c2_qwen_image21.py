"""C2 · Qwen-Image-2.1 (Single-Stream DiT, 생성 + 편집 통합) 검증 케이스."""
from __future__ import annotations

import torch

from ..cases import BATCH, Case

STAGE = "C2"
SLUG = "qwen_image21"
TITLE = "Qwen-Image-2.1 (Single-Stream DiT · block-causal · 편집 조건)"
INTRO = """
원본: diffusers `QwenImage21Transformer2DModel` (`src/diffusers/models/transformers/transformer_qwenimage21.py`,
diffusers **v0.41.0** = commit `086bf95` 고정, 이후 main 의 변경은 tensor parallel 용이라 수치 동일).
체크포인트: HF `Qwen/Qwen-Image-2.1` (revision `d26bb61`), `transformer/config.json` 의 `_class_name` 이 위 클래스.
내 구현: `miniarch/models/qwen_image21.py`. VL 인코더(Qwen3-VL)는 구현하지 않고 랜덤 hidden state 를 넣습니다.

부품
- 한 줄 시퀀스: VL 시퀀스의 이미지 칸(1칸 = latent 2×2 = 토큰 4개)에 조건 이미지 latent 를, 맨 끝에 타깃 latent 를 끼움
- txt_in connector: zero-center RMSNorm(가중치+1) → Linear → GELU(tanh) → Linear
- 3축 RoPE: 텍스트는 세 축 같이 전진, 이미지 블록은 frame 고정 + 0 중심 h/w 격자 (음수 위치)
- block-causal attention: (key ≤ query) 또는 같은 이미지 블록. padding 텍스트는 key 에서만 제외. QK RMSNorm
- 공유 modulation 하나 → 블록마다 [scale, tanh(gate)] × (attn, MLP), shift 없음. SwiGLU MLP
- causal_condition: 텍스트·조건 이미지는 t=0 행으로 modulation → prefix 가 timestep·타깃과 무관 (KV cache 가능)

축소 config: 2층, 폭 64 (head 4 × 16), RoPE 축 (4, 6, 6), latent 8채널 → 출력 12채널, VL 폭 32.
VL 시퀀스 = 텍스트5 | 조건A 2칸(2×4) | 텍스트3 | 조건B 1칸(2×2) | 조건C 1칸(1×4, B 바로 옆) | 텍스트2 | 타깃 3칸(3×4) → 결합 토큰 38개.
- 조건 이미지 셋(편집 경로), 바로 붙은 두 블록(B·C 가 서로 못 보는지), 홀수 높이·높이 1 격자(음수 위치) 를 모두 지나게 했습니다.
- 두 번째 샘플은 텍스트 3자리를 padding 으로 막고, 첫 샘플은 조건 이미지 칸의 mask 를 꺼서 (무시되어야 함) 마스크 경로를 씁니다.
- 원본은 segment 마다 SDPA 를 부르고, 내 구현은 같은 마스크를 한 번에 만든 명시적 softmax 입니다.
"""
PIP = ["diffusers==0.41.0"]   # 검토한 태그 v0.41.0 (086bf95) 그대로
HEADER = [("Qwen/Qwen-Image-2.1",
           r"^(transformer_blocks\.0\.|img_in|txt_in|modulation|norm_out|proj_out|time_text_embed)", "transformer")]
CAUSAL = False
SELF_CHECKS = [
    ("""
### attention map · 마스크 구조 · modulation 확인

- `attn_probs` [B, H, N, N]: 행 합 = 1, 막힌 칸(미래 텍스트, 뒤쪽 블록, padding key) 확률 = 0.
- 같은 이미지 블록 안은 전부 허용(양방향), 붙어 있는 조건 블록 B·C 는 B→C 방향이 막혀야 합니다.
- 텍스트→이미지 / 이미지→텍스트 / 이미지→이미지 확률 질량을 출력합니다.
- `causal_condition`: prefix 토큰의 scale/gate(t=0 행)는 두 샘플이 같고, 타깃 토큰은 샘플마다 달라야 합니다.
""", """
from miniarch.models import qwen_image21 as Q
V.set_deterministic(0)
ref, inputs = S.ref()
m = Q.verify_pair(ref)["mine"]; V.copy_weights(ref, m); m = m.to(dev).eval()
x = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()}
names = [n for n, _ in m.named_modules() if n.endswith(("attn_probs", "_probe"))]
with V.capture(m, names) as st, torch.no_grad():
    y0 = m(**x)
allow, ids, tgt = st["mask_probe"], m.last_meta["image_ids"], m.last_meta["target_mask"]
N = ids.shape[0]
i_, j_ = torch.arange(N, device=dev)[:, None], torch.arange(N, device=dev)[None, :]
same = (ids[:, None] == ids[None, :]) & (ids[:, None] >= 0)
report("같은 이미지 블록 안은 전부 허용", bool(allow[:, same].all()), 0.0)
report("미래 칸(다른 블록) 은 전부 막힘", not bool(allow[:, (j_ > i_) & ~same].any()), 0.0)
b, c = (ids == 1).nonzero()[0, 0], (ids == 2).nonzero()[0, 0]
report("붙어 있는 조건 블록 B 는 C 를 못 봄", not bool(allow[:, b, c].any()), 0.0)
txt = ids < 0
for n in [n for n in names if n.endswith("attn_probs")]:
    p = st[n]
    err = (p.sum(-1) - 1).abs().max().item()
    leak = p.masked_fill(allow[:, None], 0).abs().max().item()
    t2i = p[:, :, txt][..., ~txt].sum(-1).mean().item()
    i2t = p[:, :, ~txt][..., txt].sum(-1).mean().item()
    i2i = p[:, :, ~txt][..., ~txt].sum(-1).mean().item()
    print(f"{n:<34} 텍스트→이미지 {t2i:.3f}  이미지→텍스트 {i2t:.3f}  이미지→이미지 {i2i:.3f}")
    report(f"{n} 행 합 = 1", err < 1e-5, err)
    report(f"{n} 막힌 칸 확률 = 0", leak == 0.0, leak)
for k in ("scale1", "gate1", "scale2", "gate2"):
    g = st[f"transformer_blocks.0.{k}_probe"]
    d_pre = (g[0, ~tgt] - g[1, ~tgt]).abs().max().item()
    d_tgt = (g[0, tgt] - g[1, tgt]).abs().max().item()
    print(f"{k}: prefix |평균|={g[:, ~tgt].abs().mean():.3f}  타깃 |평균|={g[:, tgt].abs().mean():.3f}")
    report(f"{k} prefix 행은 샘플 간 동일 (t=0)", d_pre == 0.0, d_pre)
    report(f"{k} 타깃 행은 샘플마다 다름 (실제 t)", d_tgt > 1e-4, d_tgt)
"""),
    ("""
### prefix 불변성 (block-causal + causal_condition → KV cache 근거)

타깃 latent 와 timestep 을 바꿔도 텍스트·조건 이미지 위치 출력은 그대로여야 하고, 타깃 출력은 바뀌어야 합니다.
ablation: `causal_condition=False` 면 timestep 만 바꿔도 prefix 가 바뀌고, `block_causal=False` 면 타깃만 바꿔도 prefix 가 바뀝니다.
""", """
import dataclasses, math
nt = math.prod(x["img_shapes"][0][-1])
hs = x["hidden_states"].clone(); hs[:, -nt:] = torch.randn_like(hs[:, -nt:])
x_tgt = {**x, "hidden_states": hs}
x_t = {**x, "timestep": 1.0 - x["timestep"]}   # [0.3, 0.8] → [0.7, 0.2] (두 샘플 모두 바뀜)
with torch.no_grad():
    y1 = m(**{**x_tgt, "timestep": x_t["timestep"]})
d_pre, d_tgt = (y1 - y0)[:, ~tgt].abs().max().item(), (y1 - y0)[:, tgt].abs().max().item()
report("타깃·t 바꿔도 prefix 출력 불변", d_pre < 1e-5, d_pre)
report("타깃 출력은 바뀜", d_tgt > 1e-3, d_tgt)
for sw, xx in (("causal_condition", x_t), ("block_causal", x_tgt)):
    m2 = Q.QwenImage21(dataclasses.replace(m.config, **{sw: False})).to(dev).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        d = (m2(**xx) - m2(**x))[:, ~tgt].abs().max().item()
    report(f"{sw}=False 면 prefix 가 바뀜", d > 1e-4, d)
"""),
    ("""
### padding · 무시되는 입력 · CFG 경로

- padding 텍스트 위치의 VL hidden 을 바꿔도 유효 토큰 출력은 그대로 (key 에서 빠짐, 자기 행만 바뀜).
- 조건 이미지 칸의 VL hidden 은 latent 로 덮어써지므로 바꿔도 출력이 전혀 안 바뀜.
- CFG: 빈 조건(VL hidden 0) 출력과 섞은 `uncond + 4·(cond − uncond)` 가 유한하고 cond 와 다른지 확인.
""", """
em, im = x["encoder_hidden_states_mask"], x["img_mask"][:, :x["encoder_hidden_states_mask"].shape[1]]
e = x["encoder_hidden_states"]
e_pad = e.clone(); pad = (~em) & (~im); e_pad[pad] = torch.randn_like(e_pad[pad]) * 3
e_slot = e.clone(); e_slot[im] = torch.randn_like(e_slot[im]) * 3
with torch.no_grad():
    y_pad = m(**{**x, "encoder_hidden_states": e_pad})
    y_slot = m(**{**x, "encoder_hidden_states": e_slot})
    y_null = m(**{**x, "encoder_hidden_states": torch.zeros_like(e)})
kv = m.last_meta["key_valid"]
d = (y_pad - y0)[kv].abs().max().item()
report("padding 텍스트 내용은 유효 토큰에 영향 없음", d < 1e-5, d)
d = (y_pad - y0)[~kv].abs().max().item()
report("padding 토큰 자기 출력은 바뀜", d > 1e-4, d)
d = (y_slot - y0).abs().max().item()
report("조건 이미지 칸의 VL hidden 은 무시됨", d < 1e-6, d)
cfg = y_null + 4.0 * (y0 - y_null)
d = (cfg - y0)[:, tgt].abs().max().item()
report("CFG 출력 유한 + cond 와 다름", bool(torch.isfinite(cfg).all()) and d > 1e-3, d)
"""),
]


def ref():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    m = QwenImage21Transformer2DModel(
        patch_size=1, in_channels=8, out_channels=12, num_layers=2, attention_head_dim=16, num_attention_heads=4,
        context_in_dim=32, mlp_ratio=3, axes_dims_rope=(4, 6, 6), eps=1e-6, causal_condition=True)
    g = torch.Generator().manual_seed(0)
    # VL 시퀀스: 텍스트5 | 조건A 2칸 | 텍스트3 | 조건B 1칸 | 조건C 1칸 | 텍스트2 | 타깃 3칸 (타깃 칸은 encoder hidden 에 없음)
    layout = [0] * 5 + [1] * 2 + [0] * 3 + [1] + [1] + [0] * 2 + [1] * 3
    shapes = [(1, 2, 4), (1, 2, 2), (1, 1, 4), (1, 3, 4)]      # 조건들 → 타깃 (latent 토큰 단위)
    img_mask = torch.tensor(layout, dtype=torch.bool)[None].repeat(BATCH, 1)       # [B, 17]
    L = len(layout) - 3
    n_img = sum(f * h * w for f, h, w in shapes)                                   # 28
    mask = torch.ones(BATCH, L, dtype=torch.bool)
    mask[1, [4, 9, 13]] = False      # 샘플 1: 텍스트 padding (첫 토큰은 유효 → 완전히 막힌 행 없음)
    mask[0, 5] = False               # 샘플 0: 조건 이미지 칸 → 원본에서 무시됨
    inputs = {
        "hidden_states": torch.randn(BATCH, n_img, 8, generator=g),
        "encoder_hidden_states": torch.randn(BATCH, L, 32, generator=g),
        "timestep": torch.tensor([0.3, 0.8]),
        "img_shapes": [list(shapes) for _ in range(BATCH)],
        "img_mask": img_mask,
        "encoder_hidden_states_mask": mask,
    }
    return m, inputs


CASE = Case(STAGE, "Single-Stream DiT: block-causal 결합 시퀀스 + 공유 modulation + 편집 조건 이미지", "Qwen-Image-2.1",
            "diffusers transformer_qwenimage21.py (v0.41.0)", ref, "miniarch.models.qwen_image21", atol=1e-4)
