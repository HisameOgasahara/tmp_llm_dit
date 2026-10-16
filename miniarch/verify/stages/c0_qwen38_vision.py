"""C0 · Qwen3.8-27B 내장 ViT (patch embed · 보간 위치 · 2D RoPE · patch merger) + LM 쪽 M-RoPE 검증 케이스."""
from __future__ import annotations

import torch

from ..cases import VOCAB, Case

STAGE = "C0"
SLUG = "qwen38_vision"
TITLE = "Qwen3.8-27B 비전 인코더 (ViT + M-RoPE)"
INTRO = """
원본: transformers `qwen3_5` 의 `Qwen3_5VisionModel` + `Qwen3_5Model.get_rope_index`
(코드 읽은 기준: tag `v5.20.0` = commit `48e4bcd`, `src/transformers/models/qwen3_5/modeling_qwen3_5.py` + `src/transformers/vision_utils.py`).
Qwen3.8-27B 체크포인트(`Qwen/Qwen3.8-27B`)의 `model_type` 이 `qwen3_5` 라 이 코드를 그대로 씁니다.
내 구현: `miniarch/models/qwen38_vision.py` (텍스트 쪽은 A6a 의 `miniarch/models/qwen36.py` 재사용).

부품
- patch embed: 프레임 2장 × 16×16 픽셀 patch 를 Conv3d(커널 = stride) 한 번으로 벡터 하나로. patch 순서는 2×2 merge 블록 순서
- 절대 위치: 학습된 48×48 위치 표를 이미지 격자 크기로 bilinear 보간 (align_corners=True) 해서 더함
- 2D RoPE: head_dim 을 [행 주파수 | 열 주파수] × 2 로 채워 전체 회전
- attention: 프레임 하나 안에서만 양방향 full attention (Qwen2.5-VL 의 window attention 은 없음)
- patch merger: LayerNorm → 이웃 2×2 patch 이어붙이기 → MLP → LM 폭 (27B: 5120)
- LM 쪽 M-RoPE: 이미지 토큰은 (t, h, w) 3축 위치, 다음 텍스트는 max(h, w) 만큼만 건너뜀. 회전 채널은 interleaved 배치

축소 config (모든 경로가 한 번씩 쓰이도록 고름)
- ViT 2층, 폭 64, head 4 (head_dim 16 → 축마다 주파수 2개), patch 4, 위치 표 4×4
- 입력: 이미지 (1, 4, 6) + 2프레임 비디오 (2, 6, 4) + 아주 작은 이미지 (1, 2, 2) 를 한 열로 패킹
  → 위치 표보다 큰/작은 격자 보간, 끝 clamp, 프레임별 attention 구간, 비디오 프레임 반복 위치가 모두 지나감
- 최종 비교는 merger 출력 (LM 에 들어가는 토큰), 층별 출력도 비교합니다.
- 자체 점검: get_rope_index 와 위치 일치 (padding·비디오 프레임 분할 포함), M-RoPE cos/sin 일치,
  ViT + Qwen36 을 이어 붙인 미니 VLM 이 원본 `Qwen3_5Model` 과 같은지.
"""
PIP = []
HEADER = [("Qwen/Qwen3.8-27B", r"visual\.(patch_embed|pos_embed|blocks\.0\.|merger)", "")]
CAUSAL = False
SELF_CHECKS = [
    ("""
### M-RoPE 위치 == 원본 get_rope_index

텍스트·이미지·비디오가 섞인 배치 (왼쪽 padding 포함) 에서 3축 위치와 delta 가 원본과 정확히 같은지 봅니다.
- 비디오 (2, 4, 4) 는 프레임 사이에 타임스탬프 텍스트가 끼므로 프레임마다 따로 위치를 받습니다.
- 원본 함수는 모델 메서드라 config 만 흉내 낸 객체로 호출합니다 (가중치 불필요).
""", """
import types
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from miniarch.models import qwen38_vision as Q
ns = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2)))
ns.get_vision_position_ids = types.MethodType(Qwen3_5Model.get_vision_position_ids, ns)

# (종류, 토큰 수): 0 텍스트, 1 이미지, 2 비디오 프레임
rows = [[(0, 2), (1, 6), (0, 3), (2, 4), (0, 1), (2, 4), (0, 2)],
        [(0, 4), (1, 1), (0, 2), (1, 6), (0, 5), (2, 2), (0, 1)]]
img_thw = torch.tensor([[1, 4, 6], [1, 2, 2], [1, 6, 4]])
vid_thw = torch.tensor([[2, 4, 4], [1, 2, 4]])
T_ = max(sum(n for _, n in r) for r in rows) + 3
mm = torch.zeros(len(rows), T_, dtype=torch.int32)
am = torch.zeros(len(rows), T_, dtype=torch.long)
for b, r in enumerate(rows):
    types_ = sum(([k] * n for k, n in r), [])
    mm[b, T_ - len(types_):] = torch.tensor(types_)
    am[b, T_ - len(types_):] = 1                                     # 왼쪽 padding
ids = torch.zeros(len(rows), T_, dtype=torch.long)
p_ref, d_ref = Qwen3_5Model.get_rope_index(ns, ids, mm, image_grid_thw=img_thw.clone(),
                                           video_grid_thw=vid_thw.clone(), attention_mask=am)
p_me, d_me = Q.rope_index(mm, img_thw.clone(), vid_thw.clone(), attention_mask=am, merge=2)
print("t/h/w 위치 (0번 열):\\n", p_me[:, 0])
print("delta:", d_me.flatten().tolist())
err = (p_ref - p_me).abs().max().item()
report("M-RoPE 3축 위치 == get_rope_index", p_ref.shape == p_me.shape and err == 0, err)
report("rope delta 일치", torch.equal(d_ref.cpu().long(), d_me.cpu().long()), 0.0)
MROPE_POS = p_me
"""),
    ("""
### M-RoPE cos/sin == 원본 Qwen3_5TextRotaryEmbedding

위 위치로 회전 각을 만들 때, 주파수 i 를 i mod 3 = 0/1/2 → t/h/w 축으로 고르는 interleaved 배치가 원본과 같은지 봅니다.
(rotary 폭 16 → 주파수 8개, mrope_section [3, 3, 2] 로 세 축이 모두 쓰이게 함)
""", """
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
tc = Qwen3_5TextConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                       rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                        "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2]})
rot = Qwen3_5TextRotaryEmbedding(tc)
cos_r, sin_r = rot(torch.zeros(1), MROPE_POS)
cos_m, sin_m = Q.mrope_cos_sin(MROPE_POS, rot.inv_freq, [3, 3, 2])
err = max((cos_r - cos_m).abs().max().item(), (sin_r - sin_m).abs().max().item())
report("M-RoPE cos/sin (interleaved) 일치", cos_r.shape == cos_m.shape and err < 1e-6, err)
"""),
    ("""
### 미니 VLM == 원본 Qwen3_5Model

내 ViT + `qwen36.py` 텍스트 층 + M-RoPE 를 이어 붙인 `Qwen38VL` 이 원본 `Qwen3_5Model` (vision + 하이브리드 LM) 과
같은 last_hidden_state 를 내는지 봅니다. 이미지 2장 (격자 4×4, 2×8) + 2프레임 비디오 2개, 시퀀스 26.
가중치 이름이 같아 `load_state_dict` 로 그대로 옮깁니다.
""", """
import dataclasses
V.set_deterministic(0)
vlm, vin = S.ref_vlm()
V.perturb_params(vlm)
vlm = vlm.to(dev).float().eval()
for mod in vlm.modules():
    try:
        mod.config._attn_implementation = "eager"
    except Exception:
        pass
mine = Q.Qwen38VL(Q.vl_config_from_hf(vlm.config)).to(dev).eval()
res = mine.load_state_dict(vlm.state_dict(), strict=False)
print("missing:", res.missing_keys[:5], " unexpected:", res.unexpected_keys[:5])
report("VLM state_dict 키 일치", not res.missing_keys and not res.unexpected_keys, 0.0)
vin = {k: v.to(dev) for k, v in vin.items()}
with torch.no_grad():
    a = vlm(**vin).last_hidden_state
    b = mine(**vin)
err = (a - b).abs().max().item()
report("미니 VLM == Qwen3_5Model", a.shape == b.shape and err < 1e-4, err)

mine2 = Q.Qwen38VL(dataclasses.replace(mine.config, use_mrope=False)).to(dev).eval()
mine2.load_state_dict(mine.state_dict())
with torch.no_grad():
    d = (mine2(**vin) - b).abs().max().item()
report("M-RoPE 끄면 (1D 위치) 출력이 바뀜", d > 1e-4, d)
"""),
    ("""
### 프레임별 attention · ablation 스위치

- 비디오 2번째 프레임 픽셀만 바꾸면 1번째 프레임과 다른 이미지의 ViT 출력 (merger 전) 은 그대로여야 합니다.
- 2D RoPE / 보간 위치 임베딩을 끄거나 attention 범위를 비디오 전체(`clip`)로 넓히면 출력이 바뀌는지 봅니다.
""", """
import dataclasses
ref, inputs = S.ref()
V.perturb_params(ref)
m = Q.verify_pair(ref)["mine"]
m.load_state_dict(ref.state_dict())
m = m.to(dev).eval()
inp = {k: v.to(dev) for k, v in inputs.items()}
with torch.no_grad():
    base = m(**inp)
    x2 = inp["hidden_states"].clone()
    x2[48:72] += 1.0                                                  # 비디오 (2, 6, 4) 의 2번째 프레임 = patch 48..71
    alt = m(hidden_states=x2, grid_thw=inp["grid_thw"]).last_hidden_state
keep = torch.cat([torch.arange(0, 48), torch.arange(72, alt.shape[0])]).to(dev)
err = (alt[keep] - base.last_hidden_state[keep]).abs().max().item()
report("다른 프레임/이미지 출력 불변", err < 1e-5, err)
d = (alt[48:72] - base.last_hidden_state[48:72]).abs().max().item()
report("바꾼 프레임 출력은 변함", d > 1e-4, d)
for kw in (dict(use_rope=False), dict(use_abs_pos=False), dict(attn_scope="clip")):
    m2 = Q.Qwen38Vision(dataclasses.replace(m.config, **kw)).to(dev).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        d = (m2(**inp).pooler_output - base.pooler_output).abs().max().item()
    report(f"ablation {kw} → 출력 변화", d > 1e-4, d)
"""),
]

PATCH, TP, CH = 4, 2, 3


def _pixels(grid, seed):
    n = int(grid.prod(-1).sum())
    return torch.randn(n, CH * TP * PATCH * PATCH, generator=torch.Generator().manual_seed(seed))


def ref():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    cfg = Qwen3_5VisionConfig(
        depth=2, hidden_size=64, intermediate_size=96, num_heads=4, in_channels=CH,
        patch_size=PATCH, temporal_patch_size=TP, spatial_merge_size=2, out_hidden_size=48,
        num_position_embeddings=16, rope_parameters={"rope_type": "axial", "rope_theta": 10000.0})
    m = Qwen3_5VisionModel(cfg)
    m.config._attn_implementation = "eager"
    grid = torch.tensor([[1, 4, 6], [2, 6, 4], [1, 2, 2]])   # 이미지 · 2프레임 비디오 · 작은 이미지 (patch 76개)
    return m, {"hidden_states": _pixels(grid, 0), "grid_thw": grid}


def ref_vlm():
    """원본 Qwen3_5Model (ViT + 하이브리드 LM) 축소판과 텍스트·이미지·비디오가 섞인 입력."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    IMG, VID, VS, VE = 120, 121, 122, 123
    cfg = Qwen3_5Config(
        text_config=dict(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
            linear_conv_kernel_dim=4, max_position_embeddings=256, use_cache=False,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                             "mrope_section": [2, 1, 1], "mrope_interleaved": True}),
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=48, num_heads=2, in_channels=CH,
            patch_size=PATCH, temporal_patch_size=TP, spatial_merge_size=2, out_hidden_size=64,
            num_position_embeddings=16, rope_parameters={"rope_type": "axial", "rope_theta": 10000.0}),
        image_token_id=IMG, video_token_id=VID, vision_start_token_id=VS, vision_end_token_id=VE)
    m = Qwen3_5Model(cfg)
    g = torch.Generator().manual_seed(0)
    ids, types_ = [], []
    for _ in range(2):   # [텍스트 3, <vs> 이미지 4 <ve>, 텍스트 2, <vs> 프레임1 4 <ve>, 텍스트 1, <vs> 프레임2 4 <ve>, 텍스트 2]
        segs = [(None, 3), (VS, 1), (IMG, 4), (VE, 1), (None, 2), (VS, 1), (VID, 4), (VE, 1), (None, 1),
                (VS, 1), (VID, 4), (VE, 1), (None, 2)]
        row, ty = [], []
        for tok, n in segs:
            row += torch.randint(0, 100, (n,), generator=g).tolist() if tok is None else [tok] * n
            ty += [{IMG: 1, VID: 2}.get(tok, 0)] * n
        ids.append(row)
        types_.append(ty)
    img_thw = torch.tensor([[1, 4, 4], [1, 2, 8]])
    vid_thw = torch.tensor([[2, 4, 4], [2, 4, 4]])
    return m, {"input_ids": torch.tensor(ids), "mm_token_type_ids": torch.tensor(types_, dtype=torch.int32),
               "pixel_values": _pixels(img_thw, 1), "image_grid_thw": img_thw,
               "pixel_values_videos": _pixels(vid_thw, 2), "video_grid_thw": vid_thw}


CASE = Case(STAGE, "Qwen3.5 ViT (Conv3d patch · 보간 위치 · 2D RoPE · 프레임별 attention · merger)", "Qwen3.8-27B",
            "transformers/models/qwen3_5 (Qwen3_5VisionModel)", ref, "miniarch.models.qwen38_vision", atol=1e-4)
