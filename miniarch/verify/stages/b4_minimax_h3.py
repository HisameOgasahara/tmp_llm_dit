"""B4 · MiniMax-H3 (비디오 + 오디오 joint 3D DiT) 검증 케이스.

원본 = DiffSynth-Studio `diffsynth/models/minimax_h3_dit.py` (커밋 고정, diffusers 에는 아직 없음).
DiffSynth 패키지 전체를 설치하지 않고, 원본이 import 하는 파일 셋만 받아 임시 패키지(`_diffsynth_ref`)로 묶는다.
  - models/minimax_h3_dit.py           : 원본 DiT (한 글자도 안 고침)
  - core/attention/attention.py         : attention_forward (DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch → SDPA)
  - core/gradient/gradient_checkpoint.py: gradient_checkpoint_forward (끄면 그냥 module(...))
입력은 DiffSynth 파이프라인의 packed 규칙을 옮긴 `miniarch.models.minimax_h3.build_inputs` 로 만든다.
"""
from __future__ import annotations

import os
import sys
import types

import torch

from ..cases import Case

STAGE = "B4"
SLUG = "minimax_h3"
TITLE = "MiniMax-H3 (비디오 + 오디오 joint 3D DiT)"
INTRO = """
원본: DiffSynth-Studio `diffsynth/models/minimax_h3_dit.py` (`MiniMaxH3DiT`, commit `4a2d8af` 고정).
diffusers에는 아직 MiniMax-H3가 없어서, DiffSynth 전체 대신 이 파일과 이 파일이 import하는 `core/attention`, `core/gradient`
두 파일만 받아 묶어 씁니다 (attention은 torch SDPA 경로로 고정). 실제 체크포인트(`MiniMaxAI/MiniMax-H3` `FL2VA/transformer`)와
가중치 이름이 같습니다.

내 구현: `miniarch/models/minimax_h3.py`. 파라미터 이름은 원본과 같습니다.

부품
- **한 줄 packed 시퀀스** (batch 1): `[text | 조건 keyframe | audio | video | pad]`, 길이는 64의 배수로 padding
- 토큰화: video latent (1,2,2) patch → `video_patch_proj`, audio latent 채널별 시간 행 → `audio_patch_proj`,
  text → `condition_proj` → **token refiner** (RoPE·변조 없는 2층 transformer)
- **3축 RoPE** (t, h, w 실수 좌표, 축마다 주파수 16개): head_dim 128 중 앞 96 채널만 회전
- 블록 50개는 **모든 modality가 같은 가중치** (단일 스트림). attention은 실제 길이 / pad 두 구간으로 나눈 양방향
- **modality별 adaLN**: timestep 임베딩 하나 → (video / text / audio) × (shift, scale, gate) × (attn, mlp).
  토큰마다 자기 timestep과 modality 행을 고름. 조건 프레임은 t = 0.999, 편집(retake)에서 고정한 칸은 t = 1
- 출력층: modality 공통 shift/scale → `video_out` / `audio_out`

축소 config (모든 경로가 한 번씩 쓰이도록)
- hidden 64, head 4×16, 블록 2, refiner 2층. rope 주파수 축마다 2개 → 회전 12채널 + 회전 안 하는 4채널 (실제 96 / 128처럼)
- video latent [1, 4, T=3, H=4, W=6] → 18행, keyframe 조건 2장(첫/마지막 프레임 위치) → 12행, audio 2채널 × 5 → 10행,
  text 7토큰(그중 하나는 modality tag 0) → 실제 47 + pad 17 = 64
- timestep 4종(video 0.3 / audio 0.6 / 조건 0.999 / 편집 고정 1.0), 출력 mask(`skip_mask_out_condition=False`), ControlNet hint 1개
- 출력은 video 행과 audio 행을 모두 이어 붙여 비교합니다.
"""
PIP = ["einops"]
HEADER = [("MiniMaxAI/MiniMax-H3",
           r"^(blocks\.0\.|token_refiner\.blocks\.0\.|final_layer\.|time_embedder\.|rope\.|\w+_proj\.)",
           "FL2VA/transformer")]
CAUSAL = False
SELF_CHECKS = [
    ("""
### attention 구간 · 확률 합 · modality 간 주의

- `attn_probs`: [1, H, S, S]. 행 합 = 1, 실제 토큰 ↔ pad 사이 확률은 정확히 0 이어야 합니다 (구간 나눔).
- 0번 블록에서 query modality별로 key modality(text / video / audio)에 준 평균 확률 질량을 출력합니다.
- `mod_probe`: 서로 다른 변조 벡터 수 = 서로 다른 (timestep, modality) 쌍 수.
""", """
from miniarch.models import minimax_h3 as H
ref, inp = S.ref()
m = H.verify_pair(ref)["mine"]
V.copy_weights(ref, m)
m = m.to(dev).eval()
mi = H.mine_inputs({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()})
names = [n for n, _ in m.named_modules() if n.endswith(("attn_probs", "mod_probe"))]
with V.capture(m, names) as st, torch.no_grad():
    m(**mi)
S_ = inp["x"].shape[1]
used = int(inp["packed_seq_params"]["cu_seqlens_q"][1])
seg = (torch.arange(S_, device=dev) >= used).long()
cross = seg[:, None] != seg[None, :]
for n in names:
    if n.endswith("attn_probs"):
        p = st[n]
        err = (p.sum(-1) - 1).abs().max().item()
        report(f"{n} 행 합 = 1", err < 1e-5, err)
        if p.shape[-1] == S_:   # 본 블록만 (refiner 는 text 길이 L × L, pad 없음)
            leak = p[:, :, cross].abs().max().item()
            report(f"{n} 실제 ↔ pad 확률 = 0", leak == 0, leak)
tags = inp["token_tags"].to(dev)
p0 = st["blocks.0.attn.attn_probs"][0].mean(0)
names_ = {1: "text", 0: "video", 2: "audio"}
for a in (1, 0, 2):
    row = {names_[b]: round(p0[tags == a][:, tags == b].sum(-1).mean().item(), 3) for b in (1, 0, 2)}
    print(f"query {names_[a]:<5} → {row}")
mp = st["blocks.0.mod_probe"]
key = inp["inverse_indices"].to(dev) * 3 + tags.clamp(min=0)
nd = torch.unique(mp.permute(1, 0, 2).reshape(S_, -1), dim=0).shape[0]
print("timestep 고유값", inp["unique_timesteps"].tolist(), " 변조 벡터 종류", nd)
report("변조 벡터 수 = (timestep, modality) 쌍 수", nd == key.unique().numel(), 0.0)
"""),
    ("""
### padding 길이 무관 · t2va 경로 · 빈 text(CFG null) 경로

- 조건·편집 없는 기본 t2va 입력을 padding 단위 64 / 128 로 만들어, 원본 출력이 같은지 (pad 구간이 실제 토큰에 안 섞이는지) 봅니다.
- 같은 입력에서 내 구현 == 원본, 그리고 text 길이 0 (null text) 입력에서도 내 구현 == 원본인지 확인합니다.
- CFG: 길이가 다른 negative prompt로 한 번 더 돌려 `uncond + s·(cond − uncond)` 를 만들어 봅니다.
""", """
from miniarch.models import minimax_h3 as H
ref, _ = S.ref()
m = H.verify_pair(ref)["mine"]
V.copy_weights(ref, m)
ref, m = ref.to(dev).eval(), m.to(dev).eval()
g = torch.Generator().manual_seed(5)
vl, al = torch.randn(1, 4, 2, 4, 4, generator=g).to(dev), torch.randn(2, 8, 3, generator=g).to(dev)
pe, neg = torch.randn(5, 40, generator=g).to(dev), torch.randn(3, 40, generator=g).to(dev)

def run(text, align=64, tv=500, ta=500):
    d = H.build_inputs(vl, al, text, torch.ones(text.shape[0], dtype=torch.long), tv, ta, seq_align=align)
    with torch.no_grad():
        return H.joined(ref(**d)), H.joined(m(**H.mine_inputs(d)))

r64, m64 = run(pe, 64)
r128, m128 = run(pe, 128)
e = (r64 - r128).abs().max().item()
report("padding 64 vs 128 원본 출력 동일", e < 5e-5, e)   # 행 수가 달라 GEMM 커널이 바뀔 수 있음
e = max((r64 - m64).abs().max().item(), (r128 - m128).abs().max().item())
report("t2va 경로 내 구현 == 원본", e < 1e-4, e)
r0, m0 = run(pe[:0])
e = (r0 - m0).abs().max().item()
report("빈 text 경로 내 구현 == 원본", e < 1e-4, e)
rn, mn = run(neg)
e = (rn - mn).abs().max().item()
report("negative prompt(길이 3) 내 구현 == 원본", e < 1e-4, e)
for s in (1.0, 5.0):
    cfg = mn + s * (m64 - mn)
    print(f"CFG s={s}: |guided| 평균 {cfg.abs().mean():.4f}  (cond 평균 {m64.abs().mean():.4f})")
"""),
    ("""
### text 순서 · timestep 민감도

- token refiner는 위치 정보가 없어 text 순서를 바꾸면 출력도 같은 순서로 바뀌기만 해야 합니다 (permutation equivariant).
- 본 블록에서는 text도 t축 RoPE 위치(0..L-1)를 받으므로 순서를 바꾸면 video/audio 출력이 달라져야 합니다.
- audio timestep만 바꿔도 joint attention 때문에 video 출력이 달라져야 합니다.
""", """
from miniarch.models import minimax_h3 as H
ref, _ = S.ref()
m = H.verify_pair(ref)["mine"]
V.copy_weights(ref, m)
m = m.to(dev).eval()
g = torch.Generator().manual_seed(6)
vl, al = torch.randn(1, 4, 2, 4, 4, generator=g).to(dev), torch.randn(2, 8, 3, generator=g).to(dev)
pe = torch.randn(6, 40, generator=g).to(dev)
perm = torch.randperm(6, generator=g).to(dev)
L_ = pe.shape[0]
with torch.no_grad():
    mask = H.segment_mask(torch.tensor([0, L_, L_]), L_, dev)
    z = m.condition_proj(pe)
    e = (m.token_refiner(z, mask)[perm] - m.token_refiner(z[perm], mask)).abs().max().item()
report("refiner: text 순서 equivariant", e < 1e-5, e)

def video_out(text, tv=500, ta=500):
    d = H.build_inputs(vl, al, text, torch.ones(text.shape[0], dtype=torch.long), tv, ta)
    with torch.no_grad():
        return m(**H.mine_inputs(d))[0]

base = video_out(pe)
d1 = (video_out(pe[perm]) - base).abs().max().item()
report("본 블록: text 순서가 video 출력을 바꿈 (RoPE)", d1 > 1e-4, d1)
d2 = (video_out(pe, ta=900) - base).abs().max().item()
report("audio timestep만 바꿔도 video 출력이 바뀜", d2 > 1e-4, d2)
d3 = (video_out(pe, tv=900) - base).abs().max().item()
report("video timestep 민감도", d3 > 1e-4, d3)
"""),
    ("""
### ablation 스위치

나중에 부품별 비교 실험에 쓸 스위치를 하나씩 끄고 forward가 정상(유한값)이고 출력이 실제로 바뀌는지 확인합니다.
""", """
import dataclasses
from miniarch.models import minimax_h3 as H
ref, inp = S.ref()
base = H.config_from_module(ref)
m0 = H.MiniMaxH3(base)
V.copy_weights(ref, m0)
m0 = m0.to(dev).eval()
mi = H.mine_inputs({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()})
with torch.no_grad():
    y0 = H.joined(m0(**mi))
for sw in ["use_rope", "use_qk_norm", "use_token_refiner", "modality_adaln", "use_gate"]:
    m = H.MiniMaxH3(dataclasses.replace(base, **{sw: False})).to(dev).eval()
    m.load_state_dict(m0.state_dict())
    with torch.no_grad():
        y = H.joined(m(**mi))
    d = (y - y0).abs().max().item()
    report(f"{sw}=False forward (유한·출력 변화)", bool(torch.isfinite(y).all()) and d > 1e-5, d)
"""),
]

DIFFSYNTH_COMMIT = "4a2d8af77849b20702b2639338f6fe4512f2f93e"
_PKG = "_diffsynth_ref"
_FILES = [("core.attention", "diffsynth/core/attention/attention.py"),
          ("core.gradient", "diffsynth/core/gradient/gradient_checkpoint.py"),
          ("models.minimax_h3_dit", "diffsynth/models/minimax_h3_dit.py")]


def load_reference():
    """원본 파일 셋을 받아 `_diffsynth_ref` 패키지로 묶어 import → minimax_h3_dit 모듈."""
    key = f"{_PKG}.models.minimax_h3_dit"
    if key in sys.modules:
        return sys.modules[key]
    import importlib.util
    import urllib.request
    cache = os.path.join(os.path.expanduser("~"), ".cache", "miniarch", f"diffsynth_{DIFFSYNTH_COMMIT[:12]}")
    os.makedirs(cache, exist_ok=True)
    for name in (_PKG, f"{_PKG}.core", f"{_PKG}.models"):   # 상대 import (..core.attention) 용 빈 패키지
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules[name] = pkg
    prev = os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION")
    os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"   # flash/sage 가 깔려 있어도 SDPA 로
    try:
        for name, path in _FILES:
            local = os.path.join(cache, path.replace("/", "__"))
            if not os.path.exists(local):
                url = f"https://raw.githubusercontent.com/modelscope/DiffSynth-Studio/{DIFFSYNTH_COMMIT}/{path}"
                urllib.request.urlretrieve(url, local + ".part")   # 받다 끊겨도 깨진 캐시가 남지 않도록
                os.replace(local + ".part", local)
            spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}", local)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
    except Exception:
        for name, _ in _FILES:
            sys.modules.pop(f"{_PKG}.{name}", None)
        raise
    finally:
        if prev is None:
            os.environ.pop("DIFFSYNTH_ATTENTION_IMPLEMENTATION", None)
        else:
            os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = prev
    return sys.modules[key]


def ref():
    from miniarch.models.minimax_h3 import build_inputs
    R = load_reference()
    hidden = 64
    m = R.MiniMaxH3DiT(
        num_layers=2, token_refiner_num_layers=2, hidden_size=hidden, num_attention_heads=4, attention_head_dim=16,
        ffn_hidden_size=96, latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=40,
        timestep_input_dim=32, time_embed_hidden_size=48, time_embed_dim=32,
        adaln_out_features=6 * hidden * 3, final_adaln_out_features=2 * hidden, rope_inv_freq_len=2,
        norm_eps=1e-5, qk_norm_eps=1e-6, final_norm_eps=1e-4)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():   # RMSNorm weight 는 1 로 시작 → 1 근처 다른 값으로
        for name, p in m.named_parameters():
            if name.endswith("norm.weight") or name.endswith(("norm1.weight", "norm2.weight")):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))

    g = torch.Generator().manual_seed(0)
    video = torch.randn(1, 4, 3, 4, 6, generator=g)              # T=3, H=4, W=6 → 프레임당 6행
    audio = torch.randn(2, 8, 5, generator=g)                    # 채널 2 × 5 step
    text = torch.randn(7, 40, generator=g)
    tags = torch.tensor([1, 1, 0, 1, 1, 1, 1])                   # text 중 하나는 modality tag 0 (시각 토큰 흉내)
    anchor = torch.randn(12, 16, generator=g)                    # keyframe 2장 × 6행
    mask_v = torch.ones_like(video)
    mask_v[:, :, 0] = 0                                          # 편집: 0번 프레임 고정
    mask_a = torch.ones_like(audio)
    mask_a[:, :, :2] = 0                                         # 편집: 앞 2 step 고정
    inp = build_inputs(video, audio, text, tags, 700, 400, keyframe_cond_anchor=anchor, keyframe_indices=[0, -1],
                       input_latents_video=torch.randn(video.shape, generator=g), denoise_mask_video=mask_v,
                       input_latents_audio=torch.randn(audio.shape, generator=g), denoise_mask_audio=mask_a)
    n_img, n_aud, seq = inp["img_pos_info"]["position_ids"].numel(), inp["audio_pos_info"]["position_ids"].numel(), inp["x"].shape[1]
    update = torch.ones(n_img)
    update[:12] = 0                                              # 조건 행 출력은 버림
    inp.update(update_mask=update, update_audio_mask=(torch.rand(n_aud, generator=g) > 0.3).float(),
               skip_mask_out_condition=False, control_hints={1: 0.1 * torch.randn(seq, hidden, generator=g)})
    return m, inp


CASE = Case(STAGE, "MiniMax-H3 joint 비디오+오디오 3D DiT (packed 단일 스트림, 3축 RoPE, modality별 adaLN, token refiner)",
            "MiniMax-H3", "DiffSynth diffsynth/models/minimax_h3_dit.py@4a2d8af",
            ref, "miniarch.models.minimax_h3", atol=1e-4)
