"""B3 · Anima (Cosmos-Predict2-2B DiT + LLM adapter) 검증 케이스.

원본 = diffusers `CosmosTransformer3DModel` (transformer_cosmos.py) + `AnimaTextConditioner` (condition_embedder_anima.py).
파이프라인(modular_pipelines/anima)의 한 denoise step 처럼 adapter → DiT 를 이어 붙인 작은 래퍼를 원본으로 쓴다.
"""
from __future__ import annotations

import torch
from torch import nn

from ..cases import BATCH, Case

STAGE = "B3"
SLUG = "anima"
TITLE = "Anima (Cosmos-Predict2 DiT + LLM adapter)"
INTRO = """
원본: diffusers `CosmosTransformer3DModel` (`models/transformers/transformer_cosmos.py`)
+ `AnimaTextConditioner` (`models/condition_embedders/condition_embedder_anima.py`).
diffusers `v0.41.0` (commit `086bf95`)에서 처음 PyPI에 들어왔고, 두 파일은 main `af4271c`와 같습니다.
실제 체크포인트: `circlestone-labs/Anima-Base-v1.0-Diffusers` (`073c3a9`).
내 구현: `miniarch/models/anima.py`. 가중치 이름은 diffusers와 같습니다 (`text_conditioner.*`, `transformer.*`).

Anima = Cosmos-Predict2-2B T2I DiT를 그대로 쓰고, 텍스트 쪽만 바꾼 모델입니다.
- LLM adapter: T5 토큰 임베딩을 질의로 삼아 Qwen3-0.6B hidden을 cross-attn으로 읽습니다 (6블록, q/k RMSNorm + RoPE).
  결과를 RMSNorm하고 T5 padding을 0으로 만든 뒤 512 토큰까지 0으로 채웁니다.
- Cosmos DiT: padding mask 채널 concat, (1,2,2) 3D patch, 3D RoPE (head 채널을 t/h/w로 나누고 h·w에 NTK 4배),
  adaLN-LoRA 변조 (블록별 저랭크 C→256→3C + 모든 블록이 공유하는 temb), self-attn → 텍스트 cross-attn → GELU FFN, 28블록.

비교 방식: 파이프라인의 한 step처럼 adapter → DiT를 이어 붙인 래퍼를 원본으로 두고,
adapter 출력, adapter 블록, RoPE, patch, 시간 임베딩, 블록별 attn1/attn2/ff/블록 출력, 최종 출력을 층별로 비교합니다.

축소 config
- DiT: hidden 56 (head 2 × 28), 2블록, adaLN-LoRA 8. head_dim 28 → t/h/w 채널 12/8/8로 서로 달라서 축이 뒤바뀌면 드러납니다.
  latent 12×10 (patch 후 6×5, 가로세로가 달라 h/w 뒤바뀜을 잡음), rope_scale (1, 4, 4)는 Anima와 같습니다.
- padding mask는 latent 2배 크기의 무작위 0/1 → nearest resize 경로를 지나갑니다 (파이프라인은 전부 0).
- adapter: model_dim 32 ≠ target_dim 24 → `in_proj` Linear 경로, 2블록, head 4. Qwen 9토큰 / T5 7토큰,
  배치 1번은 Qwen 뒤 3칸, T5 뒤 2칸이 padding. min_sequence_length 12 → 0 행 5개를 붙여 DiT가 mask 없이 읽습니다.
- 자체 점검에서 Cosmos-Predict2 비디오 경로(5D timestep, fps, learnable 위치, cross-attn mask,
  adapter LayerNorm / self-attn 없음 / in_proj 없음)도 원본과 한 번 더 비교합니다.
"""
PIP = ["diffusers==0.41.0"]   # 검토한 태그 v0.41.0 (086bf95). 따옴표 없는 >= 는 셸 리다이렉트가 됨
HEADER = [
    ("circlestone-labs/Anima-Base-v1.0-Diffusers", r"^transformer_blocks\.0\.|^(time_embed|norm_out|patch_embed|proj_out)\.", "transformer"),
    ("circlestone-labs/Anima-Base-v1.0-Diffusers", r"^blocks\.0\.|^(embed|out_proj|norm)\.", "text_conditioner"),
]
CAUSAL = False
SELF_CHECKS = [
    ("""
### attention map · 변조 hook 확인

- `attn_probs`: DiT attn1 = 이미지↔이미지 [B, H, N, N], attn2 = 이미지→텍스트 [B, H, N, 512 자리], adapter = T5↔T5 / T5→Qwen.
  모든 행 합 = 1, adapter cross-attn에서 padding된 Qwen 칸의 확률은 정확히 0이어야 합니다.
- `shift/scale/gate_probe`: 블록마다 adaLN 변조값. 크기를 출력합니다.
""", """
from miniarch.models import anima as A
V.set_deterministic(0)
ref, inputs = S.ref()
m = A.verify_pair(ref)["mine"].to(dev).eval()
x_ = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()}
names = [n for n, _ in m.named_modules() if n.endswith(("attn_probs", "_probe"))]
with V.capture(m, names) as st, torch.no_grad():
    m(**x_)
for n in names:
    if n.endswith("attn_probs"):
        p = st[n]
        err = (p.sum(-1) - 1).abs().max().item()
        print(f"{n:<52} {tuple(p.shape)}  행합 오차={err:.1e}")
        report(f"{n} 행 합 = 1", err < 1e-5, err)
pad_ = ~x_["source_attention_mask"].bool()
for i in range(m.config.adapter.n_layer):
    p = st[f"text_conditioner.blocks.{i}.cross_attn.attn_probs"]
    leak = (p * pad_[:, None, None, :]).abs().max().item()
    report(f"adapter block {i}: padding된 Qwen 칸 확률 = 0", leak == 0.0, leak)
for i in range(m.config.dit.n_layer):
    for j in (1, 2, 3):
        b = f"transformer.transformer_blocks.{i}.norm{j}"
        print(f"{b:<40} |shift|={st[b + '.shift_probe'].abs().mean():.3f} |scale|={st[b + '.scale_probe'].abs().mean():.3f} |gate|={st[b + '.gate_probe'].abs().mean():.3f}")
"""),
    ("""
### 텍스트 토큰 순서 · padding

- DiT cross-attn에는 텍스트 쪽 위치 정보가 없습니다 → 텍스트 토큰 순서를 섞어도 DiT 출력이 같아야 합니다.
- adapter는 Qwen 위치에도 RoPE를 걸기 때문에 Qwen hidden 순서를 바꾸면 출력이 바뀌어야 합니다.
- mask된 Qwen 토큰을 뒤에 더 붙여도, padding된 T5 토큰 id를 바꿔도 adapter의 유효 출력은 같아야 하고 padding 행은 0입니다.
- 파이프라인은 512까지 붙인 0 행을 **mask 없이** DiT에 넣습니다. 0 행은 key = 0 → 점수 0이라 softmax 분모에 들어가므로
  0 행의 개수가 출력을 바꿉니다. bool mask로 가리면 잘라낸 것과 같아집니다.
""", """
from miniarch.models import anima as A
V.set_deterministic(0)
ref, inputs = S.ref()
m = A.verify_pair(ref)["mine"].to(dev).eval()
x_ = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()}
src, ids, sm, tm = x_["source_hidden_states"], x_["target_input_ids"], x_["source_attention_mask"], x_["target_attention_mask"]
with torch.no_grad():
    enc = m.encode_text(src, ids, tm, sm)
    lat, t_, pm_ = x_["hidden_states"], x_["timestep"], x_["padding_mask"]
    out = m.transformer(lat, t_, enc, padding_mask=pm_)
    perm = torch.randperm(enc.shape[1], device=dev)
    d = (m.transformer(lat, t_, enc[:, perm], padding_mask=pm_) - out).abs().max().item()
    report("DiT: 텍스트 토큰 순서를 섞어도 같음", d < 1e-4, d)

    d = (m.encode_text(src.flip(1), ids, None, None) - m.encode_text(src, ids, None, None)).abs().max().item()
    report("adapter: Qwen 순서를 뒤집으면 바뀜 (RoPE)", d > 1e-4, d)

    src2 = torch.cat([src, torch.randn_like(src[:, :4])], 1)
    sm2 = torch.cat([sm, torch.zeros_like(sm[:, :4])], 1)
    d = (m.encode_text(src2, ids, tm, sm2) - enc).abs().max().item()
    report("adapter: mask된 Qwen 토큰을 덧붙여도 같음", d < 1e-5, d)

    ids2 = torch.where(tm.bool(), ids, (ids + 7) % m.config.adapter.vocab_size)
    d = (m.encode_text(src, ids2, tm, sm) - enc).abs().max().item()
    report("adapter: padding된 T5 id 를 바꿔도 같음", d < 1e-5, d)
    Lt = ids.shape[1]
    z = enc[:, :Lt][~tm.bool()].abs().max().item()
    report("adapter: padding 행 = 0", z == 0.0, z)

    out_cut = m.transformer(lat, t_, enc[:, :Lt], padding_mask=pm_)
    d = (out - out_cut).abs().max().item()
    report("DiT: mask 없는 0 행 개수가 출력을 바꿈", d > 1e-5, d)
    am = torch.zeros(enc.shape[:2], dtype=torch.bool, device=dev)
    am[:, :Lt] = True
    d = (m.transformer(lat, t_, enc, attention_mask=am, padding_mask=pm_) - out_cut).abs().max().item()
    report("DiT: bool mask 로 가리면 잘라낸 것과 같음", d < 1e-5, d)
"""),
    ("""
### 시간 임베딩 · CFG null 텍스트

- t가 바뀌면 변조값(shift/scale/gate)과 출력이 바뀌어야 합니다. t에 따른 0번 블록 gate 크기를 출력합니다.
- CFG의 빈 negative prompt: Qwen 토큰이 0개라 파이프라인은 id 0 하나에 mask 0을 넣습니다 → adapter cross-attn 행이 전부 가려집니다.
  내 구현은 이 행을 0으로 둡니다 (torch SDPA math 경로와 같음). 출력이 유한하고 guidance 결합이 되는지 봅니다.
  원본 결과는 SDPA 백엔드에 따라 달라질 수 있어 참고로만 출력합니다.
""", """
from miniarch.models import anima as A
V.set_deterministic(0)
ref, inputs = S.ref()
pair = A.verify_pair(ref)
m = pair["mine"]
m.load_state_dict(ref.state_dict())
m, ref = m.to(dev).eval(), ref.to(dev).eval()
x_ = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()}
g_ = "transformer.transformer_blocks.0.norm1.gate_probe"
outs = []
for tv in (0.0, 0.25, 0.5, 0.75, 1.0):
    x2 = dict(x_, timestep=torch.full_like(x_["timestep"], tv))
    with V.capture(m, [g_]) as st, torch.no_grad():
        outs.append(m(**x2))
    print(f"t={tv:.2f}  block0 |gate|={st[g_].abs().mean():.4f}  |out|={outs[-1].abs().mean():.4f}")
d = (outs[1] - outs[2]).abs().max().item()
report("timestep 이 바뀌면 출력이 바뀜", d > 1e-4, d)

B_ = x_["hidden_states"].shape[0]
neg = dict(x_, source_hidden_states=torch.zeros(B_, 1, x_["source_hidden_states"].shape[-1], device=dev),
           source_attention_mask=torch.zeros(B_, 1, dtype=torch.long, device=dev))
with torch.no_grad():
    cond, unc = m(**x_), m(**neg)
    pred = unc + 4.0 * (cond - unc)
    ok = bool(torch.isfinite(unc).all() and torch.isfinite(pred).all())
    report("CFG: 빈 prompt 경로 유한", ok, (cond - unc).abs().max().item())
    try:
        r_unc = ref(**neg)[0]
        print("참고: 원본 빈 prompt 출력 유한 =", bool(torch.isfinite(r_unc).all()), " 내 구현과 차이 =", (r_unc - unc).abs().max().item())
    except Exception as e:
        print("참고: 원본 빈 prompt 실행 실패", type(e).__name__, e)
"""),
    ("""
### Cosmos-Predict2 비디오 경로도 원본과 동치

Anima는 쓰지 않지만 같은 DiT 코드에 있는 경로입니다: 프레임별 timestep [B, 1, T, 1, 1], fps로 늘린 시간축 RoPE,
블록마다 더하는 learnable 절대 위치, 텍스트 bool mask, rope_scale (2, 1, 1).
adapter도 LayerNorm · self-attn 없음 · in_proj 없음 · 512 패딩 없음 설정으로 바꿔 비교합니다.
""", """
from miniarch.models import anima as A
V.set_deterministic(0)
ref_v, inp_v = S.ref("video")
V.perturb_params(ref_v)
pair = A.verify_pair(ref_v)
mine_v = pair.pop("mine")
rep = V.check_equivalence(ref_v, mine_v, inp_v, atol=1e-4, device=dev, **pair)
print(rep.summary())
report("Predict2 비디오 경로 동치", rep.ok, max(r.max_abs for r in rep.results))
"""),
    ("""
### ablation 스위치

기본값은 원본과 같고, 하나씩 끄면 출력이 바뀌어야 합니다 (같은 가중치).
- DiT: `use_rope`(3D RoPE), `use_qk_norm`, `use_adaln_lora`(공유 temb 더하기), `use_cross_attn`(텍스트 읽기)
- adapter: `use_rope`
""", """
import dataclasses
from miniarch.models import anima as A
V.set_deterministic(0)
ref, inputs = S.ref()
m = A.verify_pair(ref)["mine"].to(dev).eval()
x_ = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()}
with torch.no_grad():
    base = m(**x_)
sw = [("dit", k) for k in ("use_rope", "use_qk_norm", "use_adaln_lora", "use_cross_attn")] + [("adapter", "use_rope")]
for part, k in sw:
    c2 = dataclasses.replace(m.config, **{part: dataclasses.replace(getattr(m.config, part), **{k: False})})
    m2 = A.Anima(c2).to(dev).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        d = (m2(**x_) - base).abs().max().item()
    report(f"{part}.{k}=False 이면 출력이 바뀜", d > 1e-5, d)
"""),
]


class AnimaPipelineRef(nn.Module):
    """파이프라인 한 step: AnimaTextConditioner → CosmosTransformer3DModel (원본 모듈을 그대로 호출만)."""

    def __init__(self, text_conditioner, transformer):
        super().__init__()
        self.text_conditioner = text_conditioner
        self.transformer = transformer

    def forward(self, hidden_states, timestep, source_hidden_states, target_input_ids,
                target_attention_mask=None, source_attention_mask=None, padding_mask=None,
                attention_mask=None, fps=None):
        enc = self.text_conditioner(source_hidden_states=source_hidden_states, target_input_ids=target_input_ids,
                                    target_attention_mask=target_attention_mask,
                                    source_attention_mask=source_attention_mask)
        return self.transformer(hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=enc,
                                attention_mask=attention_mask, fps=fps, padding_mask=padding_mask, return_dict=False)


def ref(variant: str = "anima"):
    """variant="anima": Anima 설정 축소판. "video": Cosmos-Predict2 비디오 경로 (자체 점검용)."""
    from diffusers import AnimaTextConditioner, CosmosTransformer3DModel
    video = variant == "video"
    tc = AnimaTextConditioner(
        source_dim=20, target_dim=24, model_dim=24 if video else 32, num_layers=2, num_attention_heads=4,
        mlp_ratio=2.0 if video else 4.0, target_vocab_size=64, use_self_attention=not video,
        use_layer_norm=video, min_sequence_length=4 if video else 12)
    tr = CosmosTransformer3DModel(
        in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=28, num_layers=2,
        mlp_ratio=4.0, text_embed_dim=24, adaln_lora_dim=8, max_size=(4, 32, 32), patch_size=(1, 2, 2),
        rope_scale=(2.0, 1.0, 1.0) if video else (1.0, 4.0, 4.0), concat_padding_mask=True,
        extra_pos_embed_type="learnable" if video else None)
    g = torch.Generator().manual_seed(1)
    if video:
        with torch.no_grad():  # learnable 위치는 0 초기화 → 0 이 아닌 값으로
            pe = tr.learnable_pos_embed
            for p in (pe.pos_emb_t, pe.pos_emb_h, pe.pos_emb_w):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    m = AnimaPipelineRef(tc, tr)

    g = torch.Generator().manual_seed(0)
    B, Ls, Lt = BATCH, 9, 7
    src = torch.randn(B, Ls, 20, generator=g)
    ids = torch.randint(0, 64, (B, Lt), generator=g)
    if video:
        am = torch.ones(B, Lt, dtype=torch.bool)
        am[1, -3:] = False
        return m, {
            "hidden_states": torch.randn(B, 4, 3, 8, 6, generator=g),
            "timestep": torch.rand(B, 1, 3, 1, 1, generator=g),
            "source_hidden_states": src, "target_input_ids": ids,
            "padding_mask": (torch.rand(1, 1, 8, 6, generator=g) > 0.7).float(),
            "attention_mask": am, "fps": 16,
        }
    sm = torch.ones(B, Ls, dtype=torch.long)
    sm[1, -3:] = 0
    tm = torch.ones(B, Lt, dtype=torch.long)
    tm[1, -2:] = 0
    return m, {
        "hidden_states": torch.randn(B, 4, 1, 12, 10, generator=g),
        "timestep": torch.tensor([0.3, 0.85]),                 # 파이프라인: t / 1000 (flow sigma)
        "source_hidden_states": src * sm[..., None],             # 파이프라인처럼 padding 위치 0
        "target_input_ids": ids,
        "target_attention_mask": tm,
        "source_attention_mask": sm,
        "padding_mask": (torch.rand(1, 1, 24, 20, generator=g) > 0.7).float(),
    }


CASE = Case(STAGE, "Cosmos-Predict2 DiT (3D RoPE · adaLN-LoRA · cross-attn) + LLM adapter (Qwen3 → T5 질의)",
            "Anima", "diffusers transformer_cosmos.py · condition_embedder_anima.py", ref,
            "miniarch.models.anima", atol=1e-4)
