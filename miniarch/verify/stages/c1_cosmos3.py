"""C1 · Cosmos3 (Omni): Mixture-of-Transformers (reasoner AR 탑 + generator 확산 탑) 검증 케이스.

원본 = diffusers `Cosmos3OmniTransformer` (src/diffusers/models/transformers/transformer_cosmos3.py,
main 커밋 c60830e = v0.41.0 과 같은 파일). 원본은 batch 없이 한 줄로 이어 붙인(packed) 입력과
모달리티별 리스트를 받고 리스트를 돌려주므로, forward 만 감싼다 (가중치·수식은 그대로):
  - 입력 리스트 안의 텐서까지 모델 device 로 옮김 (노트북 smoke 셀은 dict 최상위 텐서만 옮김)
  - 반환 = (모든 예측을 이어 붙인 한 줄 텐서, 원래 출력)  → smoke 셀·비교가 out[0] 을 씀
"""
from __future__ import annotations

import torch

from ..cases import VOCAB, Case

STAGE = "C1"
SLUG = "cosmos3"
TITLE = "Cosmos3 (MoT: reasoner + 확산 generator)"
INTRO = """
원본: diffusers `Cosmos3OmniTransformer` (`src/diffusers/models/transformers/transformer_cosmos3.py`,
main 커밋 `c60830e` 고정 = PyPI `diffusers 0.41.0` 과 같은 파일). 내 구현: `miniarch/models/cosmos3.py`.
가중치 이름은 실제 체크포인트(`nvidia/Cosmos3-*/transformer`)와 같습니다.

Cosmos3는 **층마다 파라미터가 두 벌**인 Mixture-of-Transformers입니다.
- reasoner(이해) 탑: 텍스트 토큰, **causal** attention → 그대로 LLM (lm_head 로 다음 토큰 예측)
- generator(확산) 탑: 영상·소리·행동 latent 토큰, query는 generator, key/value는 **reasoner + generator 전부(양방향)**
- 두 탑은 attention에서만 만나고 정보는 reasoner → generator 한 방향으로만 흐릅니다.
- 위치: interleaved 3D M-RoPE (t, h, w), head_dim 전체 회전. 시간 조건은 noisy 프레임 토큰에 timestep 임베딩을 **더하기만** 함
  (조건 프레임 = 깨끗한 토큰 → 이미지→영상 / 편집 / 행동 조건 생성이 같은 경로)
- 행동: embodiment(domain)마다 가중치가 다른 DomainAwareLinear 입출력

축소 config (모든 경로가 한 번씩 쓰이도록)
- 2층, d=64, head 4 / KV head 2 (GQA), head_dim 16, M-RoPE 채널 [4, 2, 2] (실제 [24, 20, 20] 과 같은 interleave 규칙)
- 텍스트 7 + 영상 15 + 소리 4 + 행동 5 = 31 토큰. 위치는 파이프라인처럼 실수(fps 보정) t 와 (h, w)
- 영상 2개: ① T=2, 3×5 latent (patch 2 의 배수가 아님 → 0 padding 경로), 0번 프레임은 조건(깨끗), 1번만 noisy
  ② T=3, 2×2 latent, 0·2번 프레임만 noisy (띄엄띄엄). 토큰마다 다른 timestep
- 소리 4프레임(전부 noisy), 행동 5스텝(0번은 조건) + domain 2 (domain 3개 중)
- 원본이 0 으로 만드는 modality embed, N(0,1) 인 행동 domain 표는 스케일을 맞춘 랜덤 값으로 채웁니다.
- Edge(Nemotron: relu2 MLP, reasoner QK-norm 없음 + generator 용 reasoner key norm) 변형은 자체 점검에서 한 번 더 비교합니다.

출력 비교는 영상·소리·행동 예측을 이어 붙인 한 줄 텐서, 층별 비교는 reasoner 층 출력과 두 탑의 attention 출력·generator MLP 입니다.
"""
PIP = ["diffusers==0.41.0"]
HEADER = [("nvidia/Cosmos3-Nano", r"^layers\.0\.|^(action|audio)_", "transformer"),
          ("nvidia/Cosmos3-Edge", r"^layers\.0\.self_attn\.", "transformer")]
CAUSAL = False
SELF_CHECKS = [
    ("""
### 한 방향 정보 흐름 · reasoner causal

MoT의 핵심 성질을 내 구현으로 직접 확인합니다.
1. 생성 토큰(영상·소리·행동 latent, timestep)을 바꿔도 reasoner 출력은 **그대로**, generator 출력은 바뀜
2. 뒤쪽 텍스트 토큰을 바꾸면 그 앞 reasoner 출력은 그대로 (causal), 뒤는 바뀌고, generator 출력도 바뀜 (generator 는 reasoner 를 봄)
3. ablation: `gen_sees_und=False` 이면 텍스트를 바꿔도 generator 불변, `und_sees_gen=True` 이면 생성 토큰이 reasoner 를 바꿈
""", """
import dataclasses
from miniarch.models import cosmos3 as C
ref, inp = S.ref()
m = C.verify_pair(ref)["mine"].to(dev).eval()

def run(x, model=m):
    with torch.no_grad():
        o = model(**S.to_device(x, dev))
    return o.und_hidden, o.gen_hidden

def d(a, b):
    return (a - b).abs().max().item()

g_ = torch.Generator().manual_seed(7)
x_gen = dict(inp)
for k in ("vision_tokens", "sound_tokens", "action_tokens"):
    x_gen[k] = [t + torch.randn(t.shape, generator=g_) for t in inp[k]]
x_gen["vision_timesteps"] = inp["vision_timesteps"].flip(0)
x_txt = dict(inp)
cut = 4
ids = inp["input_ids"].clone(); ids[cut:] = (ids[cut:] + 1) % V.cases.VOCAB
x_txt["input_ids"] = ids

u0, g0 = run(inp)
u1, g1 = run(x_gen)
report("생성 토큰 변경 → reasoner 불변", d(u0, u1) < 1e-6, d(u0, u1))
report("생성 토큰 변경 → generator 변함", d(g0, g1) > 1e-3, d(g0, g1))
u2, g2 = run(x_txt)
report(f"텍스트 {cut}번 이후 변경 → 앞 reasoner 불변", d(u0[:cut], u2[:cut]) < 1e-6, d(u0[:cut], u2[:cut]))
report(f"텍스트 {cut}번 이후 변경 → 뒤 reasoner 변함", d(u0[cut:], u2[cut:]) > 1e-3, d(u0[cut:], u2[cut:]))
report("텍스트 변경 → generator 변함 (reasoner 를 봄)", d(g0, g2) > 1e-3, d(g0, g2))

def variant(**sw):
    m2 = C.Cosmos3(dataclasses.replace(m.config, **sw)).to(dev).eval()
    m2.load_state_dict(m.state_dict())
    return m2
m_a = variant(gen_sees_und=False)
_, ga0 = run(inp, m_a); _, ga1 = run(x_txt, m_a)
report("gen_sees_und=False → 텍스트 바꿔도 generator 불변", d(ga0, ga1) < 1e-6, d(ga0, ga1))
m_b = variant(und_sees_gen=True)
ub0, _ = run(inp, m_b); ub1, _ = run(x_gen, m_b)
report("und_sees_gen=True → 생성 토큰이 reasoner 를 바꿈", d(ub0, ub1) > 1e-3, d(ub0, ub1))
"""),
    ("""
### attention map 확인

- `und_probs` [H, N_und, N_und]: 대각선 위(미래)는 정확히 0, 행 합 1
- `gen_probs` [H, N_gen, N_und + N_gen]: 앞쪽 N_und 칸이 reasoner 몫. generator 질의가 텍스트에 주는 평균 확률을 출력합니다.
""", """
from miniarch.models import cosmos3 as C
ref, inp = S.ref()
m = C.verify_pair(ref)["mine"].to(dev).eval()
names = [n for n, _ in m.named_modules() if n.endswith(("und_probs", "gen_probs"))]
with V.capture(m, names) as st, torch.no_grad():
    m(**S.to_device(inp, dev))
nu = inp["und_len"]
for i in range(m.config.n_layer):
    pu, pg = st[f"layers.{i}.self_attn.und_probs"], st[f"layers.{i}.self_attn.gen_probs"]
    fut = pu.triu(1).abs().max().item()
    err = max((pu.sum(-1) - 1).abs().max().item(), (pg.sum(-1) - 1).abs().max().item())
    print(f"layer {i} und={tuple(pu.shape)} gen={tuple(pg.shape)}  generator→텍스트 평균 확률 {pg[..., :nu].sum(-1).mean():.3f}")
    report(f"layer {i} reasoner 미래 확률 = 0", fut == 0, fut)
    report(f"layer {i} 확률 행 합 = 1", err < 1e-5, err)
"""),
    ("""
### Edge 변형 (Nemotron 백본) 원본 비교

Cosmos3-Edge 는 `hidden_act="relu2"` (gate 없는 relu² MLP, Nemotron RMSNorm), reasoner QK-norm 없음,
generator 가 보는 reasoner key 에만 별도 norm(`k_norm_und_for_gen`)을 씁니다. 소리 없이 같은 방식으로 원본과 비교합니다.
""", """
from miniarch.models import cosmos3 as C
V.set_deterministic(0)
r2, x2 = S.ref(edge=True)
V.perturb_params(r2)
pair = C.verify_pair(r2)
mine2 = pair.pop("mine")
rep = V.check_equivalence(r2, mine2, x2, atol=1e-4, device=dev, **pair)
print(rep.summary())
report("Edge 변형 원본 동치", rep.ok, max(r.max_abs for r in rep.results))
"""),
    ("""
### ablation 스위치

부품별 비교 실험에 쓸 스위치를 하나씩 바꾸고 forward 가 정상(유한값)이고 출력이 실제로 바뀌는지 확인합니다.
""", """
import dataclasses
from miniarch.models import cosmos3 as C
ref, inp = S.ref()
m = C.verify_pair(ref)["mine"].to(dev).eval()
x = S.to_device(inp, dev)
with torch.no_grad():
    base = m(**x).flat()
for sw, val in [("und_causal", False), ("gen_sees_und", False), ("und_sees_gen", True), ("share_towers", True),
                ("use_timestep", False), ("mrope_interleaved", False)]:
    m2 = C.Cosmos3(dataclasses.replace(m.config, **{sw: val})).to(dev).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        out = m2(**x).flat()
    dd = (out - base).abs().max().item()
    report(f"{sw}={val} 유한 · 출력 변함", bool(torch.isfinite(out).all()) and dd > 1e-4, dd)
"""),
]

DIFFUSERS_COMMIT = "c60830ee365d520ab52b110dda562dd26f7b4d7f"   # transformer_cosmos3.py 마지막 변경 (v0.41.0 에 포함)


def to_device(x, dev):
    """dict / list / tuple 안의 텐서까지 device 로."""
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, dict):
        return {k: to_device(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, dev) for v in x)
    return x


def _wrap_forward(m):
    """입력 리스트 텐서를 모델 device 로 옮기고, (예측 한 줄 텐서, 원래 출력) 을 돌려주도록 forward 만 감싼다."""
    from miniarch.models.cosmos3 import flat_preds
    orig = m.forward

    def forward(**kw):
        out = orig(**to_device(kw, m.embed_tokens.weight.device))
        return flat_preds(out.sample, out.sound, out.action), out

    m.forward = forward


def _inputs(sound: bool):
    g = torch.Generator().manual_seed(0)
    und = 7
    vis = [torch.randn(1, 4, 2, 3, 5, generator=g), torch.randn(1, 4, 3, 2, 2, generator=g)]
    vis_shapes = [(2, 2, 3), (3, 1, 1)]                    # patch grid (T, ceil(H/2), ceil(W/2))
    vis_noisy = [torch.tensor([1]), torch.tensor([0, 2])]
    v0 = und
    vis_idx = torch.arange(v0, v0 + 15)
    vis_mse = torch.tensor(list(range(v0 + 6, v0 + 12)) + [v0 + 12, v0 + 14])
    cur = v0 + 15
    # 위치: 텍스트 = 세 축 같은 번호, latent = (fps 보정 실수 t, h, w), 모달리티 사이 margin 5
    t_off = und + 5
    pos = [torch.arange(und, dtype=torch.float32).expand(3, -1)]
    for (T, Hp, Wp), dt, off in zip(vis_shapes, (1.5, 2.0), (t_off, t_off + 4)):
        t = (torch.arange(T, dtype=torch.float32) * dt + off).view(-1, 1).expand(-1, Hp * Wp).flatten()
        hh = torch.arange(Hp, dtype=torch.float32).view(1, -1, 1).expand(T, -1, Wp).flatten()
        ww = torch.arange(Wp, dtype=torch.float32).view(1, 1, -1).expand(T, Hp, -1).flatten()
        pos.append(torch.stack([t, hh, ww]))
    x = {
        "input_ids": torch.randint(0, VOCAB, (und,), generator=g),
        "text_indexes": torch.arange(und),
        "und_len": und,
        "vision_tokens": vis,
        "vision_token_shapes": vis_shapes,
        "vision_sequence_indexes": vis_idx,
        "vision_mse_loss_indexes": vis_mse,
        "vision_timesteps": torch.rand(8, generator=g) * 1000,
        "vision_noisy_frame_indexes": vis_noisy,
    }
    if sound:
        x.update(sound_tokens=[torch.randn(5, 4, generator=g)], sound_token_shapes=[(4, 1, 1)],
                 sound_sequence_indexes=torch.arange(cur, cur + 4), sound_mse_loss_indexes=torch.arange(cur, cur + 4),
                 sound_timesteps=torch.rand(4, generator=g) * 1000, sound_noisy_frame_indexes=[torch.arange(4)])
        t = torch.arange(4, dtype=torch.float32) * 0.75 + t_off
        pos.append(torch.stack([t, torch.zeros(4), torch.zeros(4)]))
        cur += 4
    a_idx = torch.arange(cur, cur + 5)
    a_noisy = torch.tensor([1, 2, 3, 4])
    x.update(action_tokens=[torch.randn(5, 6, generator=g)], action_token_shapes=[(5, 1, 1)],
             action_sequence_indexes=a_idx, action_mse_loss_indexes=a_idx[a_noisy],
             action_timesteps=torch.rand(4, generator=g) * 1000, action_noisy_frame_indexes=[a_noisy],
             action_domain_ids=[torch.tensor(2)])
    t = (torch.arange(5, dtype=torch.float32) + 1) * 0.5 + t_off
    pos.append(torch.stack([t, torch.zeros(5), torch.zeros(5)]))
    cur += 5
    x["position_ids"] = torch.cat(pos, dim=1)               # [3, N]
    x["sequence_length"] = cur
    return x


def ref(edge: bool = False):
    import diffusers
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
    if not diffusers.__version__.startswith("0.41."):
        print(f"⚠️ diffusers {diffusers.__version__}: 고정한 원본(0.41.0 = {DIFFUSERS_COMMIT[:7]})과 버전이 다름")
    kw = dict(head_dim=16, hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
              num_hidden_layers=2, latent_channel=4, latent_patch_size=2, patch_latent_dim=16, rms_norm_eps=1e-6,
              rope_theta=10000.0, rope_axes_dim=[4, 2, 2], vocab_size=VOCAB, timestep_scale=0.001,
              action_gen=True, action_dim=6, num_embodiment_domains=3, sound_gen=True, sound_dim=5)
    if edge:
        kw.update(hidden_act="relu2", qk_norm_for_text=False, use_und_k_norm_for_gen=True, rms_norm_eps=1e-5,
                  rope_theta=1e6, sound_gen=False, sound_dim=None)
    m = Cosmos3OmniTransformer(**kw)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():   # 0 초기화(modality embed)·N(0,1)(domain 표) → 스케일 맞춘 랜덤 값
        for name, p in m.named_parameters():
            r = torch.randn(p.shape, generator=g)
            if name.endswith("modality_embed"):
                p.copy_(r * 0.5)
            elif name == "action_proj_in.fc.weight":
                p.copy_(r * kw["action_dim"] ** -0.5)
            elif name == "action_proj_out.fc.weight":
                p.copy_(r * kw["hidden_size"] ** -0.5)
            elif name.startswith("action_proj") and name.endswith("bias.weight"):
                p.copy_(r * 0.1)
            elif "norm" in name:
                p.copy_(1 + 0.1 * r)
    _wrap_forward(m)
    return m, _inputs(sound=not edge)


CASE = Case(STAGE, "Cosmos3 MoT: causal reasoner 탑 + 양방향 generator 탑(joint attention, 한 방향) + M-RoPE + 영상·소리·행동 입출력",
            "Cosmos3-Nano", f"diffusers transformer_cosmos3.py@{DIFFUSERS_COMMIT[:7]}", ref, "miniarch.models.cosmos3",
            atol=1e-4)
