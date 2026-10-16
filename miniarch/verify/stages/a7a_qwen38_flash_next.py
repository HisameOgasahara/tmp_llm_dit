"""A7a · Qwen3.8-Flash-Next (Hyper-Connections + QSA indexer + n-gram PLE + MoE) 검증 케이스."""
from __future__ import annotations

import torch

from ..cases import BATCH, VOCAB, Case

STAGE = "A7a"
SLUG = "qwen38_flash_next"
TITLE = "Qwen3.8-Flash-Next (HC + QSA indexer + n-gram PLE + MoE)"
INTRO = """
원본: transformers `qwen4_exp` (`Qwen4ExpTextModel`, 텍스트 부분만). 내 구현: `miniarch/models/qwen38_flash_next.py`.
Gated DeltaNet과 partial RoPE는 A6a(`qwen36.py`) 부품을 그대로 씁니다.

부품
- Qwen3.6 하이브리드 뼈대: Gated DeltaNet 3층 : Gated Attention 1층 (DeltaNet 출력 게이트는 sigmoid)
- Hyper-Connections: residual을 줄기 4개로 들고 다니며, 층 앞에서 저랭크 sigmoid 가중치로 줄기를 합쳐 입력을 만들고 출력은 줄기마다 다른 세기로 주입합니다. 마지막에 같은 방식으로 하나로 합칩니다 (final norm 없음).
- QSA indexer: 지난 토큰을 4개씩 묶어 평균 key로 요약하고, 점수 상위 묶음과 아직 덜 찬 꼬리 묶음의 토큰만 attention합니다.
- PLE: 2-gram, 3-gram을 해시해서 표에서 임베딩을 꺼내 줄기별 게이트로 더합니다. EOS에서 n-gram 문맥이 끊깁니다.
- MoE: softmax top-k 라우터 + sigmoid 게이트 shared expert

축소 config: 4층 = [선형, 선형(PLE), 선형(PLE), indexed], hidden 64, 줄기 4개, 시퀀스 80.
- 시퀀스 80 → DeltaNet chunk(64) 2개 + padding 경로
- indexer 묶음 4토큰, budget 8(묶음 2개) → 묶음 0개, 2개 이하(전부 선택), 2개 초과(top-k 선택), 꼬리 묶음 경로가 모두 쓰입니다.
- PLE 층 2개 → 층마다 다른 해시 표 크기와 곱셈 상수. 입력에 EOS를 넣어(연속 EOS 포함) n-gram 끊김 경로까지 검사합니다.
"""
PIP = []
HEADER = [("Qwen/Qwen3.8-Flash-Next", r"layers\.(1|3)\.(?!.*ngram_embedding)", "")]
CAUSAL = True
SELF_CHECKS = [
    ("""
### QSA indexer 선택 · attention map 확인

- `select_probe`: 질의마다 허용된 토큰. 개수 = min(완성 묶음 수, 2)·4 + 꼬리 길이여야 하고, 미래 토큰은 없어야 합니다.
- `attn_probs`: 허용되지 않은 칸의 확률은 정확히 0, 행 합은 1이어야 합니다.
""", """
from miniarch.models import qwen38_flash_next as M
ref, inp = S.ref()
m = M.verify_pair(ref)["mine"].to(dev).eval()
L = m.config.layer_types.index(M.INDEXED)
pre = f"layers.{L}.self_attn"
x = inp["input_ids"].to(dev)
with V.capture(m, [pre + ".indexer.select_probe", pre + ".attn_probs"]) as st, torch.no_grad():
    m(x)
sel, p = st[pre + ".indexer.select_probe"] > 0.5, st[pre + ".attn_probs"]
T, r, k = x.shape[1], m.config.index_ratio, m.config.index_budget // m.config.index_ratio
t = torch.arange(1, T + 1, device=dev)
expect = torch.minimum(t // r, torch.tensor(k, device=dev)) * r + t % r
cnt_ok = bool((sel.sum(-1) == expect).all())
future = bool(sel.triu(1).any())
outside = p.masked_fill(sel[:, None], 0).abs().max().item()
rowerr = (p.sum(-1) - 1).abs().max().item()
print(f"layer {L}: 선택 개수 일치={cnt_ok}  미래 토큰 선택={future}  선택 밖 확률 최대={outside:.1e}  행합 오차={rowerr:.1e}")
print("질의별 선택 수 (앞 24개):", sel[0].sum(-1)[:24].tolist())
report("indexer 선택 개수 / 인과성", cnt_ok and not future)
report("attention: 선택 밖 확률 0, 행합 1", outside == 0 and rowerr < 1e-5, max(outside, rowerr))
"""),
    ("""
### PLE: EOS 에서 n-gram 문맥이 끊기는지

EOS 앞쪽 토큰을 바꿔도 EOS 다음 위치들의 n-gram 임베딩은 그대로여야 합니다 (문서 경계 너머를 섞지 않음).
""", """
from miniarch.models import qwen38_flash_next as M
ref, _ = S.ref()
m = M.verify_pair(ref)["mine"].to(dev).eval()
eos = m.config.eos_token_id
ple = next(l.ple for l in m.layers if l.ple is not None)
g = torch.Generator().manual_seed(5)
x = torch.randint(2, V.cases.VOCAB, (2, 40), generator=g)
x[:, 20] = eos
x2 = x.clone()
x2[:, :20] = torch.randint(2, V.cases.VOCAB, (2, 20), generator=g)
with torch.no_grad():
    a, b = ple.ple_embedding(x.to(dev)), ple.ple_embedding(x2.to(dev))
after = (a[:, 21:] - b[:, 21:]).abs().max().item()
before = (a[:, :20] - b[:, :20]).abs().max().item()
print(f"EOS 뒤 차이={after:.1e} (0 이어야 함)   EOS 앞 차이={before:.1e} (0 이 아니어야 함)")
report("PLE n-gram EOS 경계", after == 0 and before > 0, after)
"""),
    ("""
### ablation 스위치 · HC hook 확인

스위치를 하나씩 끈 모델이 돌아가고 causal 을 지키는지, 그리고 원본 설정에서 HC 주입 세기(0~2)와 입력 가중치(0~1) 범위를 봅니다.
""", """
import dataclasses
from miniarch.models import qwen38_flash_next as M
ref, inp = S.ref()
base = M.config_from_hf(ref.config)
for sw in ["use_hc", "use_indexer", "use_ple", "use_moe"]:
    V.set_deterministic(0)
    mm = M.Qwen38FlashNext(dataclasses.replace(base, **{sw: False})).to(dev).eval()
    r = V.causal_leak_test(lambda x: mm(x).last_hidden_state, vocab_size=V.cases.VOCAB, seq_len=32, device=dev, atol=1e-4)
    with torch.no_grad():
        ok = bool(torch.isfinite(mm(inp["input_ids"].to(dev)).logits).all())
    report(f"{sw}=False 동작 + causal", r.ok and ok, r.max_abs)

m = M.verify_pair(ref)["mine"].to(dev).eval()
names = [n for n, _ in m.named_modules() if n.endswith(("inject_probe", "mix_probe"))]
with V.capture(m, names) as st, torch.no_grad():
    m(inp["input_ids"].to(dev))
for i in range(m.config.n_layer):
    inj = st[f"layers.{i}.attn_hyper_connection.inject_probe"]
    mix = st[f"layers.{i}.attn_hyper_connection.mix_probe"]
    print(f"layer {i} {m.config.layer_types[i]:<18} 주입 세기 평균(줄기별)={[round(v, 3) for v in inj.mean((0, 1)).tolist()]}  입력 가중치 평균={mix.mean():.3f}")
"""),
]


def ref():
    """선형 3층(그중 2층 PLE) + indexed 1층, 시퀀스 80."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel
    cfg = Qwen4ExpTextConfig(
        vocab_size=VOCAB, hidden_size=64, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "indexed_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, output_gate_type="sigmoid",
        moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=8, num_experts_per_tok=2,
        hc_count=4, hc_lowrank=16,
        ple_layer_ids=[2, 3], ple_embed_dim=32, ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=50, make_ngram_vocab_size_divisible_by=128,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=8, indexer_compress_ratio=4,
        max_position_embeddings=256, use_cache=False, pad_token_id=None, bos_token_id=None, eos_token_id=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})
    ids = torch.randint(2, VOCAB, (BATCH, 80), generator=torch.Generator().manual_seed(0))
    ids[0, 10] = ids[0, 11] = 1     # EOS (연속 EOS 포함) → n-gram 끊김 경로
    ids[1, 37] = 1
    return Qwen4ExpTextModel(cfg), {"input_ids": ids}


CASE = Case(STAGE, "Qwen3.8-Flash-Next: Gated DeltaNet + QSA indexed attention + Hyper-Connections + n-gram PLE + MoE",
            "Qwen3.8-Flash-Next", "transformers/models/qwen4_exp", ref, "miniarch.models.qwen38_flash_next",
            atol=1e-4)
