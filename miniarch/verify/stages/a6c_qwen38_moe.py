"""A6c · Qwen3.8 플래그십 (2.4T-A95B): Qwen3.6 하이브리드 + MoE 검증 케이스."""
from __future__ import annotations

import torch

from ..cases import BATCH, VOCAB, Case

STAGE = "A6c"
SLUG = "qwen38_moe"
TITLE = "Qwen3.8 플래그십 2.4T-A95B (하이브리드 + MoE)"
INTRO = """
원본: transformers `qwen3_5_moe` (Qwen3.8-2.4T-A95B, 92층 / expert 512개 중 10개 + shared expert).
내 구현: `miniarch/models/qwen36.py` 에 `n_experts > 0` 으로 MoE 를 켠 것.

플래그십은 A6a(Qwen3.6 / 3.8-27B)와 attention 부분이 완전히 같고 **MLP 만 MoE 로 바뀝니다**.
(HC, sparse indexer, n-gram 임베딩은 실험 계열 Qwen3.8-Flash-Next 에만 있고 플래그십에는 없습니다 → A7a.)

- MoE: softmax 점수 → top-k → 고른 점수끼리 재정규화, + sigmoid 게이트를 곱한 shared expert
- 축소 config: 4층(선형 3 + full 1), expert 8개 중 3개, 시퀀스 100 (chunk 2개 + padding 경로)
"""
PIP = []
HEADER = [("Qwen/Qwen3.8-2.4T-A95B", r"layers\.(0|3)\.", "")]
CAUSAL = True
SELF_CHECKS = [
    ("""
### 라우팅 확인

- 토큰마다 정확히 top-k 개의 서로 다른 expert 를 고르는지, expert 별 부하가 어떻게 나뉘는지 봅니다.
- shared expert 를 끈 ablation(`use_shared_expert=False`)이 실제로 출력을 바꾸는지 확인합니다.
""", """
import dataclasses
from miniarch.models import qwen36 as Q
ref, inputs = S.ref()
m = Q.verify_pair(ref)["mine"].to(dev).eval()
x = inputs["input_ids"].to(dev)
names = [n for n, _ in m.named_modules() if n.endswith("route_probe")]
with V.capture(m, names) as st, torch.no_grad():
    h = m(x).last_hidden_state
for n in names:
    idx = st[n]
    distinct = all(len(set(r.tolist())) == idx.shape[1] for r in idx)
    load = torch.bincount(idx.flatten(), minlength=m.config.n_experts).tolist()
    print(f"{n:<28} 선택 {tuple(idx.shape)}  expert 부하={load}")
    report(f"{n} top-k 서로 다름", distinct, 0.0)

m2 = Q.Qwen36(dataclasses.replace(m.config, use_shared_expert=False)).to(dev).eval()
m2.load_state_dict(m.state_dict())
with torch.no_grad():
    d = (m2(x).last_hidden_state - h).abs().max().item()
report("shared expert 끄면 출력이 바뀜", d > 1e-4, d)
"""),
]


def ref():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=VOCAB, hidden_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, max_position_embeddings=256, use_cache=False,
        num_experts=8, num_experts_per_tok=3, moe_intermediate_size=32, shared_expert_intermediate_size=48,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0]})
    m = Qwen3_5MoeTextModel(cfg)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # router 는 0 초기화, expert 는 empty → 명시적으로 채워 라우팅이 실제로 갈리게 한다
        for layer in m.layers:
            moe = layer.mlp
            moe.gate.weight.copy_(torch.randn(moe.gate.weight.shape, generator=g) * 0.5)
            moe.experts.gate_up_proj.copy_(torch.randn(moe.experts.gate_up_proj.shape, generator=g) * 0.1)
            moe.experts.down_proj.copy_(torch.randn(moe.experts.down_proj.shape, generator=g) * 0.1)
    ids = torch.randint(0, VOCAB, (BATCH, 100), generator=torch.Generator().manual_seed(0))
    return m, {"input_ids": ids}


CASE = Case(STAGE, "Gated DeltaNet + Gated Attention + softmax top-k MoE (shared expert)", "Qwen3.8-2.4T-A95B",
            "transformers/models/qwen3_5_moe", ref, "miniarch.models.qwen36", atol=1e-4)
