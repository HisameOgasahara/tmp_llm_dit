"""A6a · Qwen3.6 하이브리드 (Gated DeltaNet + Gated Attention) 검증 케이스."""
from __future__ import annotations

import torch

from ..cases import BATCH, VOCAB, Case

STAGE = "A6a"
SLUG = "qwen36"
TITLE = "Qwen3.6 하이브리드 (Gated DeltaNet + Gated Attention)"
INTRO = """
원본: transformers `qwen3_5` (Qwen3.6 / 3.8-27B 텍스트 부분). 내 구현: `miniarch/models/qwen36.py`.

- 선형 attention(Gated DeltaNet) 3층 + Gated Attention 1층, partial RoPE
- 축소 config: 4층, 시퀀스 100 → chunk(64) 2개 + padding 경로까지 검사합니다.
"""
PIP = []
HEADER = [("Qwen/Qwen3.6-27B", r"layers\.(0|3)\.", "")]
CAUSAL = True
SELF_CHECKS = [
    ("""
### 선형 attention(Gated DeltaNet) recurrent == chunk

토큰 하나씩 메모리를 갱신하는 정의 그대로의 구현과, 학습용 chunk 행렬곱 구현이 같은 값을 내는지 확인합니다.
chunk 경계를 여러 번 넘도록 길이 150, chunk 32로 검사합니다.
""", """
from miniarch.models import qwen36 as Q

V.set_deterministic(0)
B_, H_, T_, Dk_, Dv_ = 2, 3, 150, 16, 8
q_ = Q.l2norm(torch.randn(B_, H_, T_, Dk_, device=dev)) * Dk_ ** -0.5
k_ = Q.l2norm(torch.randn(B_, H_, T_, Dk_, device=dev))
v_ = torch.randn(B_, H_, T_, Dv_, device=dev)
g_ = -torch.rand(B_, H_, T_, device=dev) * 0.3
b_ = torch.rand(B_, H_, T_, device=dev)
r = V.same_function_test("delta rule recurrent vs chunk(32)",
                         lambda: Q.delta_rule_recurrent(q_, k_, v_, g_, b_)[0],
                         lambda: Q.delta_rule_chunk(q_, k_, v_, g_, b_, chunk_size=32)[0])
report(r.name, r.ok, r.max_abs)
"""),
]


def ref():
    """선형 3층 + full 1층, 시퀀스 100 (chunk 64 → chunk 2개 + padding 경로 검사)."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, max_position_embeddings=256, use_cache=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0]})
    ids = torch.randint(0, VOCAB, (BATCH, 100), generator=torch.Generator().manual_seed(0))
    return Qwen3_5TextModel(cfg), {"input_ids": ids}


CASE = Case(STAGE, "Gated DeltaNet + Gated Attention", "Qwen3.6/3.8-27B", "transformers/models/qwen3_5",
            ref, "miniarch.models.qwen36", atol=1e-4)
