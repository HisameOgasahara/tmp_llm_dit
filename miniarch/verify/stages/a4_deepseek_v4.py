"""A4 · DeepSeek-V4 (A3 MoE · A4 attention · A5 indexer · A7 mHC 부품 포함) 검증 케이스."""
from __future__ import annotations

import torch

from ..cases import BATCH, VOCAB, Case

STAGE = "A4"
SLUG = "deepseek_v4"
TITLE = "DeepSeek-V4"
INTRO = """
원본: transformers `deepseek_v4` (순수 PyTorch). 내 구현: `miniarch/models/deepseek_v4.py`.

한 모델에 부품이 여럿 들어 있어서 통째로 검증합니다.
- attention: K = V 단일 head, sliding / CSA(압축 + indexer top-k) / HCA(강한 압축), head별 sink
- MoE: sqrt-softplus 점수 + top-k, 앞쪽 층은 토큰 id로 expert를 고정하는 hash 라우팅, shared expert
- mHC: residual을 여러 줄기로 들고 다니며 Sinkhorn으로 섞음

축소 config: 4층 = [sliding, CSA, HCA, CSA], window 8, CSA 2→1, HCA 8→1, indexer top-3, 시퀀스 40.
창 밖, 아직 다 안 찬 압축 칸, indexer가 고르지 않은 칸까지 모든 마스킹 경로가 쓰이도록 고른 값입니다.
"""
PIP = []
HEADER = [("deepseek-ai/DeepSeek-V4-Flash", r"layers\.0\.", "")]
CAUSAL = True
SELF_CHECKS = [
    ("""
### attention map · sink · 라우팅 hook 확인

- `attn_probs`: [B, H, T, T + 압축 칸]. 각 행 합 + sink 확률 = 1이어야 합니다.
- `sink_probe`: sink로 간 확률. `route_probe`: 토큰마다 고른 expert.
""", """
from miniarch.models import deepseek_v4 as D
ref, _ = S.ref()
m = D.verify_pair(ref)["mine"].to(dev).eval()
names = [n for n, _ in m.named_modules() if n.endswith(("attn_probs", "sink_probe"))]
x = torch.randint(0, V.cases.VOCAB, (2, 40), device=dev)
with V.capture(m, names) as st, torch.no_grad():
    m(x)
for i in range(m.config.n_layer):
    p, s = st[f"layers.{i}.self_attn.attn_probs"], st[f"layers.{i}.self_attn.sink_probe"]
    err = (p.sum(-1) + s - 1).abs().max().item()
    print(f"layer {i} {m.config.layer_types[i]:<32} probs={tuple(p.shape)}  행합+sink 오차={err:.1e}  평균 sink={s.mean():.3f}")
    report(f"layer {i} 확률 합", err < 1e-5, err)
"""),
]


def ref():
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model
    cfg = DeepseekV4Config(
        vocab_size=VOCAB, hidden_size=64, moe_intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=4, head_dim=32, partial_rotary_factor=0.25, q_lora_rank=32,
        o_groups=2, o_lora_rank=16, sliding_window=8,
        layer_types=["sliding_attention", "compressed_sparse_attention",
                     "heavily_compressed_attention", "compressed_sparse_attention"],
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 8},
        mlp_layer_types=["hash_moe", "moe", "moe", "moe"],
        index_n_heads=2, index_head_dim=16, index_topk=3, hc_mult=4, hc_sinkhorn_iters=20,
        n_routed_experts=4, num_experts_per_tok=2, max_position_embeddings=256, use_cache=False,
        pad_token_id=None, bos_token_id=None, eos_token_id=None)
    m = DeepseekV4Model(cfg)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # 라우팅 버퍼를 0 이 아닌 값으로 (0 이면 hash 가 expert 0 만 고르고 bias 효과도 안 보임)
        for layer in m.layers:
            gate = layer.mlp.gate
            if hasattr(gate, "tid2eid"):
                gate.tid2eid.copy_(torch.randint(0, 4, gate.tid2eid.shape, generator=g))
            else:
                gate.e_score_correction_bias.copy_(torch.randn(4, generator=g) * 0.1)
    ids = torch.randint(0, VOCAB, (BATCH, 40), generator=torch.Generator().manual_seed(0))
    return m, {"input_ids": ids}


CASE = Case(STAGE, "DeepSeek-V4 전체: K=V 단일 head + sliding/CSA/HCA attention + indexer + sink + mHC + sqrt-softplus MoE",
            "DeepSeek-V4-Flash", "transformers/models/deepseek_v4", ref, "miniarch.models.deepseek_v4", atol=1e-4)
