"""A5a · GLM-5.3 (MLA + DSA indexer 층 간 공유 + 세분화 MoE) 검증 케이스."""
from __future__ import annotations

import torch

from ..cases import BATCH, VOCAB, Case

STAGE = "A5a"
SLUG = "glm53"
TITLE = "GLM-5.3 (MLA + DSA indexer 공유 + MoE)"
INTRO = """
원본: transformers `glm_moe_dsa` (GLM-5 ~ 5.3 공통, 순수 PyTorch). 내 구현: `miniarch/models/glm53.py`.

- MLA: q 저랭크(q_a → norm → q_b), KV 를 작은 잠재 c 로 압축한 뒤 head 별 k_nope / v 로 펼침, RoPE 는 공유 k_rope 채널에만 (interleaved)
- DSA: Lightning Indexer 가 질의마다 이전 토큰 중 top-k 만 고르고 attention 은 그 토큰만 봄.
  `indexer_types` 가 "shared" 인 층은 indexer 없이 직전 "full" 층의 선택을 재사용 (원본: 앞 3층 full, 이후 4층마다 full)
- MoE: sigmoid 점수 + 그룹 제한 top-k + aux-loss-free bias(선택에만) + shared expert, 앞쪽 층은 dense MLP

축소 config: 4층, indexer = [full, shared, shared, full], MLP = [dense, sparse, sparse, sparse],
indexer top-6 에 시퀀스 24 → 앞쪽 질의(이전 토큰 6개 이하)는 전부 선택, 뒤쪽은 일부만 선택되는 두 경로가 모두 쓰입니다.
shared 가 두 번 이어진 뒤 다시 full 이 나와 선택이 넘어가고 새로 계산되는 경로를 봅니다.
expert 8개를 4그룹으로 나누고 2그룹만 허용, top-3 → 그룹 마스킹 경로를 씁니다 (원본은 n_group=1).
bias 경로도 검사하려고 attention_bias=True 로 둡니다 (원본은 False). 라우팅 bias 버퍼는 0이 아닌 값으로 채웁니다.
"""
PIP = []
HEADER = [("zai-org/GLM-5.3-BF16", r"layers\.(2|3)\.", "")]
CAUSAL = True
SELF_CHECKS = [
    ("""
### attention map · top-k 선택 · 층 간 공유 확인

- `attn_probs`: [B, H, T, T]. 행 합 = 1, 선택되지 않은 칸의 확률은 정확히 0이어야 합니다.
- `sel_probe`: 실제로 보는 칸 (causal ∧ top-k). 질의마다 최대 index_topk 개.
- "shared" 층의 선택은 직전 "full" 층과 완전히 같아야 합니다.
""", """
from miniarch.models import glm53 as G
ref, _ = S.ref()
m = G.verify_pair(ref)["mine"].to(dev).eval()
names = [n for n, _ in m.named_modules() if n.endswith(("attn_probs", "sel_probe"))]
x = torch.randint(0, V.cases.VOCAB, (2, 24), device=dev)
with V.capture(m, names) as st, torch.no_grad():
    m(x)
c, last = m.config, None
for i in range(c.n_layer):
    p, sel = st[f"layers.{i}.self_attn.attn_probs"], st[f"layers.{i}.self_attn.sel_probe"]
    err = (p.sum(-1) - 1).abs().max().item()
    leak = p.masked_fill(sel[:, None], 0).abs().max().item()
    n_sel = int(sel.sum(-1).max().item())
    ok = err < 1e-5 and leak == 0 and n_sel <= c.index_topk
    if c.indexer_types[i] == "full":
        last = sel
    else:
        ok = ok and torch.equal(sel, last)
    print(f"layer {i} {c.indexer_types[i]:<6} probs={tuple(p.shape)} 행합 오차={err:.1e} 비선택 확률={leak:.1e} 최대 선택={n_sel}")
    report(f"layer {i} ({c.indexer_types[i]}) top-k 마스크", ok, err)
"""),
    ("""
### top-k ≥ 시퀀스 길이면 DSA == dense causal attention

indexer 가 모든 이전 토큰을 고르면 sparse attention 은 일반 causal attention 과 같아야 합니다
(`use_dsa=False` 스위치와 같은 가중치로 비교).
""", """
import dataclasses
from miniarch.models import glm53 as G
V.set_deterministic(0)
c = dataclasses.replace(G.config_from_hf(S.ref()[0].config), index_topk=10**6)
m1 = G.GLM53(c).to(dev).eval()
m2 = G.GLM53(dataclasses.replace(c, use_dsa=False)).to(dev).eval()
m2.load_state_dict(m1.state_dict())
x = torch.randint(0, V.cases.VOCAB, (2, 24), device=dev)
r = V.same_function_test("DSA(top-k ≥ T) == dense causal", lambda: m1(x).last_hidden_state,
                         lambda: m2(x).last_hidden_state, atol=1e-5)
report(r.name, r.ok, r.max_abs)
"""),
]


def ref():
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaModel
    cfg = GlmMoeDsaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        attention_bias=True, rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        index_topk=6, index_head_dim=16, index_n_heads=2,
        indexer_types=["full", "shared", "shared", "full"],
        first_k_dense_replace=1, mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=3, n_group=4, topk_group=2,
        norm_topk_prob=True, routed_scaling_factor=2.5, rms_norm_eps=1e-5,
        max_position_embeddings=256, use_cache=False, tie_word_embeddings=False,
        pad_token_id=None, bos_token_id=None, eos_token_id=None)
    m = GlmMoeDsaModel(cfg)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # aux-loss-free bias 를 0 이 아닌 값으로 (0 이면 선택 경로의 bias 효과가 안 보임)
        for layer in m.layers:
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None:
                gate.e_score_correction_bias.copy_(torch.randn(8, generator=g) * 0.1)
    ids = torch.randint(0, VOCAB, (BATCH, 24), generator=torch.Generator().manual_seed(0))
    return m, {"input_ids": ids}


CASE = Case(STAGE, "GLM-5.3 전체: MLA + DSA indexer(층 간 top-k 공유) + 그룹 제한 sigmoid MoE + shared expert",
            "GLM-5.3", "transformers/models/glm_moe_dsa", ref, "miniarch.models.glm53", atol=1e-4)
