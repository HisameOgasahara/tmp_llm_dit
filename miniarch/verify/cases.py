"""단계별 검증 케이스 레지스트리.

각 케이스 = (원본 builder, 내 구현 builder).
- ref():  작은 config 로 만든 원본 모듈과 입력을 돌려준다.
- mine(ref_module): 내 구현과 name_map 을 돌려준다. 아직 구현 전이면 ImportError/NotImplementedError.
구현이 추가될 때마다 mine 쪽만 채우면 run_all() 표가 초록불로 바뀐다.
"""
from __future__ import annotations

import importlib
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .equivalence import check_equivalence, perturb_params, set_deterministic

VOCAB, SEQ, BATCH = 128, 16, 2


@dataclass
class Case:
    stage: str
    name: str
    reference: str                 # 기준 모델
    ref_code: str                  # 정답 코드 위치
    ref: Callable[[], tuple[torch.nn.Module, Any]] | None = None
    mine_module: str | None = None  # miniarch.models.xxx (verify_pair(ref) 를 제공)
    atol: float = 1e-5
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 원본 builders

def _ids():
    return {"input_ids": torch.randint(0, VOCAB, (BATCH, SEQ), generator=torch.Generator().manual_seed(0))}


def ref_gpt2():
    from transformers import GPT2Config, GPT2Model
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=64, n_layer=2, n_head=4,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                     bos_token_id=None, eos_token_id=None)
    return GPT2Model(cfg), _ids()


def ref_llama():
    from transformers import LlamaConfig, LlamaModel
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaModel(cfg), _ids()


def ref_qwen36():
    """Qwen3.5/3.6 텍스트 모델. 선형 3층 + full 1층, 시퀀스 100 (chunk 64 → chunk 2개 + padding 경로 검사)."""
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


def ref_deepseek_v4():
    """DeepSeek-V4 축소판. 층 4개 = [sliding, CSA, HCA, CSA], MoE 첫 층은 hash 라우팅. 시퀀스 40.

    window 8, CSA 2토큰→1, HCA 8토큰→1, indexer top-3 → 모든 마스킹 경로(창 밖, 미완성 압축, indexer 미선택)가 쓰인다.
    """
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


def ref_dit():
    from diffusers import DiTTransformer2DModel
    m = DiTTransformer2DModel(num_attention_heads=4, attention_head_dim=16, in_channels=1,
                              num_layers=2, sample_size=28, patch_size=4, num_embeds_ada_norm=10)
    g = torch.Generator().manual_seed(0)
    inputs = {"hidden_states": torch.randn(BATCH, 1, 28, 28, generator=g),
              "timestep": torch.tensor([10, 500]),
              "class_labels": torch.tensor([3, 7])}
    return m, inputs


# ---------------------------------------------------------------- 레지스트리
# ref=None 인 케이스는 원본 builder 도 아직 없음 (해당 단계 착수 시 작성)

CASES: list[Case] = [
    Case("A0", "GPT-2 (MHA, learned pos, LayerNorm, GELU)", "gpt2", "transformers/models/gpt2",
         ref_gpt2, "miniarch.models.gpt2"),
    Case("A1", "Llama (RoPE, RMSNorm, SwiGLU, GQA)", "Llama", "transformers/models/llama",
         ref_llama, "miniarch.models.llama"),
    Case("A2", "QK-norm / SWA+full / sink / partial RoPE", "MiMo-V2.6", "HF XiaomiMiMo/*/modeling_mimo_v2.py"),
    Case("A4", "DeepSeek-V4 전체: sliding/CSA/HCA attention + indexer + sink + mHC + sqrt-softplus MoE",
         "DeepSeek-V4-Flash", "transformers/models/deepseek_v4", ref_deepseek_v4, "miniarch.models.deepseek_v4",
         atol=1e-4),
    Case("A3", "MoE 단독 비교 (GLM-5.3 DSA+MoE)", "GLM-5.3", "transformers/models/glm_moe_dsa"),
    Case("A5a", "DSA indexer (+층 간 공유)", "GLM-5.3", "transformers/models/glm_moe_dsa"),
    Case("A5b", "블록 top-k sparse (MSA)", "MiniMax-M3", "github MiniMax-AI/MSA tests"),
    Case("A6a", "Gated DeltaNet + Gated Attention", "Qwen3.6/3.8-27B", "transformers/models/qwen3_5",
         ref_qwen36, "miniarch.models.qwen36", atol=1e-4),
    Case("A6b", "KDA + gated MLA", "Kimi-K3", "HF moonshotai/Kimi-K3/modeling_kimi_linear.py"),
    Case("A7a", "Hyper-Connections / mHC", "DeepSeek-V4 · Qwen3.8-Flash-Next",
         "transformers/models/deepseek_v4, qwen4_exp"),
    Case("A7b", "Attention Residuals", "Kimi-K3", "HF moonshotai/Kimi-K3/modeling_kimi_k3.py"),
    Case("A8", "MTP / Engram / 층 간 KV 공유", "DeepSeek-V4.1", "HF deepseek-ai/DeepSeek-V4.1-Flash/inference"),
    Case("B0", "DiT (adaLN-Zero)", "DiT", "diffusers DiTTransformer2DModel",
         ref_dit, "miniarch.models.dit"),
    Case("B1", "MM-DiT (dual stream)", "Qwen-Image", "DiffSynth qwen_image_dit.py · ComfyUI qwen_image"),
    Case("B2", "Single-Stream DiT + 편집 조건", "Qwen-Image-2.1 · Krea-2",
         "DiffSynth qwen_image_21_dit.py, krea2_dit.py"),
    Case("B3", "Cosmos-Predict2 계열 DiT", "Anima", "DiffSynth anima_dit.py · ComfyUI anima, cosmos"),
    Case("B4", "3D 비디오 + 오디오 DiT", "MiniMax-H3", "DiffSynth minimax_h3_dit.py"),
    Case("C0", "미니 VLM (ViT + projector + M-RoPE)", "Qwen3-VL", "transformers/models/qwen3_vl"),
    Case("C1", "MoT world model / VLA", "Cosmos3", "diffusers transformer_cosmos3.py · github nvidia/cosmos"),
]


def get(stage: str) -> Case:
    return next(c for c in CASES if c.stage == stage)


def run_case(case: Case, device: str | None = None) -> dict:
    row = {"stage": case.stage, "block": case.name, "기준": case.reference, "status": "", "detail": ""}
    if case.ref is None:
        row["status"] = "⬜ 원본 builder 미작성"
        return row
    set_deterministic(0)
    try:
        ref, inputs = case.ref()
    except Exception as e:  # 원본 라이브러리 버전 문제 등
        row["status"] = "⚠️ 원본 생성 실패"
        row["detail"] = f"{type(e).__name__}: {e}"
        return row
    try:
        mod = importlib.import_module(case.mine_module)
    except ModuleNotFoundError as e:
        if e.name != case.mine_module:  # 모듈은 있는데 내부 import 가 깨진 경우
            raise
        row["status"] = "⬜ 미구현"
        return row
    if not hasattr(mod, "verify_pair"):
        row["status"] = "⬜ 미구현"
        return row
    perturb_params(ref)  # 0/1 초기값 때문에 복사 누락이 가려지지 않도록
    try:
        pair = mod.verify_pair(ref)  # -> dict(mine=..., name_map=..., 기타 check_equivalence kwargs)
        mine = pair.pop("mine")
        rep = check_equivalence(ref, mine, inputs, atol=case.atol, device=device, **pair)
        row["status"] = "✅ PASS" if rep.ok else "❌ FAIL"
        row["detail"] = rep.summary()
    except Exception:
        row["status"] = "💥 에러"
        row["detail"] = traceback.format_exc(limit=3)
    return row


def run_all(device: str | None = None, stages: list[str] | None = None):
    import pandas as pd
    rows = [run_case(c, device) for c in CASES if stages is None or c.stage in stages]
    return pd.DataFrame(rows)


def smoke_refs(device: str | None = None) -> None:
    """원본 builder 들이 현재 Colab 라이브러리 버전에서 정상 동작하는지 확인."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    for c in CASES:
        if c.ref is None:
            continue
        try:
            m, x = c.ref()
            m = m.to(device).eval()
            x = {k: v.to(device) for k, v in x.items()}
            with torch.no_grad():
                out = m(**x)
            t = out[0] if isinstance(out, tuple) else getattr(out, "last_hidden_state", None)
            t = t if t is not None else out.sample
            n = sum(p.numel() for p in m.parameters())
            print(f"✅ {c.stage:<4} {c.reference:<8} params={n:,}  out={tuple(t.shape)}")
        except Exception as e:
            print(f"❌ {c.stage:<4} {c.reference:<8} {type(e).__name__}: {e}")
