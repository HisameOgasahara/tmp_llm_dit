# 커리큘럼

## 원칙
1. 공통 뼈대(MiniLM, MiniDiT)는 고정하고, **한 단계에 부품 하나만** 교체합니다.
2. 측정 세트(성능, attention map, probe, ablation)는 모든 단계에서 같게 유지합니다.
3. 모든 블록은 `00_check`의 원본 출력 비교를 통과해야 실험에 넣습니다.
4. 기본 크기는 d_model 256, 4~8층, head 4~8개, 5~20M 파라미터입니다. T4 기준 태스크당 30분~2시간입니다.

## 트랙 A · LLM 계보

| 단계 | 바뀌는 부품 | 기준 모델 | 정답 코드 |
|---|---|---|---|
| A0 | MHA, 학습형 위치 임베딩, LayerNorm, GELU | GPT-2 | transformers `gpt2` |
| A1 | RoPE, RMSNorm, SwiGLU, GQA | Llama | transformers `llama` |
| A2 | QK-norm, SWA+full 혼합, attention sink, partial RoPE | MiMo-V2.6 | HF `modeling_mimo_v2.py` |
| A3 | MoE: 잘게 나눈 expert + shared, aux-loss-free | DeepSeek-V4, GLM-5.3 | transformers `deepseek_v4`, `glm_moe_dsa` |
| A4 | MLA (V3 → V4: KV head 1개, 출력 저랭크) | DeepSeek-V4 | transformers `deepseek_v4` |
| A5 | sparse: DSA indexer(층 간 공유) / 블록 top-k | GLM-5.3 / MiniMax-M3 | `glm_moe_dsa` / GitHub `MiniMax-AI/MSA` |
| A6 | Gated DeltaNet + Gated Attn / KDA + gated MLA | Qwen3.6·3.8-27B / Kimi-K3 | `qwen3_5` / HF `modeling_kimi_linear.py` |
| A7 | Residual → HC → mHC / Attention Residuals | DeepSeek-V4, Qwen3.8-Flash-Next / Kimi-K3 | `deepseek_v4`, `qwen4_exp` / `modeling_kimi_k3.py` |
| A8 | MTP, Engram, 층 간 KV 공유 | DeepSeek-V4.1 | HF `inference/model.py`, `engram.py` |

태스크: modular addition(grokking) → 정렬/복사/MQAR → 여러 자리 덧셈 → 미로 최단경로. 보조로 TinyShakespeare.

## 트랙 B · DiT 계보

| 단계 | 구조 | 기준 모델 | 정답 코드 |
|---|---|---|---|
| B0 | DiT (adaLN-Zero) + flow matching | DiT | diffusers `DiTTransformer2DModel` |
| B1 | MM-DiT (dual stream) | Qwen-Image | DiffSynth `qwen_image_dit.py` |
| B2 | Single-Stream DiT + 편집 조건 | Qwen-Image-2.1, Krea-2 | DiffSynth `qwen_image_21_dit.py`, `krea2_dit.py` |
| B3 | Cosmos-Predict2 계열 | Anima | DiffSynth `anima_dit.py`, ComfyUI `anima` |
| B4 | 3D 비디오 + 오디오 | MiniMax-H3 | DiffSynth `minimax_h3_dit.py` |

태스크: FashionMNIST 조건부 생성 → 편집(가린 부분 채우기, 클래스 바꾸기) → Moving-MNIST.

## 트랙 C · 멀티모달 / VLA

| 단계 | 구조 | 기준 모델 | 정답 코드 |
|---|---|---|---|
| C0 | 미니 VLM: ViT + projector + M-RoPE | Qwen3-VL | transformers `qwen3_vl` |
| C1 | MoT world model / VLA (행동 출력) | Cosmos3 | diffusers `transformer_cosmos3.py`, GitHub `nvidia/cosmos` |

태스크: FashionMNIST 이미지 질의응답 → 이미지로 본 미로에서 행동 시퀀스 출력.

## 크기 비교 트랙 (실제 가중치, 학습 없음)
- 대상: DeepSeek-V4 Flash/Pro, MiMo-V2.6 Flash/Pro, Cosmos3 Edge/Nano/Super, Qwen3.6 27B dense / 35B-A3B MoE
- 방법: 헤더로 구조를 비교하고, range 요청으로 층 몇 개만 받아 가중치 스펙트럼과 norm 통계를 비교합니다.

## 진행 순서
1. Phase 0: 검증 도구 (`00_check`) ✅
2. A0, A1, B0 구현 ✅ → 첫 실험 `01_modadd`, `02_fashion_dit` ✅
3. A6a Qwen3.6 하이브리드 (Gated DeltaNet + Gated Attention) ← 현재, 다음: DeepSeek-V4.1-Flash (MLA V4 · mHC · indexer · Engram)
3. A2 ~ A5
4. A6 ~ A8
5. B1 ~ B4, C0 ~ C1
