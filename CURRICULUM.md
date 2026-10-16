# 커리큘럼

## 원칙
1. 공통 뼈대(MiniLM, MiniDiT)는 고정하고, **한 단계에 부품 하나만** 교체합니다.
2. 측정 세트(성능, attention map, probe, ablation)는 모든 단계에서 같게 유지합니다.
3. 모든 블록은 `00_check`의 원본 출력 비교를 통과해야 실험에 넣습니다.
4. 기본 크기는 d_model 256, 4~8층, head 4~8개, 5~20M 파라미터입니다. T4 기준 태스크당 30분~2시간입니다.

## 대상 모델 현황 (2026-10 기준, HF에서 직접 확인)

### LLM
| 모델 | 공개 | 핵심 구조 | 정답 코드 (T4에서 실행 가능?) | 크기 변형 |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 2026-09 | V4 + Engram, 층 간 KV/indexer 공유, 비전, DSpark | HF `inference/model.py` (TileLang FP8 커널 → 커널만 PyTorch로 교체해야 함) | V4: Flash / Pro (+Base) |
| DeepSeek-V4 | 2026-07~08 | sliding/CSA/HCA attention, Lightning Indexer, sink, mHC, sqrt-softplus MoE | transformers `deepseek_v4` ✅ | Flash(43층) / Pro(61층) |
| Kimi-K3 | 2026-06 | KDA 3 : gated MLA 1, Attention Residuals, latent MoE | HF `modeling_kimi_k3.py`, `modeling_kimi_linear.py` ✅ | 단일 |
| Qwen3.6 / 3.8-27B | 2026-08 | Gated DeltaNet 3 : Gated Attention 1, partial RoPE | transformers `qwen3_5` ✅ | 27B dense / 35B-A3B MoE / 2.4T |
| Qwen3.8-Flash-Next | 2026-08 | 위 + HC, sparse indexer, n-gram 임베딩 | transformers `qwen4_exp` ✅ | – |
| GLM-5.3 | 2026-08 | MLA + DSA (indexer 4층마다 공유) | transformers `glm_moe_dsa` ✅ | 일반 / Flash(구조 다름) |
| MiniMax-M3 | 2026-06 | GQA + 블록 top-k sparse (MSA) | 모델 코드 없음, GitHub `MiniMax-AI/MSA` 테스트 참조 ⚠️ | 단일 |
| MiMo-V2.6 | 2026-09 | SWA 7 : full 1, SWA 층 sink | HF `modeling_mimo_v2.py` ✅ | Flash / Pro |

### 생성 (DiT) · 멀티모달
| 모델 | 공개 | 핵심 구조 | 정답 코드 | 크기 변형 |
|---|---|---|---|---|
| Qwen-Image-2.1 | 2026-09 | Single-Stream DiT 32층, 생성+편집 통합, Qwen3-VL 인코더 | DiffSynth, ComfyUI `qwen_image21`, diffusers ✅ | Turbo 등 |
| Krea-2 | 2026-06 | 12B DiT (HF 약관 동의 필요) | DiffSynth `krea2_dit.py`, ComfyUI `krea2`, diffusers ✅ | Raw / Turbo |
| Anima | 2026-01 | Cosmos-Predict2-2B 기반 애니메 특화 | DiffSynth `anima_dit.py`, ComfyUI `anima` ✅ | 단일 |
| MiniMax-H3 | 2026-07 | 비디오+오디오 3D DiT 50층 | DiffSynth `minimax_h3_dit.py` ✅ | 단일 |
| Cosmos3 | 2026-03~07 | LLM + diffusion expert MoT, 행동 출력(VLA) | diffusers `transformer_cosmos3.py`, GitHub `nvidia/cosmos` ✅ | Edge / Nano / Super |

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
2. 기준선 A0 GPT-2 · A1 Llama · B0 DiT 구현 ✅ → 첫 실험 `01_modadd`, `02_fashion_dit` ✅
3. 최신 블록 (진행 중)
   - A6a Qwen3.6 하이브리드 (Gated DeltaNet + Gated Attention) — 구현, 검증 대기
   - A4 DeepSeek-V4 (sliding/CSA/HCA + indexer + sink + mHC + MoE) — 구현, 검증 대기
   - 다음: DeepSeek-V4.1 추가분 (Engram, 층 간 KV/indexer 공유), Kimi-K3 (KDA + gated MLA + AttnRes), GLM-5.3, MiniMax-M3
4. 최신 블록 비교 실험 (연관 기억 MQAR, 긴 문맥, mHC/MoE ablation)
5. B1 ~ B4, C0 ~ C1
