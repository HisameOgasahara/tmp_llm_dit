# 커리큘럼

## 원칙
1. **구현은 모델 단위, 실험은 부품 단위.** 최신 모델은 부품이 얽혀 있어 원본과 1:1로 검증하려면 통째로 구현해야 합니다. 대신 검증된 모델에 부품별 스위치(예: mHC on/off, MoE on/off, sliding/CSA/HCA)를 두고, 실험에서는 **한 번에 스위치 하나만** 바꿉니다.
2. 측정 세트(학습 로그, 태스크 지표, attention map, probe, ablation)는 **공용 코드 하나**로 모든 모델에 같게 적용합니다. 모델마다 따로 짜지 않고, hook 지점 이름(`attn_probs`, `sink_probe`, `route_probe` 등)으로 자동 수집합니다.
3. 모든 블록은 실험 전에 두 가지 검증을 통과해야 합니다.
   - **forward 검증**: 원본과 출력이 같은가 (층별 비교, causal 누수, 단계 전용 점검)
   - **학습 동작 검증**: 원본과 기울기가 같은가, 원본과 나란히 학습했을 때 궤적(손실·가중치)이 같은가, 배치 하나를 외울 수 있는가
   - **학습 코드 검증**: 원본에 없는 학습 전용 부품(MoE bias 업데이트, aux loss, MTP 손실)은 정의대로 단위 테스트, 커스텀 연산은 float64 수치 미분.
     학습 루프·데이터는 손실 마스킹, 정답 누수, train/eval 분리, 시드 재현, 체크포인트 재개를 테스트. 마지막으로 알려진 결과(modadd grokking, MQAR) 재현
   스위치는 검증 통과 후에 추가하고, 모든 스위치를 기본값으로 둔 상태가 원본과 같은지 다시 확인합니다.
4. 기본 크기는 d_model 256, 4~8층, head 4~8개, 5~20M 파라미터입니다. T4 기준 태스크당 30분~2시간입니다.

## 대상 모델 현황 (2026-10 기준, HF에서 직접 확인)

### LLM
| 모델 | 공개 | 핵심 구조 | 정답 코드 (T4에서 실행 가능?) | 크기 변형 |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 2026-09 | V4 + Engram(1·14층 n-gram 해시 테이블), CSA2(층마다 Full/Reindex/Reuse, 40층 중 4층만 전역 KV), 비전, DSpark | HF `inference/model.py` (TileLang FP8 커널 → 커널만 PyTorch로 교체해야 함) | V4: Flash / Pro (+Base) |
| DeepSeek-V4 | 2026-07~08 | K = V 단일 head + sliding/CSA/HCA attention (MLA 아님), Lightning Indexer, sink, mHC, sqrt-softplus MoE | transformers `deepseek_v4` ✅ | Flash(43층) / Pro(61층) |
| Kimi-K3 | 2026-06 | KDA 3 : gated MLA 1, Attention Residuals, latent MoE | HF `modeling_kimi_k3.py`, `modeling_kimi_linear.py` ✅ | 단일 |
| Qwen3.6 / 3.8-27B | 2026-08 | Gated DeltaNet 3 : Gated Attention 1, partial RoPE. 3.8-27B는 비전 인코더 내장(이미지·비디오 입력) | transformers `qwen3_5` ✅ | 27B dense / 35B-A3B MoE / 2.4T(오픈 가중치는 텍스트 전용) |
| Qwen3.8-2.4T-A95B (플래그십) | 2026-08 | 위 하이브리드 92층 + MoE (expert 512개 중 10개 + sigmoid 게이트 shared expert). HC·indexer·n-gram 없음. 오픈 가중치는 텍스트 전용 | transformers `qwen3_5_moe` ✅ | 2.4T-A95B / 3.6-35B-A3B |
| Qwen3.8-Flash-Next | 2026-08 | 위 + HC, sparse indexer, n-gram 임베딩, 비전 인코더 내장 | transformers `qwen4_exp` ✅ | 125B-A6B |
| GLM-5.3 | 2026-08 | MLA + DSA (indexer 4층마다 공유) | transformers `glm_moe_dsa` ✅ | 일반 / Flash(구조 다름) |
| MiniMax-M3 | 2026-06 | GQA + 블록 top-k sparse (MSA) | 모델 코드 없음, GitHub `MiniMax-AI/MSA` 테스트 참조 ⚠️ | 단일 |
| Step-5-Preview (StepFun) | 2026-09 | SWA 3 : full 1 (층별 partial RoPE·theta), head별 attention 게이트, sparse indexer, MoE 352개 중 8개 + shared + router bias, MTP 3층 | HF `stepfun-ai/Step-5-Preview` `inference/model.py` (커널 → PyTorch 교체) | Preview / BF16 / NVFP4 |
| Gemma 4 | 2026-04~06 | SWA 5 : full 1, 전역 층은 head_dim 512 + K=V, sandwich norm, logit softcap. 26B-A4B는 MoE, E2B/E4B는 층별 입력(PLE) + 뒤쪽 층 KV 공유, 12B는 unified | transformers `gemma4`, `gemma4_unified` ✅ | E2B / E4B / 12B / 26B-A4B / 31B |
| MiMo-V2.6 | 2026-09 | SWA 7 : full 1, head별 학습형 sink (gpt-oss 방식) | HF `modeling_mimo_v2.py` ✅ | Flash / Pro |

### 생성 (DiT) · 멀티모달
| 모델 | 공개 | 핵심 구조 | 정답 코드 | 크기 변형 |
|---|---|---|---|---|
| Qwen-Image-2.1 | 2026-09 | Single-Stream DiT 32층, 생성+편집 통합, Qwen3-VL 인코더 | DiffSynth, ComfyUI `qwen_image21`, diffusers ✅ | Turbo 등 |
| Krea-2 | 2026-06 | 12B DiT (HF 약관 동의 필요) | DiffSynth `krea2_dit.py`, ComfyUI `krea2`, diffusers ✅ | Raw / Turbo |
| Anima | 2026-01 | Cosmos-Predict2-2B 기반 애니메 특화 | DiffSynth `anima_dit.py`, ComfyUI `anima` ✅ | 단일 |
| MiniMax-H3 | 2026-07 | 비디오+오디오 3D DiT 50층 | DiffSynth `minimax_h3_dit.py` ✅ | 단일 |
| Cosmos3 | 2026-03~07 | MoT 두 타워: 추론(AR, causal) + 생성(diffusion, 양방향). 층마다 attention 공유, 파라미터 분리. 행동 출력(VLA) | diffusers `transformer_cosmos3.py`, GitHub `nvidia/cosmos` ✅ | Edge 4B / Nano 16B / Super 64B |
| GR00T-N1.7 | 2026-04 EA → GA | VLM 백본(Cosmos-Reason2-2B / Qwen3-VL) + 행동 생성, 휴머노이드 VLA | GitHub `NVIDIA/Isaac-GR00T`, HF `nvidia/GR00T-N1.7` ✅ | 단일 |
| Qwen-Drive-1.0-4B | 2026-09 | Qwen3.5-4B VL + 3D 인식 헤드 + flow matching DiT 경로 planner | GitHub `QwenLM/Qwen-Drive-1.0` ✅ | SFT / RL |

## 트랙 A · LLM 계보

| 단계 | 바뀌는 부품 | 기준 모델 | 정답 코드 |
|---|---|---|---|
| A0 | MHA, 학습형 위치 임베딩, LayerNorm, GELU | GPT-2 | transformers `gpt2` |
| A1 | RoPE, RMSNorm, SwiGLU, GQA | Llama | transformers `llama` |
| A2 | QK-norm, SWA+full 혼합, attention sink, partial RoPE | MiMo-V2.6 | HF `modeling_mimo_v2.py` |
| A3 | MoE: 잘게 나눈 expert + shared, aux-loss-free (sqrt-softplus, hash 라우팅) | DeepSeek-V4, GLM-5.3 | transformers `deepseek_v4`, `glm_moe_dsa` |
| A4 | K = V 단일 head + 시퀀스 방향 KV 압축 (sliding / CSA 4→1 겹침 + indexer / HCA 128→1), q·출력 저랭크 | DeepSeek-V4 | transformers `deepseek_v4` |
| A5 | sparse: DSA indexer(층 간 공유) / 블록 top-k | GLM-5.3 / MiniMax-M3 | `glm_moe_dsa` / GitHub `MiniMax-AI/MSA` |
| A6 | Gated DeltaNet + Gated Attn (a: dense, c: + MoE 플래그십) / KDA + gated MLA (b) | Qwen3.6·3.8-27B, Qwen3.8-2.4T / Kimi-K3 | `qwen3_5`, `qwen3_5_moe` / HF `modeling_kimi_linear.py` |
| A7 | Residual → HC → mHC / Attention Residuals | DeepSeek-V4, Qwen3.8-Flash-Next / Kimi-K3 | `deepseek_v4`, `qwen4_exp` / `modeling_kimi_k3.py` |
| A8 | MTP, Engram, CSA2 (층마다 Full/Reindex/Reuse로 KV·indexer 공유) | DeepSeek-V4.1 | HF `inference/model.py`, `engram.py` |
| A9 | SWA+full + head별 게이트 + sparse indexer + MoE (StepFun) | Step-5-Preview | HF `inference/model.py` |
| A10 | 경량 모델 설계: 전역 층 K=V·큰 head, PLE, 층 간 KV 공유, MoE | Gemma 4 | transformers `gemma4`, `gemma4_unified` |

MLA(latent 차원 압축)는 V4에서 빠졌습니다. GLM-5.3(A5), Kimi-K3(A6)에서 다룹니다.
A4와 V4의 A3·A5·A7 부품은 DeepSeek-V4 하나로 구현·검증하고, 실험은 스위치로 부품을 하나씩 켭니다. A2 MiMo는 "sliding + sink만 켠 상태"의 기준점입니다.

태스크: modular addition(grokking) → 정렬/복사/MQAR → 여러 자리 덧셈 → 미로 최단경로. 보조로 TinyShakespeare.

## 트랙 B · DiT 계보

| 단계 | 구조 | 기준 모델 | 정답 코드 |
|---|---|---|---|
| B0 | DiT (adaLN-Zero) + flow matching | DiT | diffusers `DiTTransformer2DModel` |
| B1 | MM-DiT (dual stream) | Qwen-Image | DiffSynth `qwen_image_dit.py` |
| B2 | Single-Stream DiT | Krea-2 | DiffSynth `krea2_dit.py`, diffusers |
| B3 | Cosmos-Predict2 계열 | Anima | DiffSynth `anima_dit.py`, ComfyUI `anima`, diffusers `AnimaModularPipeline` |
| B4 | 3D 비디오 + 오디오 | MiniMax-H3 | DiffSynth `minimax_h3_dit.py` |

B3는 Anima를 유지합니다. 최신 Cosmos3의 생성 쪽은 독립 DiT가 아니라 추론 타워와 attention을 공유하는 MoT라서 C1에서 다룹니다.

태스크: FashionMNIST 조건부 생성 → 편집(가린 부분 채우기, 클래스 바꾸기) → Moving-MNIST.

## 트랙 C · 멀티모달 / VLA

| 단계 | 구조 | 기준 모델 | 정답 코드 |
|---|---|---|---|
| C0 | 미니 VLM: 비전 인코더 내장 하이브리드 LM (A6a 위에 ViT + M-RoPE) | Qwen3.8-27B | transformers `qwen3_5` |
| C1 | MoT world model / VLA (행동 출력) | Cosmos3 (비교: GR00T-N1.7) | diffusers `transformer_cosmos3.py`, GitHub `nvidia/cosmos` / `NVIDIA/Isaac-GR00T` |
| C2 | 생성+편집 통합: Qwen3-VL 인코더 조건 + Single-Stream DiT 32층 | Qwen-Image-2.1 | diffusers, DiffSynth `qwen_image_21_dit.py` |

태스크: FashionMNIST 이미지 질의응답 → 이미지로 본 미로에서 행동 시퀀스 출력.

## 크기 비교 트랙 (실제 가중치, 학습 없음)
- 대상: DeepSeek-V4 Flash/Pro, MiMo-V2.6 Flash/Pro, Cosmos3 Edge/Nano/Super, Qwen3.6 27B dense / 35B-A3B MoE
- 방법: 헤더로 구조를 비교하고, range 요청으로 층 몇 개만 받아 가중치 스펙트럼과 norm 통계를 비교합니다.

## 진행 순서
1. Phase 0: 검증 도구 (`00_check`) ✅
2. 기준선 A0 GPT-2 · A1 Llama · B0 DiT 구현 ✅ → 첫 실험 `01_modadd`, `02_fashion_dit` ✅
   - `01_modadd` 100k-step 긴 학습: 셀 오류 수정 후 재실행 필요
3. 최신 모델 구현·검증 ✅ (2026-10-16, Colab T4, `notebooks/verify/00_run_all` → `results/verify/run_20261016-0422/`)
   - 18개 모델, 점검 328개 전부 통과. 원본 대비 최종 출력 오차 0 ~ 1e-5
   - LLM: A2 MiMo-V2.6, A4 DeepSeek-V4, A5a GLM-5.3, A5b MiniMax-M3 MSA(attention 층 단위), A6a Qwen3.6, A6b Kimi-K3, A6c Qwen3.8 플래그십 2.4T, A7a Qwen3.8-Flash-Next, A8 DeepSeek-V4.1-Flash, A9 Step-5-Preview, A10 Gemma 4
   - DiT: B1 Qwen-Image, B2 Krea-2, B3 Anima, B4 MiniMax-H3 / 멀티모달: C0 Qwen3.8-27B 비전, C1 Cosmos3, C2 Qwen-Image-2.1
   - ⚠️ 여기까지는 forward 만 검증했습니다. 학습 쪽 검증이 빠져 있었습니다.
   - **3-2. 모델 학습 동작 검증** (구현 완료, Colab 실행 대기 → `notebooks/train_verify/00_run_all`, 순전파 검증과 분리): 모델별 노트북에
     ② 기울기 == 원본, ③ 학습 궤적 == 원본(SGD 30 step), 배치 하나 외우기, ④ float64 수치 미분, 라우터 bias 업데이트
     (원본이 추론 전용인 A5b·A8·A9 는 ②③ 대신 ④ 수치 미분으로 확인)
   - **3-3. 학습 코드 검증 ✅** (2026-10-16, `notebooks/03_train_check`, `results/train_check/`): 88개 전부 통과.
     MQAR 재현: 처음엔 모든 키를 한 번씩 묻는 설계라 소거법 지름길(정확도 0.34)에 갇힘 → 질문 중복 허용으로 수정.
     Llama(2층, d=128)가 16,500 step 까지 0.16 에서 정체하다 17,000~19,500 step 에 0.28 → 0.91 로 급상승 (induction 회로 형성 시점).
     세부 항목: ⑤ 태스크·마스킹·정답 누수·학습/평가 분리,
     lr 스케줄·weight decay 대상·평가 순수성·seed 재현·체크포인트 재개·TensorBoard 기록, 검증된 LM 전부 + DiT 를 공용 루프로 짧게 학습,
     ⑥ 알려진 결과 재현(Llama 가 MQAR 풂)
   - 공용 학습 루프 `miniarch/trainer.py`: 태그 이름 통일, run 폴더 = runs/<태스크>/<모델>/<변형>/seed<k>, MI 지표 자동 수집
   - 아직 없는 학습 전용 부품: MoE aux loss(Qwen·Gemma 방식), MTP 손실 → 해당 실험에 들어갈 때 추가
   - **다음:** 위 두 노트북 통과 → 4번 실험 (MQAR 태스크는 `miniarch/tasks/mqar.py` 로 추가됨)
4. 부품별 ablation 실험: A1 → A2 → DeepSeek-V4 스위치 (sliding만 → +CSA/HCA → +MoE → +mHC), Qwen3.6 선형:full 비율. 태스크 MQAR → 덧셈 → 미로
5. DiT·멀티모달 실험: 편집, Moving-MNIST, 이미지 질의응답, 미로 행동
