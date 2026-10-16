# tmp_llm_dit · 최신 LLM / DiT 블록 해부 실습

최신 오픈웨이트 모델(DeepSeek-V4, Kimi-K3, Qwen3.6/3.8, GLM-5.3, MiniMax, Qwen-Image-2.1, Cosmos3 등)의 핵심 블록을 **네이티브 PyTorch로 축소 구현**합니다. 작은 태스크로 학습시킨 뒤 attention map, 표현, ablation을 비교합니다.

모든 블록은 **원본 코드와 출력이 같은지 검증을 통과한 뒤** 실험에 사용합니다.

## 노트북

| 노트북 | 내용 | 열기 |
|---|---|---|
| `00_check` | 검증 도구 점검, 원본 구조 읽기, 구현이 원본과 같은지 검증 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/00_check.ipynb) |
| `01_modadd` | GPT-2 vs Llama: modular addition grokking, 푸리에 해법, attention map, weight decay ablation (~10분) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/01_modadd.ipynb) |
| `02_fashion_dit` | DiT: FashionMNIST 클래스 조건 생성(flow matching + CFG), 층별 attention 거리 (~25분) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/02_fashion_dit.ipynb) |
| `03_train_check` | 학습 코드 검증: 태스크·마스킹·누수·분리, lr·weight decay, 재현·재개, TensorBoard 기록, 검증된 모델 전부 공용 루프로 짧게 학습, MQAR 재현 (~15분) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/03_train_check.ipynb) |

### 모델별 검증 (`notebooks/verify/`)

모델마다 노트북이 하나씩 있고, 서로 독립적으로 돌아갑니다. 원본 구조 확인 → 원본 동치 → causal 누수 → 단계 전용 점검 순서입니다.
**한 번에 전부 돌리기:** [`00_run_all`](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/00_run_all.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/00_run_all.ipynb) — 모델 노트북을 각각 새 프로세스로 실행하고, 요약표와 결과 zip(실행된 노트북 + csv)을 내려받습니다.

| 단계 | 모델 | 열기 |
|---|---|---|
| A2 | MiMo-V2.6 (SWA+full, sink, MoE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/A2_mimo_v26.ipynb) |
| A4 | DeepSeek-V4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/A4_deepseek_v4.ipynb) |
| A5a | GLM-5.3 (MLA + DSA indexer + MoE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/A5a_glm53.ipynb) |
| A5b | MiniMax-M3 MSA (블록 top-k, attention 층 단위) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/A5b_minimax_msa.ipynb) |
| A6a | Qwen3.6 하이브리드 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/A6a_qwen36.ipynb) |
| A6b | Kimi-K3 (KDA + gated MLA + AttnRes) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/A6b_kimi_k3.ipynb) |
| A6c | Qwen3.8 플래그십 2.4T-A95B (하이브리드 + MoE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/A6c_qwen38_moe.ipynb) |
| A7a | Qwen3.8-Flash-Next (HC, indexer, n-gram) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/A7a_qwen38_flash_next.ipynb) |
| A8 | DeepSeek-V4.1-Flash (CSA2, Engram) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/A8_deepseek_v41.ipynb) |
| A9 | StepFun Step-5-Preview (SWA+full, head 게이트, sparse indexer, MoE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/A9_step5.ipynb) |
| A10 | Gemma 4 (K=V 전역 층, MoE, PLE, KV 공유, unified) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/A10_gemma4.ipynb) |
| B1 | Qwen-Image (MM-DiT) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/B1_qwen_image.ipynb) |
| B2 | Krea-2 (Single-Stream DiT) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/B2_krea2.ipynb) |
| B3 | Anima (Cosmos-Predict2 DiT + LLM adapter) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/B3_anima.ipynb) |
| B4 | MiniMax-H3 (비디오 + 오디오 3D DiT) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/B4_minimax_h3.ipynb) |
| C0 | Qwen3.8-27B 비전 인코더 (ViT + M-RoPE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/C0_qwen38_vision.ipynb) |
| C1 | Cosmos3 (추론 + 생성 두 타워 MoT) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/C1_cosmos3.ipynb) |
| C2 | Qwen-Image-2.1 (생성 + 편집 통합) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/verify/C2_qwen_image21.ipynb) |

## 실행 방법

1. 배지를 눌러 Colab에서 엽니다.
2. **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다.
3. 첫 셀이 저장소를 clone하고 패키지를 설치합니다. 위에서부터 차례로 실행하세요.
4. 결과 저장
   - 출력이 포함된 노트북: **파일 → GitHub에 사본 저장** → 이 저장소의 `notebooks/`
   - 표와 그림: `results/`에 저장됩니다. 필요하면 Drive 저장 옵션을 켭니다.

## 구조

```
miniarch/
  verify/      원본과의 동치 검사 도구
    equivalence.py  가중치 복사 + 출력/중간 텐서 비교
    consistency.py  causal 누수, 증분 디코딩, 두 구현 비교
    hf_header.py    HF safetensors 헤더만 읽어 원본 shape 확인
    shapes.py       meta device 모델과 원본 shape 대조
    cases.py        단계별 검증 케이스 레지스트리 (전체 현황표)
    training.py     학습 동작 검증 (기울기·학습 궤적 == 원본, 배치 외우기, float64 수치 미분, 라우터 bias)
    train_tests.py  학습 코드 검증 (태스크·루프·재현·재개·기록)
    stages/         단계별 검증 케이스 (파일 하나 = 모델 하나 = 검증 노트북 하나)
  models/      단계별 구현 (각 모듈이 verify_pair 제공)
  tasks/       학습 태스크 (modadd, fashion)
  analysis/    해석 도구 (attention 캡처·거리·엔트로피, 푸리에)
  train.py     공용 학습 유틸 (seed, EMA)
  trainer.py   공용 학습 루프 (TensorBoard 기록, MI 지표 자동 수집, 라우터 bias 업데이트, 체크포인트 재개)
notebooks/     Colab 노트북 (직접 편집 금지, tools/ 에서 생성)
  verify/      모델별 검증 노트북 (tools/build_verify.py 가 stages/ 에서 생성)
tools/         노트북 생성 스크립트
results/       실험 결과
CURRICULUM.md  단계별 로드맵
```

## 런타임

| 구성 | 버전 |
|---|---|
| 런타임 | Google Colab T4 GPU · Python 3 |
| torch | Colab 기본 |
| transformers / diffusers | 최신 (`requirements-colab.txt`) |
