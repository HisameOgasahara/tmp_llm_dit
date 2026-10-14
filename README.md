# tmp_llm_dit · 최신 LLM / DiT 블록 해부 실습

최신 오픈웨이트 모델(DeepSeek-V4, Kimi-K3, Qwen3.6/3.8, GLM-5.3, MiniMax, Qwen-Image-2.1, Cosmos3 등)의 핵심 블록을 **네이티브 PyTorch로 축소 구현**합니다. 작은 태스크로 학습시킨 뒤 attention map, 표현, ablation을 비교합니다.

모든 블록은 **원본 코드와 출력이 같은지 검증을 통과한 뒤** 실험에 사용합니다.

## 노트북

| 노트북 | 내용 | 열기 |
|---|---|---|
| `00_verify` | 검증 도구 점검, 원본 구조 읽기, 단계별 검증 현황 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm_dit/blob/main/notebooks/00_verify.ipynb) |

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
    cases.py        단계별 검증 케이스 레지스트리
  models/      단계별 구현 (각 모듈이 verify_pair 제공)
notebooks/     Colab 노트북
results/       실험 결과
CURRICULUM.md  단계별 로드맵
```

## 런타임

| 구성 | 버전 |
|---|---|
| 런타임 | Google Colab T4 GPU · Python 3 |
| torch | Colab 기본 |
| transformers / diffusers | 최신 (`requirements-colab.txt`) |
