"""00_check 노트북 생성. 노트북은 직접 편집하지 말고 이 파일을 고친 뒤 다시 실행한다."""
import json, os, sys

REPO = "HisameOgasahara/tmp_llm_dit"
cells = []

def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(f"""
# 00 · 검증 (도구 점검 + 구현 검증)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/00_check.ipynb)

이후 모든 블록 구현이 통과해야 하는 **판정 기준**을 먼저 점검하는 노트북입니다.

1. 환경 준비
2. 출력 비교 도구 자체 점검: 같은 함수면 PASS, 다르면 FAIL이 나오는지
3. 자체 일관성 검사 (causal mask 누수)
4. HF 원본 구조 읽기 (가중치 다운로드 없이 헤더만)
5. 구조(shape) 대조 도구 점검
6. 단계별 검증 현황표
7. 결과 저장

**런타임 → 런타임 유형 변경 → T4 GPU** 선택 후 위에서부터 실행하세요.
""")

md("## 1. 환경 준비")
code(f"""
import os
REPO_DIR = "/content/tmp_llm_dit"
if not os.path.exists(REPO_DIR):
    !git clone -q https://github.com/{REPO}.git {{REPO_DIR}}
else:
    !git -C {{REPO_DIR}} pull -q
%cd {{REPO_DIR}}
!pip -q install -U -r requirements-colab.txt
# Colab 기본 torchao 가 최신 diffusers 와 버전이 안 맞아 import 가 깨짐 → 쓰지 않으므로 제거
!pip -q uninstall -y torchao
""")
code("""
import sys, torch, transformers, diffusers
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__)
print("diffusers   ", diffusers.__version__)
!git log -1 --format='repo commit: %h %s'
""")
code("""
# gated 모델(Krea-2 등) 헤더를 읽을 때만 필요. Colab 왼쪽 🔑(보안 비밀)에 HF_TOKEN 등록.
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN 설정됨")
except Exception:
    print("HF_TOKEN 없음 (공개 모델만 확인 가능)")
""")

md("""
## 2. 출력 비교 도구 자체 점검

원본 예시로 HF GPT-2를 씁니다.
- **같은 구조**인 별도 모델에 가중치를 복사하면 → ✅ PASS가 나와야 합니다.
- **일부러 수식을 바꾼** 모델(LayerNorm eps 변경)은 → ❌ FAIL이 나와야 합니다.
- `hook_pairs`로 층별 중간 출력도 비교해서, 어느 층부터 어긋나는지 보여줍니다.
""")
code("""
from transformers import GPT2Config, GPT2Model
from miniarch import verify as V

V.set_deterministic(0)
ref, inputs = V.cases.ref_gpt2()

same = GPT2Model(ref.config)                                  # 같은 구조, 다른 랜덤 초기화
cfg_bad = GPT2Config(**{**ref.config.to_dict(), "layer_norm_epsilon": 1e-1})
bad = GPT2Model(cfg_bad)                                      # 수식이 살짝 다름

layers = [("h.0", "h.0"), ("h.1", "h.1")]
print("[같은 구조]");    print(V.check_equivalence(ref, same, inputs, hook_pairs=layers).summary())
print("\\n[다른 수식]"); print(V.check_equivalence(ref, bad,  inputs, hook_pairs=layers).summary())
""")

md("""
## 3. 자체 일관성 검사: causal mask 누수

뒤쪽 토큰을 바꿨을 때 앞쪽 출력이 변하면 미래 정보가 새고 있다는 뜻입니다.
HF Llama(정상)는 PASS, mask를 빼고 양방향으로 본 경우는 FAIL이 나와야 합니다.
""")
code("""
dev = "cuda" if torch.cuda.is_available() else "cpu"
llama, _ = V.cases.ref_llama()
llama = llama.to(dev).eval(); V.force_eager(llama)

causal = lambda x: llama(input_ids=x).last_hidden_state
# 모든 위치를 볼 수 있는 4D mask(0 = 허용)를 넘겨 일부러 양방향으로 만든 경우
bidir  = lambda x: llama(input_ids=x, attention_mask=torch.zeros(x.shape[0], 1, x.shape[1], x.shape[1], device=x.device)).last_hidden_state

for name, f in [("Llama causal", causal), ("Llama 양방향(누수)", bidir)]:
    try:
        r = V.causal_leak_test(f, vocab_size=V.cases.VOCAB, device=dev)
        print(f"{'✅' if r.ok else '❌'} {name:<18} max_abs={r.max_abs:.2e}")
    except Exception as e:
        print(f"⚠️ {name}: {type(e).__name__}: {e}")
""")

md("""
## 4. HF 원본 구조 읽기 (헤더만, 다운로드 없음)

safetensors 헤더를 HTTP range 요청으로 읽어서 실제 가중치 이름, shape, dtype을 확인합니다.
routed expert는 개수가 많아 숨깁니다.
""")
code("""
def peek(repo, pattern, subfolder="", limit=60):
    print(f"\\n===== {repo}  /{pattern}/")
    try:
        cfg = V.fetch_config(repo, subfolder)
        cfg = cfg.get("text_config", cfg)
        keys = ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim"]
        print({k: cfg[k] for k in keys if k in cfg})
        h = V.read_model_header(repo, subfolder=subfolder, pattern=pattern)
        h = {k: v for k, v in h.items() if ".experts." not in k}
        V.show(h, limit)
    except Exception as e:
        print(f"⚠️ {type(e).__name__}: {e}")

peek("deepseek-ai/DeepSeek-V4-Flash", r"layers\\.0\\.")              # MLA(V4) + mHC + sparse
peek("Qwen/Qwen3.6-27B",             r"layers\\.(0|3)\\.")          # Gated DeltaNet(0) / Gated Attn(3)
peek("moonshotai/Kimi-K3",           r"layers\\.(1|4)\\.")          # KDA(1) / MLA(4)
peek("Qwen/Qwen-Image-2.1",          r"blocks\\.0\\.|transformer_blocks\\.0\\.", subfolder="transformer")
""")

md("""
## 5. 구조(shape) 대조 도구 점검

원본 config로 **meta device**(메모리 사용 0)에 모델을 만들고, 그 state_dict를 원본 헤더와 대조합니다.
여기서는 HF의 Qwen3 구현과 실제 Qwen3-0.6B 체크포인트를 비교하니 ✅가 나와야 정상입니다.
나중에는 HF 구현 자리에 **내 구현**을 넣어서 같은 방식으로 검사합니다.
""")
code("""
from transformers import AutoConfig, AutoModelForCausalLM

repo = "Qwen/Qwen3-0.6B"
cfg = AutoConfig.from_pretrained(repo)
mine_shapes = V.meta_state_shapes(lambda: AutoModelForCausalLM.from_config(cfg))
orig = V.read_model_header(repo)
print(V.compare_shapes(mine_shapes, orig, scope=r"layers\\.0\\.|embed_tokens|model\\.norm").summary())
""")

md("""
## 6. 단계별 검증 현황표

- 원본 builder가 현재 라이브러리 버전에서 동작하는지 먼저 확인합니다.
- 그다음 전체 표를 출력합니다. 지금은 내 구현이 없어서 ⬜ 상태가 정상이고, 구현을 추가할 때마다 ✅로 바뀝니다.
""")
code("""
V.cases.smoke_refs()
""")
code("""
import pandas as pd
pd.set_option("display.max_colwidth", 120)
status = V.cases.run_all()
status[["stage", "block", "기준", "status"]]
""")
code("""
# 검사한 블록의 층별 상세 (PASS 도 어느 층까지 오차가 얼마인지 확인)
for _, r in status[~status.status.str.startswith("⬜")].iterrows():
    print(f"--- {r.stage} {r.block}  {r.status}\\n{r.detail}\\n")
""")

md("""
### 6-1. 내 구현 자체 점검

- **causal 누수**: GPT-2, Llama 구현이 미래 토큰을 보지 않는지 확인합니다.
- **attention map 캡처**: 모든 블록의 `attn_probs` 지점에서 확률 텐서를 hook으로 꺼낼 수 있는지 확인합니다. 각 행의 합은 1이어야 합니다. 이 지점이 이후 attention map 분석에 쓰입니다.
""")
code("""
from miniarch.models.gpt2 import GPT2, GPT2Config
from miniarch.models.llama import Llama, LlamaConfig
from miniarch.models.dit import DiT, DiTConfig

V.set_deterministic(0)
lms = {"GPT2": GPT2(GPT2Config(vocab_size=128)).to(dev).eval(),
       "Llama": Llama(LlamaConfig(vocab_size=128)).to(dev).eval()}
for name, m in lms.items():
    r = V.causal_leak_test(lambda x, m=m: m(x).last_hidden_state, vocab_size=128, device=dev)
    print(f"{'✅' if r.ok else '❌'} {name:<6} causal leak  max_abs={r.max_abs:.2e}")

def probe_attn(model, call, causal):
    names = [n for n, _ in model.named_modules() if n.endswith("attn_probs")]
    with V.capture(model, names) as st, torch.no_grad():
        call()
    p = st[names[0]]
    msg = f"{type(model).__name__:<6} {len(names)}개 층 캡처, shape={tuple(p.shape)}, 행 합 오차={(p.sum(-1) - 1).abs().max().item():.1e}"
    if causal:
        msg += f", 미래 토큰에 준 확률 최대={p.triu(1).abs().max().item():.1e}"
    print(msg)

x = torch.randint(0, 128, (2, 16), device=dev)
for m in lms.values():
    probe_attn(m, lambda m=m: m(x), causal=True)
dit = DiT(DiTConfig()).to(dev).eval()
img = torch.randn(2, 1, 28, 28, device=dev)
probe_attn(dit, lambda: dit(img, torch.tensor([10, 500], device=dev), torch.tensor([3, 7], device=dev)), causal=False)
""")

md("""
## 7. 결과 저장

- `results/`에 현황표를 저장합니다.
- 노트북 출력까지 남기려면 **파일 → GitHub에 사본 저장**으로 이 저장소의 `notebooks/`에 저장하세요.
- Drive에도 남기려면 아래 셀의 `SAVE_TO_DRIVE = True`로 바꾸세요.
""")
code("""
import datetime, shutil
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
os.makedirs("results", exist_ok=True)
path = f"results/00_check_{stamp}.csv"
status.to_csv(path, index=False)
print("saved:", path)

SAVE_TO_DRIVE = False
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    dst = "/content/drive/MyDrive/tmp_llm_dit/results"
    os.makedirs(dst, exist_ok=True)
    shutil.copy(path, dst)
    print("drive:", dst)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks/00_check.ipynb")
with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("ok", len(cells))
