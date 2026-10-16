"""03_train_check 노트북 생성: 공용 학습 코드(⑤) 검증."""
from nbgen import NB

PATH = "notebooks/03_train_check.ipynb"
nb = NB()

nb.md(f"""
# 03 · 학습 코드 검증

{nb.badge(PATH)}

모델이 원본과 같다는 것(`notebooks/verify/`)과 별개로, **학습 코드**가 맞게 짜였는지 확인합니다.
모든 모델이 같은 학습 루프(`miniarch/trainer.py`)와 같은 태스크 코드(`miniarch/tasks/`)를 쓰므로 여기서 한 번 통과하면 모든 모델에 적용됩니다.

1. 태스크·데이터: MQAR 생성 규칙, 손실 마스킹, 정답 누수, 학습/평가 분리
2. 학습 루프: lr 스케줄, weight decay 대상, 평가가 학습 상태를 안 건드림, seed 재현, 체크포인트 재개, 기록(TensorBoard)
3. 검증된 모든 LM 을 공용 루프로 짧게 학습: NaN 없음, 손실 감소, 라우터 bias 업데이트, 재현·재개 / DiT 는 flow matching 으로
4. 알려진 결과 재현: attention 모델이 MQAR 을 푼다 (Zoology)
5. TensorBoard 로 기록 확인, 결과 저장

**런타임 → 런타임 유형 변경 → T4 GPU** 선택 후 위에서부터 실행하세요. (약 10~20분)
""")
nb.setup()
nb.code("""
# 재현성: cuBLAS 결정적 모드 + 결정적 알고리즘 (지원 안 되는 연산은 경고만)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import importlib, traceback, shutil
from miniarch.verify import train_tests as T
dev = "cuda" if torch.cuda.is_available() else "cpu"
TMP = "runs/train_check"
shutil.rmtree(TMP, ignore_errors=True)

CHECKS = []
def report_all(results):
    for name, ok, val in results:
        CHECKS.append({"check": name, "ok": bool(ok), "value": float(val)})
        print(f"{'✅' if ok else '❌'} {name:<52} {val:.3g}")

def run(fn, *a, **kw):
    try:
        report_all(fn(*a, **kw))
    except Exception:
        traceback.print_exc(limit=5)
        CHECKS.append({"check": fn.__name__, "ok": False, "value": float("nan")})
        print("💥", fn.__name__)
""")

nb.md("""
## 1. 태스크 · 데이터

- **MQAR 생성 규칙**: 키/값 범위, 키 중복 없음, 정답이 짝지어진 값, 손실 위치가 질문 위치뿐, 정답이 다음 칸 입력과 일치, 거리 표시가 맞음
- **학습/평가 분리**: modadd 는 (a, b) 쌍이 겹치지 않고, MQAR 평가 시퀀스는 학습 스트림에 나오지 않음
- **손실 마스킹**: 손실이 표시된 위치만의 평균 CE 와 같음
- **정답 누수**: 질문 위치 출력이 그 뒤 칸(정답 포함)을 바꿔도 같음
""")
nb.code("""
run(T.test_mqar_data)
run(T.test_split_disjoint)
run(T.test_loss_masking, dev)
run(T.test_no_answer_leak, dev)
""")

nb.md("""
## 2. 학습 루프

- **lr 스케줄**: 선형 warmup → cosine 으로 min_lr
- **weight decay**: 2차원 이상 가중치에만 (norm·bias·sink 제외), 라우터 bias 는 optimizer 에서 제외
- **평가 순수성**: 평가가 가중치·학습 배치 RNG 를 안 바꾸고, train 모드를 복원하고, 결정적
- **재현·재개**: 같은 seed 두 번 → 같은 손실, 10 step 저장 후 이어서 10 step == 한 번에 20 step
- **기록**: 필수 항목, attention MI 지표 자동 수집, TensorBoard 이벤트 파일
""")
nb.code("""
run(T.test_schedule_and_groups)
run(T.test_eval_purity, dev, f"{TMP}/purity")
from miniarch.tasks.mqar import MQAR, MQARConfig
run(T.test_reproducible_and_resume, T.tiny_llama, MQAR(MQARConfig(vocab_size=128, n_pairs=8)), dev, f"{TMP}/rr", "Llama: ")
run(T.test_logging, dev, TMP)
""")

nb.md("""
## 3. 검증된 모델들을 공용 루프로

`notebooks/verify/` 에서 원본과 같다고 확인한 축소 모델을 그대로 가져와 MQAR 로 30 step 학습합니다.
NaN 없음, 손실 감소, 라우터 bias 업데이트(있는 모델만), 같은 seed 재현, 체크포인트 재개를 봅니다.
MoE 의 index_add 같은 연산은 GPU 에서 비결정적일 수 있어 재현 허용 오차는 1e-4 입니다.
""")
nb.code("""
from miniarch.verify import stages
for S in stages.modules():
    if not getattr(S, "CAUSAL", False):
        continue
    mod = importlib.import_module(S.CASE.mine_module)
    model_fn = lambda S=S, mod=mod: mod.verify_pair(S.ref()[0])["mine"].float()
    print(f"\\n--- {S.STAGE} {S.TITLE}")
    run(T.model_smoke, f"{S.STAGE}_{S.SLUG}", model_fn, dev, TMP)
""")
nb.code("""
# DiT (B0): flow matching 손실로 같은 점검 (가짜 이미지 512장, 다운로드 없음)
from miniarch.models.dit import DiT, DiTConfig
from miniarch.tasks.lm_tasks import FlowMatchingTask
g = torch.Generator().manual_seed(0)
fm_task = FlowMatchingTask(torch.randn(512, 1, 28, 28, generator=g).clamp(-1, 1), torch.randint(0, 10, (512,), generator=g))
dit_fn = lambda: DiT(DiTConfig(d_model=64, n_layer=2, n_head=4))
def dit_smoke():
    tr = T._run(dit_fn, fm_task, f"{TMP}/B0_dit", dev, 30, steps=30, lr=1e-3, warmup=5, eval_every=30, log_every=5, mi_every=0)
    ls = [h["train/loss"] for h in tr.history]
    return [("B0_dit: 손실 유한", all(torch.isfinite(torch.tensor(ls))), ls[-1]), ("B0_dit: 손실 감소", ls[-1] < ls[0], ls[-1] / ls[0])] + \\
           T.test_reproducible_and_resume(dit_fn, fm_task, dev, f"{TMP}/B0_dit_rr", tag="B0_dit: ", atol=1e-4)
run(dit_smoke)
""")

nb.md("""
## 4. 알려진 결과 재현

작은 Llama(2층, d=64)가 MQAR(짝 8개)에서 평가 정확도 0.9 이상에 도달하는지 봅니다 (1500 step, 1분 안팎).
attention 모델은 이 설정을 거의 완벽히 푸는 것으로 알려져 있습니다. 여기서 실패하면 학습 루프·태스크·모델 어딘가가 어긋난 것입니다.
""")
nb.code("""
run(T.test_learns_mqar, dev, TMP)
""")

nb.md("""
## 5. TensorBoard · 결과 저장

같은 태그 이름(`train/loss`, `eval/acc`, `attn/entropy/L0` …)으로 기록되므로 run 들이 한 차트에 겹쳐 보입니다.
왼쪽 아래 정규식 필터로 보고 싶은 run 만 고르세요.
""")
nb.code("""
%load_ext tensorboard
%tensorboard --logdir runs/train_check
""")
nb.code("""
import datetime, pandas as pd
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(CHECKS)
df.to_csv(f"results/train_check_{stamp}.csv", index=False)
print(df.to_string(index=False))
print("\\n전체:", "✅ 모두 통과" if df.ok.all() else f"❌ 실패 {int((~df.ok).sum())}개")
zip_path = shutil.make_archive(f"/content/train_check_{stamp}", "zip", ".", "results")
shutil.make_archive(f"/content/train_check_runs_{stamp}", "zip", "runs", "train_check")
try:
    from google.colab import files
    files.download(zip_path)
except Exception as e:
    print("자동 다운로드 실패, 왼쪽 파일 탭에서 받으세요:", e)
""")
nb.save(PATH)
