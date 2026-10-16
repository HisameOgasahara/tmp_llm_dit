"""04_mqar 노트북 생성: 4단계 부품별 실험 (MQAR)."""
from nbgen import NB

PATH = "notebooks/04_mqar.ipynb"
nb = NB()

nb.md(f"""
# 04 · 부품별 실험 — MQAR (연관 기억)

{nb.badge(PATH)}

같은 태스크(MQAR, 키-값 짝 16개), 같은 측정, 같은 크기(d_model=128)에서 **스위치 하나만** 바꿔 비교합니다.
모델 코드와 학습 루프는 `notebooks/verify`, `train_verify`, `03_train_check` 에서 검증을 마친 것을 그대로 씁니다.

    입력:  k1 v1 k2 v2 … k16 v16 | q1 a1 q2 a2 … q16 a16      (qi 위치에서 ai 를 맞힘)
    거리 = 질문 위치 − 그 키의 위치 (2 ~ 62)

| 그룹 | 비교 | 보는 것 |
|---|---|---|
| `window` | A1 Llama · A2 MiMo: full ↔ sliding(창 8·16·32), sink on/off, full 층 위치 | 창 밖 키 회상이 창 크기에서 꺾이는가, sink 확률은 언제 오르는가 |
| `dsv4` | DeepSeek-V4: sliding 만 → +CSA / +HCA → +MoE → +mHC | 압축 KV 가 창 밖 회상을 얼마나 되살리는가 |
| `linear` | Qwen3.6 선형:full = 4:0 · 3:1 · 1:1 · 0:4, Kimi KDA 4:0 · 3:1 | 선형 층 비율과 회상 |
| `capacity` | GDN · KDA · 3:1 하이브리드 · full × 짝 8·16·32·64 | 고정 크기 메모리가 키 몇 개에서 무너지는가 |
| `moe` | MoE on/off, 라우터 bias on/off | expert 쏠림, 학습 곡선 |

**진행 방식**
- 각 run 은 정확도 {{solve}} 에 닿으면 멈추고, 아니면 최대 step 까지 학습합니다 (기본 20,000).
- 500 step 마다 체크포인트를 남기므로 **Drive 저장을 켜 두면** 세션이 끊겨도 같은 셀을 다시 실행하면 이어집니다.
  이미 끝난 run 은 건너뜁니다. 그룹이 달라도 설정이 같은 run(예: `window` 의 `A2_mimo/full` == `moe` 의 dense)은 한 번만 학습합니다.
- 전부 돌리면 여러 시간이 걸립니다. 1번 점검 셀이 변형별 최대 소요 시간을 알려 주니, 그룹 하나씩 나눠 실행하세요.

**런타임 → 런타임 유형 변경 → T4 GPU** 선택 후 위에서부터 실행하세요.
""".replace("{solve}", "0.99"))
nb.setup()

nb.md("""
### 설정

- `USE_DRIVE = True` 면 run 폴더(`runs/mqar`)를 Drive 에 둡니다. 긴 실험은 켜 두는 것을 권합니다.
- `SEEDS`: 처음엔 `[0]` 으로 전체 그림을 보고, 차이가 애매한 비교만 seed 를 늘리세요.
- `LR` 을 바꾸면 run 폴더 이름(`..._lr0.001`)이 달라져 새로 학습합니다 (이전 결과는 그대로 남음).
""")
nb.code("""
import koreanize_matplotlib  # 그래프 한글 폰트
import matplotlib.pyplot as plt, pandas as pd, importlib
from miniarch.experiments import mqar_sweep as X
importlib.reload(X)

USE_DRIVE = True
SEEDS = [0]
LR = 1e-3
MAX_STEPS = 20000

root = "runs/mqar"
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    root = "/content/drive/MyDrive/tmp_llm_dit/runs/mqar"
SC = X.SweepConfig(root=root, lr=LR, max_steps=MAX_STEPS)
print(SC)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.precision", 3)

FIGS = {}
def show(name, fig):
    FIGS[name] = fig
    plt.show()

def report(group, curve_keys=("eval/acc",)):
    df = X.table(group, SC, SEEDS)
    display(df)
    if len(df):
        show(f"{group}_curves", X.plot_curves(group, SC, SEEDS, keys=curve_keys))
        show(f"{group}_distance", X.plot_distance(group, SC, SEEDS))
    return df
""")

nb.md("""
## 1. 실험 목록 · 점검

모든 변형을 만들고 20 step 학습해 봅니다: 오류 없음, 손실 유한, 파라미터 수, step 시간 → `max_min` = 최대 step 까지 걸릴 분 (조기 종료하면 더 짧음).
""")
nb.code("""
rows = [{"group": g, "variant": v.label(), "N": v.n_pairs, "window": v.window, "train": v.train or "", "note": v.note}
        for g, vs in X.GROUPS.items() for v in vs]
display(pd.DataFrame(rows))
""")
nb.code("""
check = pd.DataFrame(X.dry_run(list(X.GROUPS), SC))
display(check)
print("전부 정상" if check.ok.all() else f"❌ 실패: {list(check[~check.ok].variant)}")
print(f"최대 합계 ≈ {check.drop_duplicates(['variant', 'N']).max_min.sum() / 60:.1f} 시간 × seed {len(SEEDS)}개")
""")

nb.md("""
## 2. `window` — full vs sliding, sink

2층 모델이 MQAR 를 푸는 방법은 induction 회로입니다: 0층 head 가 "바로 앞 토큰(키)"을 값 위치로 옮겨 두고, 1층 head 가 질문 키와 같은 키를 찾아 그 옆 값을 복사합니다.
1층이 sliding 이면 창 밖의 키는 찾을 수 없으므로 **거리 ≈ 창 크기에서 정확도가 꺾여야** 합니다.

- `swa8 / swa16 / swa32`: 꺾이는 지점이 창 크기를 따라 움직이는가
- `swa16_full` (0층 sliding, 1층 full) vs `full_swa16` (반대): 앞 칸 head 는 창이 좁아도 되므로 앞쪽은 회상을 되살리고, 뒤쪽은 못 해야 함
- sink 확률(`sink/mass/*`): 볼 것이 없는 질의(창 안에 정답 키가 없을 때)에서 sink 로 확률을 버리는가
""")
nb.code("""
r = X.run_group("window", SC, SEEDS)
""")
nb.code("""
df_window = report("window", curve_keys=("eval/acc", "attn/entropy/L1"))
sink_runs = [(v, s) for v, r in X.summaries("window", SC, SEEDS) for s in [r["seed"]] if "nosink" not in v.name and v.model == "A2_mimo"]
fig, ax = plt.subplots(figsize=(7, 4))
for i, (v, s) in enumerate(sink_runs):
    for k in sorted({k for h in X.read_metrics(v, SC, s) for k in h if k.startswith("sink/")}):
        pts = [(h["step"], h[k]) for h in X.read_metrics(v, SC, s) if k in h]
        ax.plot(*zip(*pts), color=f"C{i % 10}", ls="-" if "layers.1" in k else "--", label=f"{v.name} {k.split('/')[-1]}")
ax.set_title("sink 로 간 확률 (실선 1층, 점선 0층)"); ax.set_xlabel("step"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
show("window_sink", fig)
""")

nb.md("""
## 3. `dsv4` — DeepSeek-V4 스위치

창 16 만 있는 모델에 1층 압축 KV 를 더합니다.
- **CSA**: 4 토큰 → 압축 1칸, indexer 가 top-8 칸만 고름 / **HCA**: 16 토큰 → 1칸, 전부 봄
- 압축 칸 하나에 짝 2개(CSA) 또는 8개(HCA)가 섞이므로, 창 밖 회상이 되살아나더라도 full 보다 낮을 수 있습니다.
- 이어서 MoE, mHC 를 켜서 학습 안정성(손실 곡선, `train/grad_norm`)과 속도를 봅니다.
""")
nb.code("""
r = X.run_group("dsv4", SC, SEEDS)
""")
nb.code("""
df_dsv4 = report("dsv4", curve_keys=("eval/acc", "train/grad_norm"))
""")

nb.md("""
## 4. `linear` — 선형 attention 비율 (4층)

Gated DeltaNet(Qwen3.6) 과 KDA(Kimi) 는 고정 크기 상태 행렬에 키-값을 씁니다 (head 4개 × 64×64).
짝 16개는 상태 크기 안이므로 둘 다 풀 수 있어야 하고, 차이는 **학습 속도**와 **거리별 정확도**에서 나타날 것입니다.
`full4` 는 같은 모델에서 모든 층을 gated attention 으로 바꾼 기준선입니다.
""")
nb.code("""
r = X.run_group("linear", SC, SEEDS)
""")
nb.code("""
df_linear = report("linear")
""")

nb.md("""
## 5. `capacity` — 키 몇 개까지 기억하는가

짝 수 N 을 8 → 64 로 늘립니다 (길이 32 → 256). full attention 은 N 과 무관하게 풀 수 있지만,
고정 크기 메모리(GDN, KDA)는 어딘가에서 무너져야 합니다. 3:1 하이브리드는 full 층 하나로 되살아나는가?
""")
nb.code("""
r = X.run_group("capacity", SC, SEEDS)
""")
nb.code("""
df_capacity = X.table("capacity", SC, SEEDS)
display(df_capacity)
show("capacity", X.plot_capacity(SC, SEEDS))
""")

nb.md("""
## 6. `moe` — MoE, 라우터 bias

MiMo full attention 에서 1층 FFN 만 바꿉니다: dense ↔ MoE(8 expert, top-2) ↔ MoE 에서 bias 업데이트 끔.
`moe/load_cv/*` 는 expert 부하의 변동계수 (0 = 고르게 분산). bias 업데이트가 없으면 쏠림이 커지는지 봅니다.
""")
nb.code("""
r = X.run_group("moe", SC, SEEDS)
""")
nb.code("""
df_moe = X.table("moe", SC, SEEDS)
display(df_moe)
moe_runs = [(v, r["seed"]) for v, r in X.summaries("moe", SC, SEEDS)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, (v, s) in enumerate(moe_runs):
    hist = X.read_metrics(v, SC, s)
    pts = [(h["step"], h["eval/acc"]) for h in hist if "eval/acc" in h]
    axes[0].plot(*zip(*pts), color=f"C{i}", label=v.name)
    for k in sorted({k for h in hist for k in h if k.startswith("moe/load_cv")}):
        pts = [(h["step"], h[k]) for h in hist if k in h]
        axes[1].plot(*zip(*pts), color=f"C{i}", label=v.name)
axes[0].set_title("eval/acc"); axes[1].set_title("expert 부하 변동계수 (0 = 고르게)")
for ax in axes: ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
show("moe", fig)
""")

nb.md("""
## 7. TensorBoard · 결과 저장

같은 태그 이름으로 기록되므로 run 들이 한 차트에 겹쳐 보입니다. 왼쪽 아래 정규식 필터로 고르세요 (예: `A2_mimo/swa`).
""")
nb.code("""
%load_ext tensorboard
%tensorboard --logdir {SC.root}
""")
nb.code("""
import datetime, shutil, os
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
out = f"results/mqar/{stamp}"
os.makedirs(out, exist_ok=True)
all_df = X.table(list(X.GROUPS), SC, SEEDS)
all_df.to_csv(f"{out}/table.csv", index=False)
for name, fig in FIGS.items():
    fig.savefig(f"{out}/{name}.png", dpi=120, bbox_inches="tight")
for v, r in X.summaries(list(X.GROUPS), SC, SEEDS):   # 요약(거리별 정확도·곡선 포함)만 모음, 체크포인트는 제외
    dst = f"{out}/summaries/{v.model}/{v.name}_N{v.n_pairs}_seed{r['seed']}.json"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(X.run_dir(v, SC, r["seed"]), "summary.json"), dst)
print(all_df.to_string(index=False))
zip_path = shutil.make_archive(f"/content/mqar_{stamp}", "zip", "results/mqar", stamp)
try:
    from google.colab import files
    files.download(zip_path)
except Exception as e:
    print("자동 다운로드 실패, 왼쪽 파일 탭에서 받으세요:", e)
""")
nb.save(PATH)
