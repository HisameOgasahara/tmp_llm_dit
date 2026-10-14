from nbgen import NB, SAVE_RESULTS, DRIVE

PATH = "notebooks/01_modadd.ipynb"
nb = NB()

nb.md(f"""
# 01 · Modular addition: grokking과 attention 해부

{nb.badge(PATH)}

`[a, b, =]`를 보고 `(a + b) mod 113`을 맞히는 태스크입니다. 전체 12,769개 조합 중 일부만 학습에 씁니다.

**grokking이란**
1. 학습 데이터는 금방 100% 맞힙니다(외우기). 이때 테스트 정확도는 낮습니다.
2. 한참 동안 아무 일도 없는 것처럼 보이다가, 어느 순간 **테스트 정확도가 갑자기 100%로** 뜁니다(일반화).
3. 그 시점에 숫자 임베딩이 **소수의 푸리에 주파수로 정리**됩니다. 덧셈을 원 위의 회전으로 계산하는 알고리즘을 찾았다는 신호입니다.

**이전 실행에서 알게 된 것**: 표준 GPT-2와 Llama(LayerNorm + 임베딩 공유)는 외우자마자 바로 일반화해서 grokking 지연이 없었습니다.
grokking이 잘 알려진 원 실험(Nanda et al. 2023)은 **LayerNorm이 없고 출력층이 분리된** 모델을 씁니다. 그래서 이번엔 두 부품을 하나씩 꺼 봅니다.

| 그룹 | 실행 | 바꾼 것 |
|---|---|---|
| A. 구조 비교 | GPT-2 / Llama / Llama wd=0 | 아키텍처, weight decay |
| B. grokking ablation | GPT-2 −LN / GPT-2 −tie / GPT-2 −LN −tie (원 논문) / 원 논문 30%→20% | LayerNorm, 임베딩 공유, 학습 데이터 비율 |

1층, d_model 128, head 4개, 실행 7개. T4 기준 전체 약 20분입니다.
""")
nb.setup()

nb.md("## 1. 설정")
nb.code("""
import matplotlib.pyplot as plt
import koreanize_matplotlib  # 그래프 한글 폰트
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
import torch
from miniarch.train import seed_all, get_device, count_params
from miniarch.tasks import modadd
from miniarch.analysis.fourier import freq_energy, fourier_2d

PREFIX = "01_modadd"
dev = get_device()
P, STEPS = 113, 30000                 # 빨리 훑어보려면 STEPS 를 줄이세요
D_MODEL, N_HEAD, N_LAYER = 128, 4, 1

RUNS = {
    # 이름: 구조, weight decay, LayerNorm, 임베딩 공유, 학습 데이터 비율, 그룹
    "GPT-2":              dict(kind="gpt2",  wd=1.0, ln=True,  tie=True,  frac=0.3, group="A"),
    "Llama":              dict(kind="llama", wd=1.0, ln=True,  tie=True,  frac=0.3, group="A"),
    "Llama, wd=0":        dict(kind="llama", wd=0.0, ln=True,  tie=True,  frac=0.3, group="A"),
    "GPT-2 −LN":          dict(kind="gpt2",  wd=1.0, ln=False, tie=True,  frac=0.3, group="B"),
    "GPT-2 −tie":         dict(kind="gpt2",  wd=1.0, ln=True,  tie=False, frac=0.3, group="B"),
    "GPT-2 −LN −tie":     dict(kind="gpt2",  wd=1.0, ln=False, tie=False, frac=0.3, group="B"),
    "GPT-2 −LN −tie 20%": dict(kind="gpt2",  wd=1.0, ln=False, tie=False, frac=0.2, group="B"),
}
COLORS = dict(zip(RUNS, ["tab:blue", "tab:orange", "tab:gray", "tab:green", "tab:purple", "tab:red", "tab:brown"]))
datasets = {f: modadd.make_data(P, f, seed=0, device=dev) for f in {c["frac"] for c in RUNS.values()}}
for f, d in sorted(datasets.items()):
    print(f"학습 비율 {f:.0%}: 학습 {len(d.train_y):,}개 / 테스트 {len(d.test_y):,}개")
""")

nb.md("""
## 2. 학습

진행 표시줄의 `train`/`test`는 정확도, `wn`은 전체 가중치 크기입니다.
""")
nb.code("""
models, hists = {}, {}
for name, c in RUNS.items():
    seed_all(0)
    m = modadd.build_model(c["kind"], P, D_MODEL, N_HEAD, N_LAYER, use_ln=c["ln"], tie=c["tie"]).to(dev)
    print(f"\\n▶ {name}  params={count_params(m):,}")
    hists[name] = modadd.train(m, datasets[c["frac"]], steps=STEPS, wd=c["wd"])
    models[name] = m

summary = pd.DataFrame({n: {"외우기 (train>99%)": h["memorize_step"], "일반화 (test>99%)": h["grok_step"],
                            "지연 (step)": (h["grok_step"] - h["memorize_step"]) if h["grok_step"] else None,
                            "최종 test": round(h["test_acc"][-1], 4), "최종 가중치 크기": round(h["weight_norm"][-1], 1),
                            "상위5 주파수 비율": round(h["fourier_top5"][-1], 3)} for n, h in hists.items()}).T
summary
""")

nb.md("""
## 3. 학습 곡선

- **왼쪽**: 점선이 학습, 실선이 테스트 정확도입니다. 두 선 사이 간격이 "외웠지만 아직 일반화 못 한" 구간, 즉 grokking 지연입니다.
- **가운데**: 가중치 크기. weight decay는 외운 뒤에도 가중치를 계속 줄여서 더 단순한(일반화되는) 해법 쪽으로 밉니다.
- **오른쪽**: 임베딩 에너지 중 상위 5개 주파수 비율. 올라가면 푸리에 해법으로 정리되고 있다는 뜻입니다.
- 세로 점선은 각 실행의 일반화 시점입니다.
""")
nb.code("""
def plot_curves(names, title):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for name in names:
        h, c = hists[name], COLORS[name]
        axes[0].plot(h["step"], h["train_acc"], "--", color=c, alpha=0.6)
        axes[0].plot(h["step"], h["test_acc"], "-", color=c, label=name)
        axes[1].plot(h["step"], h["weight_norm"], color=c, label=name)
        axes[2].plot(h["step"], h["fourier_top5"], color=c, label=name)
        if h["grok_step"]:
            for ax in axes:
                ax.axvline(h["grok_step"], color=c, ls=":", lw=1)
    for ax, t in zip(axes, ["정확도 (점선 학습 / 실선 테스트)", "가중치 크기", "상위 5개 주파수 에너지 비율"]):
        ax.set_xscale("symlog", linthresh=100); ax.set_xlim(0, STEPS)
        ax.set_title(t); ax.set_xlabel("step"); ax.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); plt.show()
    return fig

fig_curves_A = plot_curves([n for n, c in RUNS.items() if c["group"] == "A"], "A. 구조 비교")
fig_curves_B = plot_curves(["GPT-2"] + [n for n, c in RUNS.items() if c["group"] == "B"], "B. grokking ablation (GPT-2 기준)")
""")

nb.md("""
## 4. 임베딩의 푸리에 스펙트럼

숫자 0~112의 임베딩을 주파수별로 분해한 결과입니다.
- 뾰족한 막대 몇 개 = 모델이 고른 "핵심 주파수". 깔끔한 푸리에 해법입니다.
- 넓게 퍼져 있으면 정답은 맞혀도 해법이 덜 정리된 상태입니다.
- 실행마다 **다른 주파수**를 고르는지도 보세요.
""")
nb.code("""
fig_fourier, axes = plt.subplots(len(models), 1, figsize=(14, 1.9 * len(models)), sharex=True)
for ax, (name, m) in zip(axes, models.items()):
    e = freq_energy(modadd.embedding_weight(m).detach(), P).cpu()
    e = e / e.sum()
    ax.bar(range(1, len(e) + 1), e, color=COLORS[name])
    ax.set_title(f"{name}   상위 주파수 k = {sorted((e.topk(5).indices + 1).tolist())}", fontsize=9)
axes[-1].set_xlabel("주파수 k")
fig_fourier.tight_layout(); plt.show()
""")

nb.md("""
## 5. Attention: `=`가 `a`를 얼마나 보는가, 모든 (a, b)에 대해

이전 실행의 평균 attention map은 a와 b를 반반 본다는 것만 보여줬습니다. 평균을 내면 패턴이 지워지기 때문입니다.
그래서 이번엔 **모든 113×113 입력**에 대해 `=` 위치가 `a`에 주는 attention 확률을 격자로 그립니다(행 = a, 열 = b, 나머지는 b와 `=`가 나눠 가짐).

- **윗줄**: head별 격자. 줄무늬나 격자무늬가 보이면 attention이 숫자 값에 따라 **주기적으로** 바뀐다는 뜻입니다. 원 논문에서는 이런 주기 패턴이 나옵니다.
- **아랫줄**: 그 격자의 2D 푸리에 크기. 밝은 점이 (a 방향 주파수, b 방향 주파수)입니다. 4번에서 찾은 임베딩 핵심 주파수와 같은 위치에 점이 찍히는지 비교해 보세요.
""")
nb.code("""
GRID_MODELS = ["GPT-2", "Llama", "GPT-2 −LN −tie"]
grid_figs = {}
for name in GRID_MODELS:
    G = modadd.attention_grid(models[name], P).cpu()          # [H, p, p]
    H = G.shape[0]
    fig, axes = plt.subplots(2, H, figsize=(3.2 * H, 6.4))
    for h in range(H):
        im = axes[0, h].imshow(G[h], cmap="viridis", vmin=0, vmax=1)
        axes[0, h].set_title(f"head {h}  (평균 {G[h].mean():.2f})", fontsize=9)
        axes[0, h].set_xlabel("b"); axes[0, h].set_ylabel("a")
        F = fourier_2d(G[h])
        axes[1, h].imshow(F.log1p(), cmap="magma")
        top = torch.topk(F.flatten(), 3).indices
        axes[1, h].set_title("상위 (ka,kb): " + ", ".join(f"({i // F.shape[1]},{i % F.shape[1]})" for i in top.tolist()), fontsize=8)
        axes[1, h].set_xlabel("kb"); axes[1, h].set_ylabel("ka")
    fig.suptitle(f"{name} · '=' → a attention 격자 (위) 와 2D 푸리에 (아래)"); fig.tight_layout(); plt.show()
    grid_figs[name] = fig
""")

nb.md("""
## 6. 결과 저장

- `results/`에 요약표, 곡선 수치(json), 그림(png)이 저장됩니다.
- 출력이 담긴 노트북을 기록으로 남기려면 **파일 → GitHub에 사본 저장**을 쓰세요.
""")
nb.code(SAVE_RESULTS + """
with open(f"results/{PREFIX}_{STAMP}.json", "w") as f:
    json.dump({"config": dict(P=P, STEPS=STEPS, D_MODEL=D_MODEL, N_HEAD=N_HEAD, N_LAYER=N_LAYER, RUNS=RUNS),
               "summary": json.loads(summary.to_json(orient="index", force_ascii=False)), "history": hists}, f, ensure_ascii=False)
save_fig(fig_curves_A, "curves_A"); save_fig(fig_curves_B, "curves_B"); save_fig(fig_fourier, "fourier")
for n, f in grid_figs.items():
    save_fig(f, "attn_grid_" + n.replace(" ", "").replace("−", "no"))
print([f for f in os.listdir("results") if STAMP in f])
""" + DRIVE)

nb.save(PATH)
