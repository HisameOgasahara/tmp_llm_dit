"""4단계 · MQAR 부품 실험: 같은 태스크·같은 측정·같은 크기(d_model=128)에서 스위치 하나씩 바꿔 비교.

그룹 (GROUPS[이름] = Variant 목록)
  window    A1 Llama full / A2 MiMo: full ↔ sliding(창 8·16·32), sink on/off, 층 배치(swa→full, full→swa)
  dsv4      DeepSeek-V4: sliding 만 → +CSA / +HCA → +MoE → +mHC
  linear    Qwen3.6 선형(Gated DeltaNet):full 비율 4:0 · 3:1 · 1:1 · 0:4, Kimi KDA 4:0 · 3:1(MLA)
  capacity  고정 크기 메모리가 키 몇 개에서 무너지는가: GDN·KDA·하이브리드·full × 짝 8·16·32·64
  moe       MoE on/off, 라우터 bias 업데이트 on/off (MiMo, full attention)

run 폴더 = <root>/<모델>/<변형>_N<짝 수>_lr<lr>/seed<k>
- 그룹이 달라도 같은 설정이면 같은 폴더 → 한 번만 학습 (예: window 의 full == moe 의 full)
- summary.json 이 있으면 끝난 run → 건너뜀. ckpt.pt 만 있으면 중단된 run → 이어서 학습.
  root 를 Drive 에 두면 Colab 세션이 끊겨도 다음 세션에서 이어진다.

측정
- 학습 중: eval/acc, 거리 구간별 eval/acc_d*, attention 엔트로피·거리, sink 확률, expert 부하 (trainer 자동 수집)
- 끝: 거리별 정확도(짝수 거리 하나하나), 정답 도달 step (eval/acc ≥ solve_acc 인 첫 평가)
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import torch
from torch import nn

from ..tasks.base import model_logits
from ..tasks.mqar import MQAR, MQARConfig
from ..trainer import TrainConfig, Trainer, seed_everything

D_MODEL = 128
VOCAB = 128
DISTANCE_EDGES = (8, 16, 32, 64, 128)


# ---------------------------------------------------------------- 모델 (모두 d_model=128)

def llama(vocab, **kw):
    from ..models.llama import Llama, LlamaConfig
    return Llama(LlamaConfig(**{**dict(vocab_size=vocab, d_model=D_MODEL, n_layer=2, n_head=4, n_kv_head=4, d_ff=256), **kw}))


def mimo(vocab, **kw):
    """기본: 두 층 모두 sliding 형식(같은 파라미터) + use_swa 로 창을 켜고 끈다 → 창 하나만 다른 비교."""
    from ..models.mimo_v26 import SWA, MiMoV26, MiMoV26Config
    base = dict(vocab_size=vocab, d_model=D_MODEL, n_layer=2, layer_types=[SWA, SWA], d_ff=256, window=16,
                use_moe=False)
    return MiMoV26(MiMoV26Config(**{**base, **kw}))


def dsv4(vocab, **kw):
    from ..models.deepseek_v4 import SLIDING, DeepseekV4, DeepseekV4Config
    base = dict(vocab_size=vocab, d_model=D_MODEL, n_layer=2, layer_types=[SLIDING, SLIDING], window=16,
                use_moe=False, use_mhc=False)
    return DeepseekV4(DeepseekV4Config(**{**base, **kw}))


def qwen36(vocab, **kw):
    from ..models.qwen36 import Qwen36, Qwen36Config
    return Qwen36(Qwen36Config(**{**dict(vocab_size=vocab, d_model=D_MODEL, n_layer=4, d_ff=256), **kw}))


def kimi(vocab, **kw):
    """KDA / MLA 만 비교: MoE·Attention Residuals 는 끈다."""
    from ..models.kimi_k3 import KimiK3, KimiK3Config
    base = dict(vocab_size=vocab, d_model=D_MODEL, n_layer=4, d_ff=256, use_moe=False, use_attn_res=False)
    return KimiK3(KimiK3Config(**{**base, **kw}))


# ---------------------------------------------------------------- 변형 목록

@dataclass
class Variant:
    model: str                              # 모델 이름 (run 폴더 첫 단계)
    name: str                               # 변형 이름
    build: Callable[[int], nn.Module]       # vocab → 모델
    n_pairs: int = 16
    window: int | None = None               # 그림에 세로선으로 표시할 sliding window
    train: dict = field(default_factory=dict)   # TrainConfig 덮어쓰기 (예: bias_update_rate=0)
    note: str = ""

    def label(self) -> str:
        return f"{self.model}/{self.name}"


def _window_group():
    from ..models.mimo_v26 import FULL, SWA
    v = [Variant("A1_llama", "full", lambda V: llama(V), note="기준선: RoPE + full attention, sink 없음")]
    v.append(Variant("A2_mimo", "full", lambda V: mimo(V, use_swa=False), note="창 끔 (sink 있음)"))
    v.append(Variant("A2_mimo", "full_nosink", lambda V: mimo(V, use_swa=False, use_sink=False)))
    for w in (8, 16, 32):
        v.append(Variant("A2_mimo", f"swa{w}", lambda V, w=w: mimo(V, window=w), window=w))
    v.append(Variant("A2_mimo", "swa16_nosink", lambda V: mimo(V, use_sink=False), window=16))
    v.append(Variant("A2_mimo", "swa16_full", lambda V: mimo(V, layer_types=[SWA, FULL]), window=16,
                     note="0층 sliding, 1층 full (실제 MiMo 의 full 층: sink 없음, theta 1e7)"))
    v.append(Variant("A2_mimo", "full_swa16", lambda V: mimo(V, layer_types=[FULL, SWA]), window=16,
                     note="0층 full, 1층 sliding"))
    return v


def _dsv4_group():
    from ..models.deepseek_v4 import CSA, HCA, SLIDING
    S = SLIDING
    return [
        Variant("A4_dsv4", "sliding", lambda V: dsv4(V), window=16, note="창 16 만"),
        Variant("A4_dsv4", "csa", lambda V: dsv4(V, layer_types=[S, CSA]), window=16, note="1층: 창 + 4:1 압축 KV(indexer top-k)"),
        Variant("A4_dsv4", "hca", lambda V: dsv4(V, layer_types=[S, HCA]), window=16, note="1층: 창 + 16:1 압축 KV(전부)"),
        Variant("A4_dsv4", "csa_moe", lambda V: dsv4(V, layer_types=[S, CSA], use_moe=True), window=16),
        Variant("A4_dsv4", "csa_moe_mhc", lambda V: dsv4(V, layer_types=[S, CSA], use_moe=True, use_mhc=True), window=16),
    ]


LIN, FULL_Q = "linear_attention", "full_attention"


def _linear_group(n_pairs: int = 16, tag: str = ""):
    from ..models.kimi_k3 import KDA, MLA
    L, F = LIN, FULL_Q
    q = lambda name, lt: Variant("A6a_qwen36", name + tag, lambda V: qwen36(V, layer_types=lt), n_pairs=n_pairs)
    k = lambda name, lt: Variant("A6b_kimi", name + tag, lambda V: kimi(V, layer_types=lt), n_pairs=n_pairs)
    return [q("lin4", [L, L, L, L]), q("lin3full1", [L, L, L, F]), q("lin1full1", [L, F, L, F]), q("full4", [F, F, F, F]),
            k("kda4", [KDA] * 4), k("kda3mla1", [KDA, KDA, KDA, MLA])]


def _capacity_group(pairs=(8, 16, 32, 64)):
    keep = {"lin4", "lin3full1", "full4", "kda4"}
    return [v for n in pairs for v in _linear_group(n) if v.name in keep]


def _moe_group():
    return [
        Variant("A2_mimo", "full", lambda V: mimo(V, use_swa=False), note="dense (window 그룹과 같은 run)"),
        Variant("A2_mimo", "full_moe", lambda V: mimo(V, use_swa=False, use_moe=True), note="1층 MoE 8 expert top-2, bias 업데이트"),
        Variant("A2_mimo", "full_moe_nobias", lambda V: mimo(V, use_swa=False, use_moe=True),
                train={"bias_update_rate": 0.0}, note="bias 업데이트 끔"),
    ]


GROUPS: dict[str, list[Variant]] = {
    "window": _window_group(),
    "dsv4": _dsv4_group(),
    "linear": _linear_group(),
    "capacity": _capacity_group(),
    "moe": _moe_group(),
}


# ---------------------------------------------------------------- 실행

@dataclass
class SweepConfig:
    root: str = "runs/mqar"
    max_steps: int = 20000
    chunk: int = 500                # 평가·체크포인트·조기 종료 판단 주기
    solve_acc: float = 0.99         # eval/acc 가 이 값에 닿으면 멈춤
    lr: float = 1e-3
    batch_size: int = 64
    warmup: int = 200
    weight_decay: float = 0.1
    eval_batches: int = 4


def task_for(v: Variant) -> MQAR:
    return MQAR(MQARConfig(vocab_size=VOCAB, n_pairs=v.n_pairs, distance_edges=DISTANCE_EDGES))


def train_config(v: Variant, sc: SweepConfig, seed: int) -> TrainConfig:
    kw = dict(steps=sc.max_steps, batch_size=sc.batch_size, lr=sc.lr, warmup=sc.warmup, weight_decay=sc.weight_decay,
              eval_every=sc.chunk, eval_batches=sc.eval_batches, log_every=100, mi_every=sc.chunk, seed=seed)
    kw.update(v.train)
    kw["run_dir"] = run_dir(v, sc, seed, lr=kw["lr"])
    return TrainConfig(**kw)


def run_dir(v: Variant, sc: SweepConfig, seed: int, lr: float | None = None) -> str:
    lr = v.train.get("lr", sc.lr) if lr is None else lr
    return os.path.join(sc.root, v.model, f"{v.name}_N{v.n_pairs}_lr{lr:g}", f"seed{seed}")


def build(v: Variant, seed: int) -> nn.Module:
    seed_everything(seed)                       # 초기 가중치도 seed 로 고정
    return v.build(VOCAB).float()


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def distance_profile(model: nn.Module, task: MQAR, device: str, n_batches: int = 8, batch_size: int = 256,
                     seed: int = 999) -> dict[int, float]:
    """거리(질문 위치 − 키 위치)마다 정확도. 학습·평가 배치와 다른 seed 흐름."""
    was = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    N, M = task.cfg.n_pairs, task.n_q
    qpos = 2 * N + 2 * torch.arange(M)
    hit = torch.zeros(4 * N + 1)
    tot = torch.zeros(4 * N + 1)
    for _ in range(n_batches):
        b = task.batch("eval", g, batch_size)
        logits = model_logits(model, b["input_ids"].to(device))
        ok = (logits[:, qpos].argmax(-1).cpu() == b["targets"][:, qpos]).float().flatten()
        d = b["distance"].flatten()
        hit.index_add_(0, d, ok)
        tot.index_add_(0, d, torch.ones_like(ok))
    model.train(was)
    return {int(i): (hit[i] / tot[i]).item() for i in torch.nonzero(tot).flatten()}


def _truncate_jsonl(path: str, max_step: int) -> None:
    """중단된 run 을 이어갈 때, 마지막 체크포인트 이후에 기록된 줄을 지운다 (이어서 다시 기록되므로)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        keep = [l for l in f if l.strip() and json.loads(l)["step"] <= max_step]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(keep)


def load_summary(v: Variant, sc: SweepConfig, seed: int) -> dict | None:
    p = os.path.join(run_dir(v, sc, seed), "summary.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def run_variant(v: Variant, sc: SweepConfig, seed: int, device: str | None = None, progress: bool = True) -> dict:
    """한 run 을 끝까지 (이미 끝났으면 저장된 요약을 그대로). chunk 마다 체크포인트 → 끊겨도 이어서."""
    done = load_summary(v, sc, seed)
    if done is not None:
        return done
    task = task_for(v)
    cfg = train_config(v, sc, seed)
    ckpt = os.path.join(cfg.run_dir, "ckpt.pt")
    state_path = os.path.join(cfg.run_dir, "progress.json")
    resume = os.path.exists(ckpt)
    if not resume and os.path.exists(cfg.run_dir):
        shutil.rmtree(cfg.run_dir)              # 첫 체크포인트 전에 끊긴 run: 처음부터
    tr = Trainer(build(v, seed), task, cfg, device)
    state = {"elapsed": 0.0, "curve": []}
    if resume:
        tr.load()
        _truncate_jsonl(os.path.join(cfg.run_dir, "metrics.jsonl"), tr.step)
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
        state["curve"] = [c for c in state["curve"] if c[0] <= tr.step]
        print(f"  이어서: step {tr.step}")

    solved_step = next((s for s, a in state["curve"] if a >= sc.solve_acc), None)
    while tr.step < sc.max_steps and solved_step is None:
        t0 = time.time()
        tr.train(min(sc.chunk, sc.max_steps - tr.step), progress=False)   # 끝에서 ckpt 저장
        state["elapsed"] += time.time() - t0
        acc = next((h["eval/acc"] for h in reversed(tr.history) if "eval/acc" in h), None)
        if acc is None:
            acc = tr.evaluate()["eval/acc"]
        state["curve"].append([tr.step, round(acc, 4)])
        if acc >= sc.solve_acc:
            solved_step = tr.step
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        if progress:
            print(f"\r  step {tr.step:>6}  acc {acc:.3f}  ({state['elapsed'] / 60:.1f}분)", end="", flush=True)
    if progress:
        print()

    final = tr.evaluate()
    summary = {
        "model": v.model, "name": v.name, "n_pairs": v.n_pairs, "seed": seed, "window": v.window, "note": v.note,
        "params": n_params(tr.model), "steps": tr.step, "solved_step": solved_step,
        "elapsed_min": state["elapsed"] / 60, "ms_per_step": 1000 * state["elapsed"] / max(tr.step, 1),
        "final": final, "mi": tr.mi_metrics(), "curve": state["curve"],
        "distance_acc": distance_profile(tr.model, task, tr.device),
        "config": asdict(cfg),
    }
    with open(os.path.join(cfg.run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    if tr.writer:
        tr.writer.close()
    return summary


def run_group(names, sc: SweepConfig, seeds=(0,), device: str | None = None) -> list[dict]:
    """여러 그룹을 순서대로. 같은 run 은 한 번만. 한 run 이 실패해도 나머지는 계속."""
    import traceback
    seen, out = set(), []
    for g in ([names] if isinstance(names, str) else names):
        for v in GROUPS[g]:
            for s in seeds:
                key = run_dir(v, sc, s)
                if key in seen:
                    continue
                seen.add(key)
                tag = f"[{g}] {v.label()} N{v.n_pairs} seed{s}"
                if load_summary(v, sc, s) is not None:
                    print(f"✔ {tag} (이미 끝남)")
                    out.append(load_summary(v, sc, s))
                    continue
                print(f"▶ {tag}")
                try:
                    r = run_variant(v, sc, s, device)
                    out.append(r)
                    print(f"  → acc {r['final']['eval/acc']:.3f}, 도달 step {r['solved_step']}, {r['elapsed_min']:.1f}분")
                except Exception:
                    traceback.print_exc(limit=6)
                    print(f"💥 {tag}")
    return out


def dry_run(names, sc: SweepConfig, device: str | None = None, steps: int = 20) -> list[dict]:
    """학습 전 점검: 모든 변형을 만들고 몇 step 돌려 오류·파라미터 수·step 시간 → 최대 소요 시간 추정."""
    import traceback
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows, seen = [], set()
    for g in ([names] if isinstance(names, str) else names):
        for v in GROUPS[g]:
            key = (v.model, v.name, v.n_pairs)
            if key in seen:
                continue
            seen.add(key)
            row = {"group": g, "variant": v.label(), "N": v.n_pairs}
            try:
                task = task_for(v)
                m = build(v, 0).to(device)
                opt = torch.optim.AdamW(m.parameters(), lr=sc.lr)
                gen = torch.Generator().manual_seed(0)
                for i in range(steps + 3):
                    if i == 3:                              # 앞 3 step 은 워밍업 (cuda 초기화 등)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.time()
                    b = {k: x.to(device) if torch.is_tensor(x) else x for k, x in task.batch("train", gen, sc.batch_size).items()}
                    loss, _ = task.loss(m, b)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                ms = 1000 * (time.time() - t0) / steps
                row.update(ok=bool(torch.isfinite(loss)), params=n_params(m), ms_per_step=round(ms, 1),
                           max_min=round(ms * sc.max_steps / 60000, 1))
                del m, opt
            except Exception:
                traceback.print_exc(limit=4)
                row.update(ok=False)
            rows.append(row)
    return rows


# ---------------------------------------------------------------- 집계 · 그림

def summaries(names, sc: SweepConfig, seeds=(0,)) -> list[tuple[Variant, dict]]:
    out = []
    for g in ([names] if isinstance(names, str) else names):
        for v in GROUPS[g]:
            for s in seeds:
                r = load_summary(v, sc, s)
                if r is not None:
                    out.append((v, r))
    return out


def table(names, sc: SweepConfig, seeds=(0,)):
    """변형마다 seed 평균: 최종 정확도, 거리 구간별 정확도, 도달 step, 파라미터, 시간."""
    import pandas as pd
    rows = []
    for v, r in summaries(names, sc, seeds):
        row = {"variant": v.label(), "N": v.n_pairs, "seed": r["seed"], "params": r["params"],
               "acc": r["final"]["eval/acc"],
               "solved_step": float("nan") if r["solved_step"] is None else r["solved_step"], "steps": r["steps"], "min": r["elapsed_min"]}
        row.update({k.replace("eval/", ""): x for k, x in r["final"].items() if k.startswith("eval/acc_d")})
        sinks = [x for k, x in r["mi"].items() if k.startswith("sink/")]
        if sinks:
            row["sink_mass"] = sum(sinks) / len(sinks)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("variant", "N", "seed")]
    agg = df.groupby(["variant", "N"], sort=False)[num].mean(numeric_only=True)
    agg["seeds"] = df.groupby(["variant", "N"], sort=False).size()
    return agg.reset_index()


def read_metrics(v: Variant, sc: SweepConfig, seed: int) -> list[dict]:
    p = os.path.join(run_dir(v, sc, seed), "metrics.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def plot_curves(names, sc: SweepConfig, seeds=(0,), keys=("eval/acc",)):
    """keys 마다 그림 하나: 변형별 곡선 (seed 는 같은 색, 얇게)."""
    import matplotlib.pyplot as plt
    vs = [(v, r["seed"]) for v, r in summaries(names, sc, seeds)]
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4), squeeze=False)
    colors = {}
    for v, s in vs:
        c = colors.setdefault(v.label() + f"N{v.n_pairs}", f"C{len(colors) % 10}")
        hist = read_metrics(v, sc, s)
        for ax, k in zip(axes[0], keys):
            pts = [(h["step"], h[k]) for h in hist if k in h]
            if pts:
                ax.plot(*zip(*pts), color=c, lw=1.2, label=f"{v.label()} N{v.n_pairs}" if s == seeds[0] else None)
    for ax, k in zip(axes[0], keys):
        ax.set_title(k)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_distance(names, sc: SweepConfig, seeds=(0,)):
    """거리별 정확도 (seed 평균). sliding window 는 점선."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    acc: dict[str, dict] = {}
    meta = {}
    for v, r in summaries(names, sc, seeds):
        key = f"{v.label()} N{v.n_pairs}"
        meta[key] = v
        for d, a in r["distance_acc"].items():
            acc.setdefault(key, {}).setdefault(int(d), []).append(a)
    windows = set()
    for i, (key, dd) in enumerate(acc.items()):
        ds = sorted(dd)
        ax.plot(ds, [sum(dd[d]) / len(dd[d]) for d in ds], marker=".", color=f"C{i % 10}", label=key)
        if meta[key].window:
            windows.add(meta[key].window)
    for w in sorted(windows):
        ax.axvline(w, color="gray", ls=":", lw=1)
        ax.text(w, 1.02, f"W={w}", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("거리 (질문 위치 − 키 위치)")
    ax.set_ylabel("정확도")
    ax.set_ylim(-0.02, 1.08)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def plot_capacity(sc: SweepConfig, seeds=(0,)):
    """짝 수(N) 별 최종 정확도: 고정 크기 메모리(선형)가 어디서 무너지는가."""
    import matplotlib.pyplot as plt
    pts: dict[str, dict] = {}
    for v, r in summaries("capacity", sc, seeds):
        pts.setdefault(v.label(), {}).setdefault(v.n_pairs, []).append(r["final"]["eval/acc"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (k, d) in enumerate(pts.items()):
        ns = sorted(d)
        ax.plot(ns, [sum(d[n]) / len(d[n]) for n in ns], marker="o", color=f"C{i % 10}", label=k)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("키-값 짝 수 N")
    ax.set_ylabel("최종 정확도")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
