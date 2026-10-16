"""⑤ 학습 코드 검증 — 학습 루프(miniarch/trainer.py)와 태스크(miniarch/tasks)가 맞게 짜였는지.

각 함수는 [(점검 이름, 통과 여부, 값)] 을 돌려준다. 모델과 무관한 점검은 작은 Llama 로 한다.
"""
from __future__ import annotations

import glob
import json
import os
import shutil

import torch

from ..models.llama import Llama, LlamaConfig
from ..tasks.base import IGNORE
from ..tasks.lm_tasks import ModAddTask
from ..tasks.mqar import MQAR, MQARConfig
from ..trainer import Trainer, TrainConfig, lr_at, param_groups

Result = list[tuple[str, bool, float]]


def tiny_llama(vocab: int = 128, seed: int = 0) -> Llama:
    torch.manual_seed(seed)
    return Llama(LlamaConfig(vocab_size=vocab, d_model=64, n_layer=2, n_head=4, n_kv_head=2, d_ff=128))


# ---------------------------------------------------------------- 태스크 / 데이터

def test_mqar_data(n_pairs: int = 8) -> Result:
    task = MQAR(MQARConfig(vocab_size=128, n_pairs=n_pairs))
    b = task.batch("train", torch.Generator().manual_seed(0), 256)
    ids, tgt, dist = b["input_ids"], b["targets"], b["distance"]
    N, M, H = n_pairs, task.n_q, task.half
    keys, vals = ids[:, 0:2 * N:2], ids[:, 1:2 * N:2]
    q, a = ids[:, 2 * N::2], ids[:, 2 * N + 1::2]
    res: Result = []
    res.append(("MQAR 키는 앞 절반·값은 뒤 절반", bool((keys < H).all() and (vals >= H).all()), 0.0))
    res.append(("MQAR 문맥 안 키 중복 없음", all(len(set(r.tolist())) == N for r in keys), 0.0))
    # 질문 qi 의 정답 = 문맥에서 그 키 바로 뒤의 값
    ok = True
    for r in range(len(ids)):
        d = dict(zip(keys[r].tolist(), vals[r].tolist()))
        ok &= all(d[k] == v for k, v in zip(q[r].tolist(), a[r].tolist()))
    res.append(("MQAR 정답 = 짝지어진 값", ok, 0.0))
    # 손실 위치 = 질문 위치뿐, targets[t] = input[t+1] (한 칸 밀림)
    m = tgt != IGNORE
    pos = m.nonzero()[:, 1].unique().tolist()
    res.append(("MQAR 손실은 질문 위치에만", pos == list(range(2 * N, 2 * N + 2 * M, 2)), 0.0))
    shifted = ids[:, 1:][m[:, :-1]]
    res.append(("MQAR 정답 = 다음 칸 입력 (shift 일치)", bool((tgt[:, :-1][m[:, :-1]] == shifted).all()), 0.0))
    # 거리 = 질문 위치 − 키 위치 이고, 그 위치에 정말 그 키가 있다
    qpos = torch.arange(2 * N, 2 * N + 2 * M, 2)
    kpos = qpos[None] - dist
    res.append(("MQAR 거리 표시 일치", bool((ids.gather(1, kpos) == q).all()), 0.0))
    return res


def test_split_disjoint() -> Result:
    res: Result = []
    t = ModAddTask(p=31, train_frac=0.5)
    tr = {tuple(r) for r in t.data.train_x.tolist()}
    te = {tuple(r) for r in t.data.test_x.tolist()}
    res.append(("modadd 학습/평가 (a,b) 겹침 없음", not (tr & te), float(len(tr & te))))
    task = MQAR(MQARConfig(vocab_size=128, n_pairs=8))
    ev = {tuple(r) for b in task.eval_batches(4, 64) for r in b["input_ids"].tolist()}
    g = torch.Generator().manual_seed(0)
    trs = {tuple(r) for _ in range(50) for r in task.batch("train", g, 64)["input_ids"].tolist()}
    res.append(("MQAR 평가 시퀀스가 학습 스트림에 없음", not (ev & trs), float(len(ev & trs))))
    return res


def test_loss_masking(device: str) -> Result:
    """손실 = 표시된 위치만의 평균 CE (직접 계산과 일치), 정답을 바꾸면 손실이 바뀜."""
    import torch.nn.functional as F
    task = MQAR(MQARConfig(vocab_size=128, n_pairs=8))
    m = tiny_llama().to(device).eval()
    b = {k: v.to(device) for k, v in task.batch("train", torch.Generator().manual_seed(1), 16).items()}
    with torch.no_grad():
        base = task.loss(m, b)[0].item()
        logits = m(b["input_ids"]).logits.float()
        mask = b["targets"] != IGNORE
        manual = F.cross_entropy(logits[mask], b["targets"][mask]).item()
        t2 = b["targets"].clone()
        t2[mask] = (t2[mask] + 1) % 128
        changed = task.loss(m, dict(b, targets=t2))[0].item()
    return [("손실 = 표시된 위치만의 평균 CE", abs(manual - base) < 1e-5, abs(manual - base)),
            ("손실: 정답을 바꾸면 손실이 바뀜", abs(changed - base) > 1e-6, abs(changed - base))]


def test_no_answer_leak(device: str) -> Result:
    """질문 위치의 출력은 그 뒤 입력(정답 칸 포함)을 바꿔도 같아야 한다 → 정답이 입력으로 새지 않는다."""
    task = MQAR(MQARConfig(vocab_size=128, n_pairs=8))
    m = tiny_llama().to(device).eval()
    b = task.batch("train", torch.Generator().manual_seed(2), 8)
    ids = b["input_ids"].to(device)
    N = task.cfg.n_pairs
    qpos = 2 * N
    ids2 = ids.clone()
    ids2[:, qpos + 1:] = torch.randint(0, 128, ids2[:, qpos + 1:].shape, device=device)
    with torch.no_grad():
        d = (m(ids).logits[:, qpos] - m(ids2).logits[:, qpos]).abs().max().item()
    return [("정답 누수 없음 (질문 위치 출력이 뒤 칸에 무관)", d < 1e-5, d)]


# ---------------------------------------------------------------- 학습 루프

def _cfg(run_dir, **kw) -> TrainConfig:
    base = dict(steps=20, batch_size=32, lr=3e-3, warmup=5, eval_every=10, eval_batches=2, log_every=5,
                mi_every=10, seed=0, run_dir=run_dir, tensorboard=True)
    base.update(kw)
    return TrainConfig(**base)


def test_schedule_and_groups() -> Result:
    cfg = TrainConfig(steps=100, lr=1.0, warmup=10, min_lr_ratio=0.1)
    lrs = [lr_at(s, cfg) for s in range(100)]
    res: Result = [("lr warmup 선형 증가", abs(lrs[0] - 0.1) < 1e-9 and abs(lrs[9] - 1.0) < 1e-9, lrs[0]),
                   ("lr cosine 단조 감소 → min_lr", all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:])) and
                    abs(lr_at(100, cfg) - 0.1) < 1e-9, lr_at(100, cfg))]
    m = tiny_llama()
    groups = {g["name"]: g for g in param_groups(m, 0.1, set())}
    decayed = {id(p) for p in groups["decay"]["params"]}
    names_nd = [n for n, p in m.named_parameters() if id(p) not in decayed]
    ok = all(p.dim() < 2 for n, p in m.named_parameters() if id(p) not in decayed) and \
         all(p.dim() >= 2 for n, p in m.named_parameters() if id(p) in decayed)
    res.append(("weight decay 는 2차원 이상만 (norm·bias 제외)", ok and len(names_nd) > 0, float(len(names_nd))))
    return res


def test_eval_purity(device: str, tmp: str) -> Result:
    task = MQAR(MQARConfig(vocab_size=128, n_pairs=8))
    tr = Trainer(tiny_llama(), task, _cfg(tmp, tensorboard=False), device)
    tr.train(5, progress=False)
    sd = {k: v.clone() for k, v in tr.model.state_dict().items()}
    gen_state, rng = tr.gen.get_state(), torch.get_rng_state()
    tr.model.train()
    e1 = tr.evaluate()
    e2 = tr.evaluate()
    same_w = all(torch.equal(sd[k], v) for k, v in tr.model.state_dict().items())
    return [("평가가 가중치를 안 바꿈", same_w, 0.0),
            ("평가가 학습 배치 RNG 를 안 건드림", torch.equal(gen_state, tr.gen.get_state()) and
             torch.equal(rng, torch.get_rng_state()), 0.0),
            ("평가 후 train 모드 복원", tr.model.training, 0.0),
            ("평가는 결정적 (두 번 같은 값)", e1 == e2, 0.0)]


def _make(model_fn, seed: int = 0):
    torch.manual_seed(seed)          # 모델 초기값도 seed 로 고정 (재현·재개 비교용)
    return model_fn()


def _run(model_fn, task, run_dir, device, n_steps, **kw):
    """kw 는 TrainConfig 값 (steps 를 주면 lr 스케줄 길이, n_steps 는 이번에 돌릴 step 수)."""
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    tr = Trainer(_make(model_fn), task, _cfg(run_dir, **kw), device)
    tr.train(n_steps, progress=False)
    return tr


def test_reproducible_and_resume(model_fn, task, device: str, tmp: str, tag: str = "", atol: float = 1e-5) -> Result:
    """같은 seed 두 번 → 같은 손실 곡선. 10 step + 저장 + 새로 만들어 불러오기 + 10 step → 20 step 한 번에 한 것과 같음."""
    a = _run(model_fn, task, f"{tmp}/a", device, 20)
    b = _run(model_fn, task, f"{tmp}/b", device, 20)
    la = [h["train/loss"] for h in a.history]
    lb = [h["train/loss"] for h in b.history]
    d_rep = max(abs(x - y) for x, y in zip(la, lb))
    c1 = _run(model_fn, task, f"{tmp}/c", device, 10)
    c1.save()
    c2 = Trainer(_make(model_fn), task, _cfg(f"{tmp}/c"), device)
    c2.load()
    c2.history = []
    c2.train(10, progress=False)
    lc = [h["train/loss"] for h in c2.history]
    tail = [h["train/loss"] for h in a.history if h["step"] > 10]
    d_res = max(abs(x - y) for x, y in zip(tail, lc)) if lc else float("inf")
    d_w = max((p - q).abs().max().item() for p, q in zip(a.model.state_dict().values(), c2.model.state_dict().values())
              if p.is_floating_point())
    return [(f"{tag}같은 seed → 같은 손실 곡선", d_rep <= atol, d_rep),
            (f"{tag}체크포인트 재개 == 한 번에 학습 (손실)", d_res <= atol and len(lc) == len(tail), d_res),
            (f"{tag}체크포인트 재개 == 한 번에 학습 (가중치)", d_w <= max(atol, 1e-5), d_w)]


def test_logging(device: str, tmp: str) -> Result:
    task = MQAR(MQARConfig(vocab_size=128, n_pairs=8))
    tr = _run(tiny_llama, task, f"{tmp}/log", device, 20)
    recs = [json.loads(l) for l in open(f"{tmp}/log/metrics.jsonl", encoding="utf-8")]
    keys = set().union(*recs)
    need = {"train/loss", "train/grad_norm", "train/lr", "eval/acc", "eval/ce"}
    has_attn = any(k.startswith("attn/entropy/") for k in keys)
    ev = glob.glob(f"{tmp}/log/events.out.tfevents.*")
    gn = [r["train/grad_norm"] for r in recs if "train/grad_norm" in r]
    return [("기록: 필수 항목 (loss, grad_norm, lr, eval)", need <= keys, float(len(need - keys))),
            ("기록: attention MI 지표 자동 수집", has_attn, 0.0),
            ("기록: TensorBoard 이벤트 파일 생성", len(ev) > 0, float(len(ev))),
            ("기록: 기울기 norm 유한", all(torch.isfinite(torch.tensor(gn))), max(gn) if gn else 0.0)]


def test_learns_mqar(device: str, tmp: str, max_steps: int = 10000, chunk: int = 500, threshold: float = 0.9) -> Result:
    """알려진 결과: attention 모델은 MQAR(짝 8개)을 거의 완벽히 푼다 (Zoology). 학습 루프 전체가 맞물리는지 확인.

    induction 회로가 생기기 전까지 정확도가 한참 정체하다가 오르므로, 500 step 마다 평가해서 threshold 에 닿으면 멈춘다
    (최대 max_steps, Zoology 는 수만 step 학습). 정확도 추이를 함께 출력한다.
    """
    task = MQAR(MQARConfig(vocab_size=128, n_pairs=8))
    tr = _run(lambda: tiny_llama(seed=0), task, f"{tmp}/learn", device, 0,
              steps=max_steps, batch_size=64, lr=1e-3, warmup=200, eval_every=chunk, eval_batches=4, log_every=100,
              mi_every=chunk, weight_decay=0.1)
    curve = []
    while tr.step < max_steps:
        tr.train(chunk, progress=False)
        acc = tr.evaluate()["eval/acc"]
        curve.append((tr.step, round(acc, 3)))
        if acc >= threshold:
            break
    print("  step·정확도:", curve)
    return [(f"Llama 가 MQAR(짝 8) 학습 ≥ {threshold} ({tr.step} step)", acc >= threshold, acc)]


def model_smoke(name: str, model_fn, device: str, tmp: str, steps: int = 30) -> Result:
    """검증된 모델 하나를 공용 학습 루프로 MQAR 30 step: NaN 없음, 손실 감소, 라우터 bias 동작, 재개 일치."""
    task = MQAR(MQARConfig(vocab_size=128, n_pairs=8))
    tr = _run(model_fn, task, f"{tmp}/{name}", device, steps, steps=steps, lr=3e-3, warmup=5,
              eval_every=steps, log_every=5, mi_every=steps)
    losses = [h["train/loss"] for h in tr.history]
    first, last = losses[0], min(losses[-2:])
    res: Result = [(f"{name}: 손실 유한", all(torch.isfinite(torch.tensor(losses))), last),
                   (f"{name}: 손실 감소", last < first, last / first)]
    if tr.balancer.routers:
        cv = [v for h in tr.history for k, v in h.items() if k.startswith("moe/load_cv/")]
        res.append((f"{name}: 라우터 bias 업데이트 동작 ({len(tr.balancer.routers)}개)", len(cv) > 0, cv[-1] if cv else 0.0))
    res += test_reproducible_and_resume(model_fn, task, device, f"{tmp}/{name}_rr", tag=f"{name}: ", atol=1e-4)
    return res
