"""모델별 검증 노트북 생성: miniarch/verify/stages/*.py 하나당 notebooks/verify/{STAGE}_{SLUG}.ipynb 하나.

stage 모듈의 리터럴 항목(STAGE, SLUG, TITLE, INTRO, PIP, HEADER, CAUSAL, SELF_CHECKS)을 ast 로 읽으므로 torch 가 없어도 된다.
사용: python tools/build_verify.py            (전부)
      python tools/build_verify.py a2_mimo    (하나)
"""
import ast
import os
import sys

from nbgen import NB, ROOT

STAGES_DIR = os.path.join(ROOT, "miniarch", "verify", "stages")
LITERALS = ("STAGE", "SLUG", "TITLE", "INTRO", "PIP", "HEADER", "CAUSAL", "SELF_CHECKS")


def read_spec(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    spec = {"PIP": [], "HEADER": [], "CAUSAL": False, "SELF_CHECKS": [], "INTRO": ""}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in LITERALS:
                spec[name] = ast.literal_eval(node.value)
    for k in ("STAGE", "SLUG", "TITLE"):
        if k not in spec:
            raise ValueError(f"{path}: {k} 없음")
    return spec


def verify_cells(nb, spec):
    """순전파 검증: 원본 구조, 원본 동작, 원본 동치, causal 누수, 단계 전용 점검."""

    if spec["HEADER"]:
        nb.md("""
## 원본 구조 (HF 헤더)

실제 체크포인트의 safetensors 헤더만 읽어 가중치 이름, shape, dtype을 봅니다 (다운로드 없음).
**구현 검증이 아니라 원본 구조 확인용**입니다. routed expert는 개수가 많아 숨깁니다.
""")
        calls = "\n".join(f"peek({r!r}, {p!r}, subfolder={sf!r})" for r, p, sf in spec["HEADER"])
        nb.code(f"""
def peek(repo, pattern, subfolder="", limit=60):
    print(f"\\n===== {{repo}}  /{{pattern}}/")
    try:
        cfg = V.fetch_config(repo, subfolder)
        cfg = cfg.get("text_config", cfg)
        keys = ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim"]
        print({{k: cfg[k] for k in keys if k in cfg}})
        h = V.read_model_header(repo, subfolder=subfolder, pattern=pattern)
        V.show({{k: v for k, v in h.items() if ".experts." not in k}}, limit)
    except Exception as e:
        print(f"⚠️ {{type(e).__name__}}: {{e}}")

{calls}
""")

    nb.md("""
## 원본 모델 동작 확인

축소 config로 원본을 만들고 한 번 forward 합니다. 여기서 실패하면 라이브러리 버전 문제입니다 (내 구현과 무관).
""")
    nb.code("""
V.set_deterministic(0)
ref, inputs = S.ref()
ref = ref.to(dev).eval()
with torch.no_grad():
    out = ref(**{k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()})
t = out[0] if isinstance(out, tuple) else getattr(out, "last_hidden_state", None)
t = t if t is not None else out.sample
print(f"{type(ref).__name__}  params={sum(p.numel() for p in ref.parameters()):,}  out={tuple(t.shape)}")
""")

    nb.md("""
## 내 구현 == 원본

원본 가중치에 잡음을 넣고(초기값 0/1 때문에 복사 누락이 가려지지 않도록) 내 구현에 복사한 뒤, 층별 출력과 최종 출력을 비교합니다.
""")
    nb.code("""
row = V.cases.run_case(S.CASE)
print(row["status"]); print(row["detail"])
report("원본 동치 (" + S.CASE.reference + ")", row["status"].startswith("✅"))
""")

    if spec["CAUSAL"]:
        nb.md("""
## causal 누수

뒤쪽 토큰을 바꿨을 때 앞쪽 출력이 변하면 미래 정보가 새고 있다는 뜻입니다.
""")
        nb.code("""
import importlib
V.set_deterministic(0)
mine = importlib.import_module(S.CASE.mine_module).verify_pair(S.ref()[0])["mine"].to(dev).eval()
r = V.causal_leak_test(lambda x: mine(x).last_hidden_state, vocab_size=V.cases.VOCAB, seq_len=48, device=dev, atol=1e-4)
report("내 구현 " + r.name, r.ok, r.max_abs)
""")
    if spec["SELF_CHECKS"]:
        nb.md("## 단계 전용 자체 점검")
        for m, c in spec["SELF_CHECKS"]:
            nb.md(m)
            nb.code(c)


def train_cells(nb):
    """학습 동작 검증: 기울기·학습 궤적 == 원본, 배치 외우기, float64 수치 미분, 라우터 bias."""

    nb.md("""
## 학습 동작 확인

forward 가 원본과 같아도 학습이 되는지는 따로 확인해야 합니다.
- **기울기 == 원본**: 같은 가중치·같은 손실로 원본과 내 구현을 backward 해서 파라미터 기울기를 비교합니다 (파라미터별 상대 오차 ≤ 1e-3).
  원본이 추론 전용 코드(no_grad)면 건너뜁니다.
- **학습 궤적 == 원본**: 같은 가중치·데이터·optimizer(SGD momentum 0.9, 30 step)로 원본과 내 구현을 나란히 학습해
  step 별 손실과 최종 가중치를 비교합니다 (상대차 ≤ 1e-3). Adam 은 아주 작은 잡음 차이를 키워서 궤적 비교에 쓰지 않습니다.
- **배치 하나 외우기**: 고정 배치 하나로 300 step 학습해 손실이 처음의 20% 이하로 내려가는지 봅니다 (LM은 다음 토큰 CE, 그 외는 고정 목표 MSE).
  NaN/inf 가 없어야 하고, 기울기를 한 번도 받지 못한 파라미터를 보여줍니다 (MoE 에서 안 뽑힌 expert 는 정상).
""")
    nb.code("""
import importlib, traceback
from miniarch.verify import training as TR
mod = importlib.import_module(S.CASE.mine_module)
V.set_deterministic(0)
ref, inputs = S.ref()
V.perturb_params(ref)
pair = mod.verify_pair(ref)
try:
    g = TR.grad_equivalence(ref, pair, inputs, dev)
    if g.note.startswith("원본 출력에 기울기가 없음"):
        print("⏭️", g.note); report("기울기 == 원본 (원본 추론 전용 → 건너뜀)", True)
    else:
        print(f"비교한 파라미터 {g.compared}개, 최대 상대 오차 {g.max_rel:.2e} ({g.worst})", g.note)
        if g.skipped_ref:
            print("원본에서 기울기 없는 파라미터:", g.skipped_ref[:8])
        report("기울기 == 원본", g.ok, g.max_rel)
except Exception:
    traceback.print_exc(limit=4); report("기울기 == 원본", False)
""")
    nb.code("""
try:
    t = TR.train_parity(ref, pair, inputs, dev)
    if t.note.startswith("원본 출력에 기울기가 없음"):
        print("⏭️", t.note); report("학습 궤적 == 원본 (원본 추론 전용 → 건너뜀)", True)
    else:
        print(f"{t.steps} step  손실 최대 상대차 {t.max_loss_rel:.2e}  최종 가중치 최대 상대차 {t.max_param_rel:.2e} ({t.worst_param})", t.note)
        print("  원본:", " ".join(f"{v:.4g}" for v in t.loss_ref[::6]))
        print("  내것:", " ".join(f"{v:.4g}" for v in t.loss_mine[::6]))
        report("학습 궤적 == 원본 (SGD 30 step)", t.ok, max(t.max_loss_rel, t.max_param_rel))
except Exception:
    traceback.print_exc(limit=4); report("학습 궤적 == 원본", False)
""")
    nb.code("""
try:
    o = TR.overfit_batch(pair["mine"], pair, inputs, dev)
    print(f"{o.kind}: {o.loss_start:.4f} → {o.loss_end:.4f}  (유한={o.finite})")
    print("  손실:", " ".join(f"{v:.3g}" for v in o.curve[::30]))
    if o.no_grad_params:
        print(f"  기울기를 못 받은 파라미터 {len(o.no_grad_params)}개:", o.no_grad_params[:8])
    report("배치 하나 외우기 (" + o.kind + ")", o.ok, o.loss_end / max(o.loss_start, 1e-12))
except Exception:
    traceback.print_exc(limit=4); report("배치 하나 외우기", False)
""")
    nb.md("""
### 원본과 비교할 수 없는 부분

- **수치 미분 (float64)**: 무작위 방향으로 가중치를 ±ε 움직여 잰 손실 변화가 autograd 기울기와 같은지 봅니다.
  원본이 추론 전용이라 기울기를 비교할 수 없는 모델도 이걸로 backward 를 확인합니다.
- **라우터 bias 업데이트 (aux-loss-free)**: 원본 추론 코드에는 없는 학습 전용 규칙입니다. 많이 뽑힌 expert 의 bias 는 내려가고
  적게 뽑힌 쪽은 올라가는지, bias 가 optimizer 에서 빠져 있는지, bias 가 실제로 선택을 바꾸는지 봅니다. 라우터 bias 가 없는 모델은 건너뜁니다.
""")
    nb.code("""
try:
    f = TR.directional_grad_check(pair["mine"], pair, inputs, dev)
    print(f"방향 {len(f.rel_errors)}개 상대 오차 {[f'{e:.1e}' for e in f.rel_errors]}  (ε={f.eps})")
    report("수치 미분 == autograd (float64)", f.ok, max(f.rel_errors))
except Exception:
    traceback.print_exc(limit=4); report("수치 미분 == autograd (float64)", False)
try:
    b = TR.router_bias_check(pair["mine"], pair, inputs, dev)
    print("\\n".join(b.details))
    report("라우터 bias 업데이트" if b.n_routers else "라우터 bias 업데이트 (라우터 bias 없음 → 건너뜀)", b.ok)
except Exception:
    traceback.print_exc(limit=4); report("라우터 bias 업데이트", False)
""")





KINDS = {   # kind: (노트북 폴더, 결과 csv 접두어, 제목 꼬리)
    "verify": ("verify", "verify", "순전파 검증"),
    "train": ("train_verify", "train", "학습 동작 검증"),
}


def build(module_name, kind="verify"):
    spec = read_spec(os.path.join(STAGES_DIR, module_name + ".py"))
    folder, prefix, tail = KINDS[kind]
    stage, path = spec["STAGE"], f"notebooks/{folder}/{spec['STAGE']}_{spec['SLUG']}.ipynb"
    nb = NB()
    if kind == "verify":
        steps = ["환경 준비"] + (["원본 구조 (HF 헤더)"] if spec["HEADER"] else []) + \
                ["원본 모델 동작 확인", "내 구현 == 원본 (층별 출력 비교)"] + \
                (["causal 누수"] if spec["CAUSAL"] else []) + \
                (["단계 전용 자체 점검"] if spec["SELF_CHECKS"] else []) + ["결과 저장"]
    else:
        steps = ["환경 준비", "기울기 == 원본", "학습 궤적 == 원본", "배치 하나 외우기", "float64 수치 미분",
                 "라우터 bias 업데이트", "결과 저장"]
    nb.md(f"""
# {stage} · {spec['TITLE']} {tail}

{nb.badge(path)}

{spec['INTRO'].strip()}

순서: {' → '.join(f'{i}. {s}' for i, s in enumerate(steps))}

**런타임 → 런타임 유형 변경 → T4 GPU** 선택 후 위에서부터 실행하세요. 다른 모델 노트북과 독립적으로 돌아갑니다.
""")
    nb.setup()
    if spec["PIP"]:
        nb.code("if not os.environ.get('MINIARCH_SKIP_PIP'):   # 전체 실행 노트북이 한 번에 설치\n"
                "    !pip -q install -U " + " ".join("'" + p.strip("'\"") + "'" for p in spec["PIP"]))  # 따옴표: >= 가 셸 리다이렉트가 되지 않게
    nb.code(f"""
from miniarch import verify as V
from miniarch.verify import stages
S = stages.get("{stage}")
dev = "cuda" if torch.cuda.is_available() else "cpu"
try:   # gated 모델 헤더/원격 코드용. Colab 왼쪽 🔑(보안 비밀)에 HF_TOKEN 등록.
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass

CHECKS = []   # (이름, 통과, 오차) → 마지막에 저장
def report(name, ok, err=float("nan")):
    CHECKS.append({{"check": name, "ok": bool(ok), "max_abs": float(err)}})
    print(f"{{'✅' if ok else '❌'}} {{name:<44}} max_abs={{err:.2e}}")
""")
    if kind == "train":
        train_cells(nb)
    else:
        verify_cells(nb, spec)

    nb.md("""
## 결과 저장

- `results/`에 점검 결과를 저장합니다.
- 노트북 출력까지 남기려면 **파일 → GitHub에 사본 저장** → 이 저장소의 `""" + path + """`.
""")
    nb.code(f"""
import datetime, pandas as pd
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(CHECKS)
p = f"results/{prefix}_{stage}_{{datetime.datetime.now():%Y%m%d-%H%M}}.csv"
df.to_csv(p, index=False)
print(df.to_string(index=False)); print("saved:", p)
print("전체:", "✅ 모두 통과" if df.ok.all() else "❌ 실패 있음")
""")
    nb.save(path)
    return path, spec


def build_run_all(specs, kind="verify"):
    """모든 모델 노트북을 각각 별도 프로세스로 실행하고 결과를 한 표로 모으는 노트북."""
    folder, prefix, tail = KINDS[kind]
    path = f"notebooks/{folder}/00_run_all.ipynb"
    stages = [s["STAGE"] for s in specs]
    pips = sorted({p.strip("'\"") for s in specs for p in s["PIP"]})
    nb = NB()
    nb.md(f"""
# {tail} 전체 실행

{nb.badge(path)}

`notebooks/{folder}/`의 모델별 노트북을 **한 번에** 돌립니다.

- 노트북마다 **새 프로세스**(`jupyter nbconvert --execute`)로 실행합니다. 원본 코드 패치나 원격 코드 import 같은 전역 상태가 다음 모델로 새지 않고, 끝나면 GPU 메모리도 비워집니다.
- 패키지는 여기서 한 번만 설치하고, 각 노트북의 설치 셀은 건너뜁니다.
- 한 노트북에서 에러가 나도 멈추지 않고 다음으로 넘어갑니다. 실행된 노트북(출력 포함)은 `results/executed/`에 남습니다.
- 끝나면 요약표를 보여주고, 결과 전체를 zip으로 내려받습니다.

T4 기준 모델당 2~5분 정도입니다. 일부만 돌리려면 아래 `SELECT`에 단계 이름을 적으세요.
""")
    nb.setup()
    nb.code(f"""
# 모든 모델 노트북이 요구하는 추가 패키지 (한 번만 설치)
!pip -q install -U {" ".join("'" + p + "'" for p in pips)}
!pip -q install -U nbconvert ipykernel
""")
    nb.code(f"""
import glob, subprocess, time, pandas as pd
PREFIX = "{prefix}"   # 결과 csv 접두어
SELECT = None   # 예: ["B1", "B2", "C2"]  (None = 전부: {", ".join(stages)})

books = sorted(p for p in glob.glob("notebooks/{folder}/*.ipynb") if not os.path.basename(p).startswith("00_"))
if SELECT:
    books = [p for p in books if os.path.basename(p).split("_")[0] in SELECT]
os.makedirs("results/executed", exist_ok=True)
try:   # 자식 프로세스에서는 Colab 보안 비밀을 못 읽으므로 여기서 환경변수로 넘김 (gated 모델 헤더용)
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass
env = dict(os.environ, MINIARCH_SKIP_PIP="1")
runs = []
for p in books:
    name = os.path.basename(p)
    t0 = time.time()
    r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--allow-errors",
                        "--ExecutePreprocessor.timeout=1800", "--output-dir", "results/executed", p],
                       env=env, capture_output=True, text=True)
    runs.append({{"notebook": name, "exit": r.returncode, "min": round((time.time() - t0) / 60, 1)}})
    print(f"{{'✅' if r.returncode == 0 else '💥'}} {{name:<34}} {{runs[-1]['min']}}분")
    if r.returncode:
        print(r.stderr[-1500:])
""")
    nb.md("""
## 요약

단계마다 가장 최근 결과 csv를 읽습니다. `통과/전체`가 다르면 아래에 실패한 점검 이름이 나옵니다.
csv가 없으면 노트북이 저장 셀까지 못 간 것이니 `results/executed/`의 해당 노트북 출력을 보세요.
""")
    nb.code("""
rows, fails = [], []
for p in books:
    stage = os.path.basename(p).split("_")[0]
    csvs = sorted(glob.glob(f"results/{PREFIX}_{stage}_*.csv"))
    if not csvs:
        rows.append({"stage": stage, "통과/전체": "-", "결과": "💥 csv 없음"}); continue
    df = pd.read_csv(csvs[-1])
    rows.append({"stage": stage, "통과/전체": f"{int(df.ok.sum())}/{len(df)}", "결과": "✅" if df.ok.all() else "❌"})
    fails += [f"{stage}: {c}" for c in df.loc[~df.ok, "check"]]
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print("\\n실패한 점검:" if fails else "\\n모든 점검 통과", *fails, sep="\\n  ")
""")
    nb.code("""
import datetime, shutil
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
summary.to_csv(f"results/{PREFIX}_summary_{stamp}.csv", index=False)
zip_path = shutil.make_archive(f"/content/{PREFIX}_results_{stamp}", "zip", "results")
print("zip:", zip_path)
try:
    from google.colab import files
    files.download(zip_path)
except Exception as e:
    print("자동 다운로드 실패, 왼쪽 파일 탭에서 받으세요:", e)
""")
    nb.save(path)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(f[:-3] for f in os.listdir(STAGES_DIR) if f.endswith(".py") and not f.startswith("_"))
    for kind, (folder, _, _) in KINDS.items():
        os.makedirs(os.path.join(ROOT, "notebooks", folder), exist_ok=True)
        specs = [build(n, kind)[1] for n in names]
        if not sys.argv[1:]:
            build_run_all(specs, kind)
