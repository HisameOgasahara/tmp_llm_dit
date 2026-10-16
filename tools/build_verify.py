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


def build(module_name):
    spec = read_spec(os.path.join(STAGES_DIR, module_name + ".py"))
    stage, path = spec["STAGE"], f"notebooks/verify/{spec['STAGE']}_{spec['SLUG']}.ipynb"
    nb = NB()
    steps = ["환경 준비"] + (["원본 구조 (HF 헤더)"] if spec["HEADER"] else []) + \
            ["원본 모델 동작 확인", "내 구현 == 원본 (층별 출력 비교)"] + \
            (["causal 누수"] if spec["CAUSAL"] else []) + (["단계 전용 자체 점검"] if spec["SELF_CHECKS"] else []) + \
            ["결과 저장"]
    nb.md(f"""
# {stage} · {spec['TITLE']} 검증

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

    nb.md("""
## 결과 저장

- `results/`에 점검 결과를 저장합니다.
- 노트북 출력까지 남기려면 **파일 → GitHub에 사본 저장** → 이 저장소의 `""" + path + """`.
""")
    nb.code(f"""
import datetime, pandas as pd
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(CHECKS)
p = f"results/verify_{stage}_{{datetime.datetime.now():%Y%m%d-%H%M}}.csv"
df.to_csv(p, index=False)
print(df.to_string(index=False)); print("saved:", p)
print("전체:", "✅ 모두 통과" if df.ok.all() else "❌ 실패 있음")
""")
    nb.save(path)
    return path, spec


def build_run_all(specs):
    """모든 모델 노트북을 각각 별도 프로세스로 실행하고 결과를 한 표로 모으는 노트북."""
    path = "notebooks/verify/00_run_all.ipynb"
    stages = [s["STAGE"] for s in specs]
    pips = sorted({p.strip("'\"") for s in specs for p in s["PIP"]})
    nb = NB()
    nb.md(f"""
# 검증 전체 실행

{nb.badge(path)}

`notebooks/verify/`의 모델별 검증 노트북을 **한 번에** 돌립니다.

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
SELECT = None   # 예: ["B1", "B2", "C2"]  (None = 전부: {", ".join(stages)})

books = sorted(p for p in glob.glob("notebooks/verify/*.ipynb") if not os.path.basename(p).startswith("00_"))
if SELECT:
    books = [p for p in books if os.path.basename(p).split("_")[0] in SELECT]
os.makedirs("results/executed", exist_ok=True)
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
    csvs = sorted(glob.glob(f"results/verify_{stage}_*.csv"))
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
summary.to_csv(f"results/verify_summary_{stamp}.csv", index=False)
zip_path = shutil.make_archive(f"/content/verify_results_{stamp}", "zip", "results")
print("zip:", zip_path)
try:
    from google.colab import files
    files.download(zip_path)
except Exception as e:
    print("자동 다운로드 실패, 왼쪽 파일 탭에서 받으세요:", e)
""")
    nb.save(path)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "notebooks", "verify"), exist_ok=True)
    names = sys.argv[1:] or sorted(f[:-3] for f in os.listdir(STAGES_DIR) if f.endswith(".py") and not f.startswith("_"))
    specs = [build(n)[1] for n in names]
    if not sys.argv[1:]:
        build_run_all(specs)
