"""노트북 생성 공용 헬퍼. 노트북은 직접 편집하지 말고 tools/build_*.py 를 고쳐서 다시 생성한다."""
import ast
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO = "HisameOgasahara/tmp_llm_dit"


class NB:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})

    def code(self, s):
        self.cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": s.strip("\n")})

    def badge(self, path):
        return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                f"(https://colab.research.google.com/github/{REPO}/blob/main/{path})")

    def setup(self):
        """저장소 clone/pull + 패키지 설치 + 환경 출력."""
        self.md("## 0. 환경 준비")
        self.code(f"""
import os
import re
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
        self.code("""
import sys, torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음 (런타임 유형을 GPU로 바꾸세요)")
!git log -1 --format='repo commit: %h %s'
""")

    def save(self, path):
        cells = [dict(c, source=c["source"].splitlines(keepends=True)) for c in self.cells]
        nb = {"cells": cells,
              "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                           "kernelspec": {"display_name": "Python 3", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 0}
        full = os.path.join(ROOT, path)  # path 는 저장소 기준 (배지 URL 에도 그대로 쓰임)
        with open(full, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        json.load(open(full, encoding="utf-8"))  # 깨진 JSON 방지
        for i, c in enumerate(self.cells):       # 셀 문법 오류 방지 (! / % 줄은 pass 로 바꿔 검사)
            if c["cell_type"] == "code":
                src = "\n".join(re.sub(r"^(\s*)[!%].*", r"\1pass", l) for l in c["source"].splitlines())
                ast.parse(src, filename=f"{path} cell {i}")
        print("ok", path, len(cells), "cells")


SAVE_RESULTS = '''
import datetime, json, shutil
STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")
os.makedirs("results", exist_ok=True)

def save_fig(fig, name):
    path = f"results/{PREFIX}_{name}_{STAMP}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path
'''

DRIVE = '''
SAVE_TO_DRIVE = False   # True 로 바꾸면 results/ 를 Drive 에도 복사 (체크포인트 포함)
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    dst = "/content/drive/MyDrive/tmp_llm_dit/results"
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir("results"):
        if STAMP in f:
            shutil.copy(f"results/{f}", dst)
    print("drive:", dst)
'''
