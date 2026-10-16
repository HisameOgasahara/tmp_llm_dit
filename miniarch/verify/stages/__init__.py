"""단계별 검증 케이스 (파일 하나 = 단계 하나 = 검증 노트북 하나).

각 모듈이 정의하는 것
  STAGE, SLUG, TITLE : 단계 이름, 노트북 파일 이름(notebooks/verify/{STAGE}_{SLUG}.ipynb), 제목  — 문자열 리터럴
  INTRO              : 노트북 첫 설명 (markdown)                                              — 문자열 리터럴
  PIP                : 추가로 설치할 pip 패키지 목록                                            — 리터럴
  HEADER             : 원본 구조 미리보기 [(repo, 정규식, subfolder), ...]                     — 리터럴
  CAUSAL             : True 면 내 구현 causal 누수 검사 셀을 자동으로 넣는다                    — 리터럴
  SELF_CHECKS        : [(markdown, code), ...] 단계 전용 자체 점검 셀                           — 리터럴
  ref()              : (원본 모듈, 입력 dict)
  CASE               : cases.Case

리터럴 항목은 tools/build_verify.py 가 torch 없이 ast 로 읽어 노트북을 만든다.
여러 사람이(에이전트가) 동시에 다른 단계를 추가해도 공유 파일을 건드리지 않도록 자동 수집한다.
"""
from __future__ import annotations

import importlib
import pkgutil


def modules():
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if not info.name.startswith("_"):
            yield importlib.import_module(f"{__name__}.{info.name}")


def get(stage: str):
    return next(m for m in modules() if m.STAGE == stage)
