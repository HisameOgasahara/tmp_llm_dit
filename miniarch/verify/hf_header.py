"""HF safetensors 헤더만 HTTP range 요청으로 읽어 원본 가중치 이름/shape/dtype 을 얻는다.

가중치 본체는 받지 않으므로 수백 GB 모델도 수 MB 이내로 구조를 확인할 수 있다.
gated 모델은 환경변수 HF_TOKEN 이 필요하다.
"""
from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass

import requests

HF = "https://huggingface.co"


def _headers(extra: dict | None = None) -> dict:
    h = dict(extra or {})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _url(repo: str, path: str, revision: str = "main") -> str:
    return f"{HF}/{repo}/resolve/{revision}/{path}"


def fetch_json(repo: str, path: str, revision: str = "main") -> dict:
    r = requests.get(_url(repo, path, revision), headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_config(repo: str, subfolder: str = "", revision: str = "main") -> dict:
    return fetch_json(repo, f"{subfolder}/config.json" if subfolder else "config.json", revision)


def read_safetensors_header(repo: str, filename: str, revision: str = "main") -> dict:
    """파일 하나의 헤더: {tensor_name: {"dtype":..., "shape":[...], "data_offsets":[a,b]}}"""
    url = _url(repo, filename, revision)
    r = requests.get(url, headers=_headers({"Range": "bytes=0-7"}), timeout=60)
    r.raise_for_status()
    (n,) = struct.unpack("<Q", r.content[:8])
    r = requests.get(url, headers=_headers({"Range": f"bytes=8-{7 + n}"}), timeout=120)
    r.raise_for_status()
    h = json.loads(r.content)
    h.pop("__metadata__", None)
    return h


_INDEX_CANDIDATES = [
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
]
_SINGLE_CANDIDATES = ["model.safetensors", "diffusion_pytorch_model.safetensors"]


@dataclass
class TensorInfo:
    dtype: str
    shape: tuple
    file: str


def read_model_header(repo: str, subfolder: str = "", pattern: str | None = None,
                      revision: str = "main", max_files: int | None = None) -> dict[str, TensorInfo]:
    """repo 전체(혹은 pattern 정규식에 맞는 텐서만)의 shape 정보.

    pattern 을 주면 해당 텐서가 들어있는 shard 의 헤더만 읽는다.
    예: pattern=r"layers\\.0\\.self_attn"
    """
    pre = f"{subfolder}/" if subfolder else ""
    rx = re.compile(pattern) if pattern else None
    files: list[str] | None = None
    for idx in _INDEX_CANDIDATES:
        try:
            wm = fetch_json(repo, pre + idx, revision)["weight_map"]
        except requests.HTTPError:
            continue
        keys = [k for k in wm if rx is None or rx.search(k)]
        files = sorted({wm[k] for k in keys})
        break
    if files is None:  # 단일 파일
        for single in _SINGLE_CANDIDATES:
            r = requests.head(_url(repo, pre + single, revision), headers=_headers(),
                              allow_redirects=True, timeout=60)
            if r.ok:
                files = [single]
                break
    if not files:
        raise FileNotFoundError(f"{repo}/{pre} 에서 safetensors 를 찾지 못함")
    if max_files:
        files = files[:max_files]

    out: dict[str, TensorInfo] = {}
    for f in files:
        for k, v in read_safetensors_header(repo, pre + f, revision).items():
            if rx is None or rx.search(k):
                out[k] = TensorInfo(v["dtype"], tuple(v["shape"]), f)
    return out


def show(header: dict[str, TensorInfo], limit: int = 200) -> None:
    for i, (k, v) in enumerate(sorted(header.items())):
        if i >= limit:
            print(f"... ({len(header) - limit} more)")
            break
        print(f"{k:<70} {v.dtype:<8} {list(v.shape)}")
