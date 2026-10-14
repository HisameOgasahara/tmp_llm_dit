"""원본 없이도 할 수 있는 자체 일관성 검사."""
from __future__ import annotations

from typing import Callable

import torch

from .equivalence import CompareResult, compare_tensors

Fwd = Callable[[torch.Tensor], torch.Tensor]  # input_ids [B,T] -> hidden/logits [B,T,...]


@torch.no_grad()
def causal_leak_test(fwd: Fwd, vocab_size: int, seq_len: int = 32, batch: int = 2,
                     device: str = "cpu", atol: float = 1e-5) -> CompareResult:
    """위치 t 이후 토큰을 바꿔도 t 이전 출력이 그대로여야 한다 (미래 정보 누수 검사)."""
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, vocab_size, (batch, seq_len), generator=g).to(device)
    t = seq_len // 2
    x2 = x.clone()
    x2[:, t:] = torch.randint(0, vocab_size, (batch, seq_len - t), generator=g).to(device)
    a, b = fwd(x)[:, :t], fwd(x2)[:, :t]
    return compare_tensors(f"causal leak (prefix<{t})", a, b, atol=atol, rtol=0)


@torch.no_grad()
def incremental_decode_test(full_fwd: Fwd,
                            step_fwd: Callable[[torch.Tensor, object], tuple[torch.Tensor, object]],
                            vocab_size: int, seq_len: int = 24, device: str = "cpu",
                            atol: float = 1e-4) -> CompareResult:
    """전체 시퀀스 forward == 토큰 하나씩 cache/state 를 넘기며 forward.

    step_fwd(x_t [B,1], cache) -> (out [B,1,...], new_cache). 첫 호출 cache=None.
    KV cache(softmax 계열)와 recurrent state(선형 계열) 모두 이 형태로 맞춘다.
    """
    x = torch.randint(0, vocab_size, (1, seq_len), generator=torch.Generator().manual_seed(1)).to(device)
    full = full_fwd(x)
    cache, outs = None, []
    for i in range(seq_len):
        o, cache = step_fwd(x[:, i:i + 1], cache)
        outs.append(o)
    return compare_tensors("full vs incremental", full, torch.cat(outs, 1), atol=atol, rtol=1e-3)


@torch.no_grad()
def same_function_test(name: str, f1: Callable, f2: Callable, *args,
                       atol: float = 1e-4, rtol: float = 1e-3) -> CompareResult:
    """같은 수식의 두 구현 비교 (예: 선형 어텐션 recurrent vs chunk, naive vs 최적화)."""
    return compare_tensors(name, f1(*args), f2(*args), atol=atol, rtol=rtol)
