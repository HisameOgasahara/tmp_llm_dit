"""Modular addition: [a, b, =] → (a + b) mod p.

grokking 의 대표 태스크. 학습 데이터를 외운 뒤 한참 지나서 갑자기 일반화되고,
그 시점에 임베딩이 몇 개의 푸리에 주파수로 정리된다 (Nanda et al. 2023).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..analysis.fourier import topk_energy_frac
from ..models.gpt2 import GPT2, GPT2Config
from ..models.llama import Llama, LlamaConfig


@dataclass
class ModAddData:
    p: int
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


def make_data(p: int = 113, train_frac: float = 0.3, seed: int = 0, device: str = "cpu") -> ModAddData:
    a, b = torch.cartesian_prod(torch.arange(p), torch.arange(p)).T
    x = torch.stack([a, b, torch.full_like(a, p)], dim=1)  # 토큰 p = "="
    y = (a + b) % p
    perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(seed))
    n = int(train_frac * p * p)
    tr, te = perm[:n], perm[n:]
    return ModAddData(p, x[tr].to(device), y[tr].to(device), x[te].to(device), y[te].to(device))


def build_model(kind: str, p: int, d_model: int = 128, n_head: int = 4, n_layer: int = 1,
                use_ln: bool = True, tie: bool = True) -> nn.Module:
    """use_ln/tie 는 GPT-2 ablation 용. use_ln=False, tie=False 가 grokking 원 논문(Nanda et al.) 설정에 가깝다."""
    vocab = p + 1
    if kind == "gpt2":
        return GPT2(GPT2Config(vocab_size=vocab, n_positions=3, d_model=d_model, n_layer=n_layer, n_head=n_head,
                               use_ln=use_ln, tie_embeddings=tie))
    if kind == "llama":
        return Llama(LlamaConfig(vocab_size=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head,
                                 n_kv_head=n_head, d_ff=4 * d_model))
    raise ValueError(kind)


@torch.no_grad()
def attention_grid(model: nn.Module, p: int, layer: int = 0) -> torch.Tensor:
    """모든 (a, b) 입력에 대해 '=' 위치가 a 를 보는 attention 확률 → [H, p, p] (행 a, 열 b)."""
    from ..analysis.attention import capture_attention
    dev = next(model.parameters()).device
    a, b = torch.cartesian_prod(torch.arange(p), torch.arange(p)).T
    x = torch.stack([a, b, torch.full_like(a, p)], dim=1).to(dev)
    model.eval()
    att = capture_attention(model, lambda: model(x))
    P = list(att.values())[layer]                 # [p*p, H, 3, 3]
    return P[:, :, 2, 0].T.reshape(-1, p, p)      # '=' → a


def embedding_weight(model: nn.Module) -> torch.Tensor:
    """숫자 0..p-1 의 임베딩 행렬 (마지막 '=' 토큰 제외)"""
    emb = model.tok_emb if hasattr(model, "tok_emb") else model.embed_tokens
    return emb.weight[:-1]


def _logits(model, x, p):
    return model(x).logits[:, -1, :p].float()  # 마지막 위치('=')에서 정답 숫자 예측


@torch.no_grad()
def evaluate(model, x, y, p) -> tuple[float, float]:
    model.eval()
    lg = _logits(model, x, p)
    return F.cross_entropy(lg, y).item(), (lg.argmax(-1) == y).float().mean().item()


def train(model: nn.Module, data: ModAddData, steps: int = 30000, lr: float = 1e-3, wd: float = 1.0,
          log_every: int = 100, progress: bool = True) -> dict:
    """전체 학습 데이터를 한 배치로 쓰는 full-batch 학습."""
    from tqdm.auto import tqdm

    p = data.p
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10))
    hist = {k: [] for k in ["step", "train_loss", "train_acc", "test_loss", "test_acc",
                            "weight_norm", "fourier_top5"]}
    bar = tqdm(range(steps + 1), disable=not progress, mininterval=1.0)
    for step in bar:
        if step % log_every == 0:
            trl, tra = evaluate(model, data.train_x, data.train_y, p)
            tel, tea = evaluate(model, data.test_x, data.test_y, p)
            with torch.no_grad():
                wn = torch.sqrt(sum(q.pow(2).sum() for q in model.parameters())).item()
                f5 = topk_energy_frac(embedding_weight(model).float(), p, k=5)
            for k, v in zip(hist, [step, trl, tra, tel, tea, wn, f5]):
                hist[k].append(v)
            bar.set_postfix(train=f"{tra:.2f}", test=f"{tea:.2f}", wn=f"{wn:.1f}")
        if step == steps:
            break
        model.train()
        loss = F.cross_entropy(_logits(model, data.train_x, p), data.train_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    hist["grok_step"] = next((s for s, a in zip(hist["step"], hist["test_acc"]) if a > 0.99), None)
    hist["memorize_step"] = next((s for s, a in zip(hist["step"], hist["train_acc"]) if a > 0.99), None)
    return hist
