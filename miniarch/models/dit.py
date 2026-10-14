"""B0 · DiT: patchify + 2D sincos 위치 + adaLN-Zero 블록 + 클래스/시간 조건.

원 논문(Peebles & Xie) 구조대로 timestep/class 임베더는 모델 전체가 하나를 공유한다.
(diffusers 구현은 블록마다 임베더를 복제해 두고 0번 것만 최종층에 쓴다 → 검증 시 0번으로 통일)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .common import attention


@dataclass
class DiTConfig:
    in_ch: int = 1
    img_size: int = 28
    patch: int = 4
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 4
    n_classes: int = 10
    class_dropout: float = 0.1   # >0 이면 "조건 없음" 클래스가 하나 추가됨 (CFG 용)
    t_freq_dim: int = 256
    out_ch: int | None = None

    def __post_init__(self):
        self.out_ch = self.out_ch or self.in_ch


# ---------------------------------------------------------------- 임베딩

def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0) -> torch.Tensor:
    """[cos | sin] 정현파 시간 임베딩 (diffusers: flip_sin_to_cos=True, shift=1)."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    args = t[:, None].float() * exponent.exp()[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def sincos_2d(d: int, grid: int) -> torch.Tensor:
    """고정 2D 위치 임베딩 [grid*grid, d]. 앞 절반 = 가로(x), 뒤 절반 = 세로(y)."""
    def one(dim, pos):
        omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2.0))
        out = pos.reshape(-1)[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)
    p = torch.arange(grid, dtype=torch.float64)
    gy, gx = torch.meshgrid(p, p, indexing="ij")
    return torch.cat([one(d // 2, gx), one(d // 2, gy)], dim=1).float()


class PatchEmbed(nn.Module):
    def __init__(self, c: DiTConfig):
        super().__init__()
        self.proj = nn.Conv2d(c.in_ch, c.d_model, kernel_size=c.patch, stride=c.patch)
        self.register_buffer("pos", sincos_2d(c.d_model, c.img_size // c.patch)[None], persistent=False)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2) + self.pos


# ---------------------------------------------------------------- 블록

class Attention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.h = n_head
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.attn_probs = nn.Identity()  # hook 지점: [B, H, N, N] (양방향)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = (f(x).view(B, N, self.h, -1).transpose(1, 2) for f in (self.q, self.k, self.v))
        y = attention(q, k, v, causal=False, probs_hook=self.attn_probs)
        return self.o(y.transpose(1, 2).reshape(B, N, C))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, c: DiTConfig):
        super().__init__()
        d = c.d_model
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d, c.n_head)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-5)
        self.mlp = MLP(d)
        self.adaLN = nn.Linear(d, 6 * d)  # 조건 → shift/scale/gate × (attn, mlp)

    def forward(self, x, cond):
        sa, sca, ga, sm, scm, gm = self.adaLN(F.silu(cond)).chunk(6, dim=1)
        x = x + ga[:, None] * self.attn(modulate(self.norm1(x), sa, sca))
        return x + gm[:, None] * self.mlp(modulate(self.norm2(x), sm, scm))


class DiT(nn.Module):
    def __init__(self, c: DiTConfig):
        super().__init__()
        self.config = c
        d = c.d_model
        self.patch_embed = PatchEmbed(c)
        self.t_embedder = nn.Sequential(nn.Linear(c.t_freq_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.y_embedder = nn.Embedding(c.n_classes + int(c.class_dropout > 0), d)
        self.blocks = nn.ModuleList(DiTBlock(c) for _ in range(c.n_layer))
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Linear(d, 2 * d)
        self.final_out = nn.Linear(d, c.patch * c.patch * c.out_ch)
        self.initialize_weights()

    def initialize_weights(self):
        """DiT 논문 초기화: adaLN 과 출력층을 0 으로 → 학습 시작 시 각 블록이 항등함수."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        for lin in (self.t_embedder[0], self.t_embedder[2]):
            nn.init.normal_(lin.weight, std=0.02)
        for b in self.blocks:
            nn.init.zeros_(b.adaLN.weight); nn.init.zeros_(b.adaLN.bias)
        for lin in (self.final_adaLN, self.final_out):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def condition(self, t, y):
        if self.training and self.config.class_dropout > 0:  # CFG 용 라벨 드롭
            drop = torch.rand(y.shape[0], device=y.device) < self.config.class_dropout
            y = torch.where(drop, torch.full_like(y, self.config.n_classes), y)
        return self.t_embedder(timestep_embedding(t, self.config.t_freq_dim)) + self.y_embedder(y)

    def forward(self, x, t, y):
        c = self.config
        g = c.img_size // c.patch
        cond = self.condition(t, y)
        h = self.patch_embed(x)
        for b in self.blocks:
            h = b(h, cond)
        shift, scale = self.final_adaLN(F.silu(cond)).chunk(2, dim=1)
        h = self.final_out(modulate(self.final_norm(h), shift, scale))
        # unpatchify: [B, g*g, p*p*C] → [B, C, H, W]
        h = h.reshape(-1, g, g, c.patch, c.patch, c.out_ch)
        return torch.einsum("nhwpqc->nchpwq", h).reshape(-1, c.out_ch, g * c.patch, g * c.patch)


# ---------------------------------------------------------------- 검증

def diffusers_name_map(name: str) -> str | None:
    m = re.match(r"transformer_blocks\.(\d+)\.(.*)", name)
    if not m:
        return (name.replace("pos_embed.proj.", "patch_embed.proj.")
                    .replace("proj_out_1.", "final_adaLN.").replace("proj_out_2.", "final_out."))
    i, rest = int(m.group(1)), m.group(2)
    if rest.startswith("norm1.emb."):
        if i != 0:
            return None  # 공유 임베더는 0번 블록 것만 사용
        rest = rest[len("norm1.emb."):]
        rest = (rest.replace("timestep_embedder.linear_1.", "t_embedder.0.")
                    .replace("timestep_embedder.linear_2.", "t_embedder.2.")
                    .replace("class_embedder.embedding_table.", "y_embedder."))
        return rest
    for a, b in [("norm1.linear.", "adaLN."), ("attn1.to_q.", "attn.q."), ("attn1.to_k.", "attn.k."),
                 ("attn1.to_v.", "attn.v."), ("attn1.to_out.0.", "attn.o."),
                 ("ff.net.0.proj.", "mlp.fc1."), ("ff.net.2.", "mlp.fc2.")]:
        if rest.startswith(a):
            return f"blocks.{i}.{b}{rest[len(a):]}"
    return f"blocks.{i}.{rest}"  # 매핑 안 된 이름 → 검증 리포트에 드러남


def verify_pair(ref) -> dict:
    """ref = diffusers.DiTTransformer2DModel"""
    c = ref.config
    blocks = ref.transformer_blocks
    with torch.no_grad():  # 블록별로 복제된 임베더를 0번 것으로 통일 (원 논문 구조)
        for b in blocks[1:]:
            b.norm1.emb.load_state_dict(blocks[0].norm1.emb.state_dict())
    n_rows = blocks[0].norm1.emb.class_embedder.embedding_table.weight.shape[0]
    mine = DiT(DiTConfig(
        in_ch=c.in_channels, img_size=c.sample_size, patch=c.patch_size,
        d_model=c.num_attention_heads * c.attention_head_dim, n_layer=c.num_layers,
        n_head=c.num_attention_heads, n_classes=c.num_embeds_ada_norm,
        class_dropout=0.1 if n_rows > c.num_embeds_ada_norm else 0.0,
        out_ch=c.out_channels or c.in_channels))
    return dict(mine=mine, name_map=diffusers_name_map,
                mine_input_fn=lambda d: (d["hidden_states"], d["timestep"], d["class_labels"]),
                hook_pairs=[("pos_embed", "patch_embed")] +
                           [(f"transformer_blocks.{i}", f"blocks.{i}") for i in range(c.num_layers)])
