"""A6b·A7b · Kimi-K3 텍스트 모델: KDA 3 : gated MLA 1 + Attention Residuals + latent MoE.

HF 원본: moonshotai/Kimi-K3 원격 코드 `modeling_kimi_linear.py` (KimiLinearModel). 파라미터 이름은 원본과 같다.
KDA 핵심 연산은 원본에서 fla(flash-linear-attention) Triton 커널 `chunk_kda` 를 부르는데,
여기서는 같은 수식을 순수 PyTorch 로 쓴다 (recurrent = 정의 그대로, chunk = 학습용 행렬곱).

블록 구성 (한 층)
1. 토큰 섞기 — 층 종류가 둘이다 (K3 는 KDA 3 층마다 MLA 1 층).
   - KDA (Kimi Delta Attention): Gated DeltaNet 의 메모리 S [Dk, Dv] 갱신과 같지만
     망각 게이트가 head 마다 스칼라가 아니라 **key 채널마다** 다르다 (fine-grained decay).
         S ← diag(exp(g_t)) · S                 (g_t ∈ [lower_bound, 0) 채널별 망각)
         S ← S + β_t k_t ⊗ (v_t − Sᵀk_t)        (delta rule)
         o_t = Sᵀ q_t
     q/k/v 에 짧은 causal conv + SiLU, q/k 는 L2 정규화, 출력은 RMSNorm · sigmoid 게이트.
     게이트 g 는 저랭크(f_a → f_b) 투영 후  lower_bound · sigmoid(exp(A_log) · (g + dt_bias)).
   - gated MLA: DeepSeek-V3 식 latent attention (q 저랭크, kv 를 kv_lora_rank 로 압축 후 head 별 복원).
     K3 는 RoPE 를 쓰지 않는다 (NoPE, 위치 정보는 KDA 가 담당). "rope" 채널은 head 끼리 공유하는 key 채널일 뿐이다.
     출력에 sigmoid 게이트(g_proj)를 곱한다.
2. MoE — sigmoid 점수 + bias 로 top-k 선택(aux-loss-free), 가중치는 bias 없는 점수로 재정규화 × scale.
   latent MoE: routed expert 는 hidden 을 작은 latent 차원으로 줄여(down) 그 안에서 계산하고 RMSNorm 후 다시 키운다(up).
   shared expert 는 원래 차원에서 계산. 활성함수 SiTU: β·tanh(gate/β)·sigmoid(gate) × β'·tanh(up/β').
   앞쪽 first_k_dense 층은 dense MLP.
3. Attention Residuals (AttnRes) — residual 을 단순히 더해 가지 않는다.
   층들을 attn_res_block_size 개씩 블록으로 묶고, 끝난 블록의 합(과 embedding)을 목록으로 들고 다닌다.
   각 sub-layer 입력 = softmax(score) 로 [이전 블록들, 현재 블록 부분합] 을 섞은 값.
   score = RMSnorm(값) · (norm.weight ⊙ proj.weight) — 층마다 학습되는 query 벡터 하나로 깊이 방향 attention.

검증용 기본값: 원본 KimiLinearModel 에는 lm_head 가 없으므로 tie_embeddings=True (실제 K3 는 untied).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput, RMSNorm, attention

KDA, MLA = "kda", "mla"


@dataclass
class KimiK3Config:
    vocab_size: int = 128
    d_model: int = 256
    n_layer: int = 4
    layer_types: list[str] | None = None      # None → [kda, kda, kda, mla] 반복
    # KDA
    kda_heads: int = 4
    kda_head_dim: int = 64
    conv_kernel: int = 4
    gate_lower_bound: float | None = -5.0     # None → -exp(A_log)·softplus(g + dt_bias)
    full_rank_gate: bool = True               # 출력 게이트: True g_proj / False g_a_proj → g_b_proj
    delta_mode: str = "chunk"                 # "chunk"(학습용) / "recurrent"(토큰별, 해석용)
    chunk_size: int = 64
    # gated MLA (NoPE)
    n_head: int = 4
    q_lora_rank: int | None = 64              # None → q_proj 하나
    kv_lora_rank: int = 32
    qk_nope_dim: int = 32
    qk_rope_dim: int = 16                     # head 공유 key 채널 (RoPE 는 적용 안 함)
    v_head_dim: int = 32
    mla_output_gate: bool = True
    # MoE
    n_experts: int | None = 8                 # None → 전부 dense
    top_k: int = 2
    moe_inter: int = 64
    n_shared: int | None = 1
    latent_dim: int | None = 128              # routed expert 의 latent 차원 (None → d_model 그대로)
    latent_norm: bool = True
    router_act: str = "sigmoid"
    renormalize: bool = True
    route_scale: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    first_k_dense: int = 1
    moe_layer_freq: int = 1
    # dense MLP / 활성함수
    d_ff: int = 512
    hidden_act: str = "situ"                  # "situ" / "silu"
    situ_beta: float | None = 4.0
    situ_linear_beta: float | None = 25.0
    # AttnRes
    attn_res_block_size: int | None = 2       # None → 일반 residual
    # 기타
    rms_eps: float = 1e-5
    tie_embeddings: bool = True
    # ablation 스위치 (기본값 = 원본)
    use_attn_res: bool = True                 # False: 일반 pre-norm residual
    use_kda: bool = True                      # False: 모든 층을 MLA 로
    use_moe: bool = True                      # False: 모든 FFN 을 dense MLP 로
    use_latent_moe: bool = True               # False: routed expert 를 원래 차원에서 계산
    use_output_gate: bool = True              # False: MLA g_proj / KDA sigmoid 출력 게이트 제거

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [MLA if (i + 1) % 4 == 0 else KDA for i in range(self.n_layer)]
        assert len(self.layer_types) == self.n_layer
        if not self.use_kda:
            self.layer_types = [MLA] * self.n_layer

    @property
    def attn_res(self) -> bool:
        return self.use_attn_res and self.attn_res_block_size is not None


# ---------------------------------------------------------------- 활성함수 / MLP

def act_mul(gate, up, c: KimiK3Config):
    """SiTU (fp32 계산): β·tanh(gate/β)·sigmoid(gate) × β'·tanh(up/β'). silu 면 silu(gate)·up."""
    if c.hidden_act != "situ":
        assert c.hidden_act == "silu", c.hidden_act
        return F.silu(gate) * up
    dtype, beta = gate.dtype, c.situ_beta or 1.0
    g, u = gate.float(), up.float()
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if c.situ_linear_beta is not None:
        u = c.situ_linear_beta * torch.tanh(u / c.situ_linear_beta)
    return (a * u).to(dtype)


class MLP(nn.Module):
    """dense 층 · shared expert (원본 KimiMLP)."""

    def __init__(self, c: KimiK3Config, d: int, inter: int):
        super().__init__()
        self.c = c
        self.gate_proj = nn.Linear(d, inter, bias=False)
        self.up_proj = nn.Linear(d, inter, bias=False)
        self.down_proj = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        return self.down_proj(act_mul(self.gate_proj(x), self.up_proj(x), self.c))


class Expert(nn.Module):
    """routed expert (원본 KimiBlockSparseMLP: w1=gate, w3=up, w2=down)."""

    def __init__(self, c: KimiK3Config, d: int, inter: int):
        super().__init__()
        self.c = c
        self.w1 = nn.Linear(d, inter, bias=False)
        self.w2 = nn.Linear(inter, d, bias=False)
        self.w3 = nn.Linear(d, inter, bias=False)

    def forward(self, x):
        return self.w2(act_mul(self.w1(x), self.w3(x), self.c))


# ---------------------------------------------------------------- MoE

class Router(nn.Module):
    """sigmoid 점수 + bias(선택에만) → top-k. n_group > topk_group 이면 그룹을 먼저 고른다 (DeepSeek-V3 식)."""

    def __init__(self, c: KimiK3Config):
        super().__init__()
        self.c = c
        self.weight = nn.Parameter(torch.empty(c.n_experts, c.d_model))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.register_buffer("e_score_correction_bias", torch.zeros(c.n_experts))
        self.score_probe = nn.Identity()   # hook 지점: 점수 [N, E]
        self.route_probe = nn.Identity()   # hook 지점: 선택된 expert [N, k]

    def forward(self, x):                  # x [N, D]
        c, N = self.c, x.shape[0]
        logits = F.linear(x.float(), self.weight.float())
        scores = self.score_probe(logits.sigmoid() if c.router_act == "sigmoid" else logits.softmax(dim=1))
        choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if c.n_group > 1 and c.n_group > c.topk_group:
            gs = choice.view(N, c.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)          # 그룹 점수 = 상위 2개 합
            gidx = torch.topk(gs, k=c.topk_group, dim=-1, sorted=False)[1]
            gmask = torch.zeros_like(gs).scatter_(1, gidx, 1)
            smask = gmask.unsqueeze(-1).expand(N, c.n_group, c.n_experts // c.n_group).reshape(N, -1)
            choice = choice.masked_fill(~smask.bool(), float("-inf"))
        idx = self.route_probe(torch.topk(choice, k=c.top_k, dim=-1, sorted=False)[1])
        w = scores.gather(1, idx)
        if c.top_k > 1 and c.renormalize:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return idx, w * c.route_scale


class SparseMoE(nn.Module):
    """routed expert (latent 차원) + shared expert (원래 차원)."""

    def __init__(self, c: KimiK3Config):
        super().__init__()
        self.c = c
        self.latent = c.use_latent_moe and c.latent_dim is not None
        d = c.latent_dim if self.latent else c.d_model
        self.experts = nn.ModuleList(Expert(c, d, c.moe_inter) for _ in range(c.n_experts))
        self.gate = Router(c)
        if c.n_shared is not None:
            self.shared_experts = MLP(c, c.d_model, c.moe_inter * c.n_shared)
        if self.latent:
            self.routed_expert_down_proj = nn.Linear(c.d_model, d, bias=False)
            self.routed_expert_up_proj = nn.Linear(d, c.d_model, bias=False)
            if c.latent_norm:
                self.routed_expert_norm = RMSNorm(d, c.rms_eps)

    def forward(self, x):
        shape = x.shape
        idx, w = self.gate(x.reshape(-1, shape[-1]))
        h = x.reshape(-1, shape[-1])
        if self.latent:
            h = self.routed_expert_down_proj(h)
        y = torch.zeros_like(h)
        for e in range(self.c.n_experts):
            tok, slot = torch.where(idx == e)
            if tok.numel():
                y.index_add_(0, tok, (self.experts[e](h[tok]).to(w.dtype) * w[tok, slot, None]).to(y.dtype))
        if self.latent:
            if self.c.latent_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)
        y = y.view(shape)
        if self.c.n_shared is not None:
            y = y + self.shared_experts(x)
        return y


# ---------------------------------------------------------------- KDA

def l2norm(x, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def kda_gate(g, A_log, dt_bias, lower_bound: float | None):
    """g [B, T, H, K] (저랭크 투영 출력) → 채널별 log 망각 [B, T, H, K] (fp32, ≤ 0)."""
    H = g.shape[-2]
    g = g.float() + dt_bias.float().view(H, -1)
    A = A_log.float().view(H, 1).exp()
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(A * g)       # [lower_bound, 0) 로 자연스럽게 제한
    return -A * F.softplus(g)


def kda_recurrent(q, k, v, g, beta, state=None, return_states: bool = False):
    """정의 그대로: 토큰 하나씩 S 를 갱신. q/k [B, H, T, K], v [B, H, T, V], g [B, H, T, K], beta [B, H, T].

    q 는 미리 1/sqrt(K) 스케일. return_states=True 면 매 시점 S_t [B, H, T, K, V] 도 돌려준다.
    """
    B, H, T, K = k.shape
    S = torch.zeros(B, H, K, v.shape[-1], device=v.device, dtype=v.dtype) if state is None else state
    out, states = torch.zeros_like(v), []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None]                                  # 채널별 망각
        kv_mem = (k[:, :, t, :, None] * S).sum(-2)                           # Sᵀk
        S = S + (beta[:, :, t, None] * k[:, :, t])[..., None] * (v[:, :, t] - kv_mem)[..., None, :]
        out[:, :, t] = (q[:, :, t, :, None] * S).sum(-2)                     # Sᵀq
        if return_states:
            states.append(S)
    return out, S, (torch.stack(states, 2) if return_states else None)


def kda_chunk(q, k, v, g, beta, chunk_size: int = 64, state=None):
    """같은 수식을 chunk 단위로 (fla naive_chunk_kda 와 같은 알고리즘, 벡터화). 입력 형식은 recurrent 와 같다.

    채널별 망각이라 chunk 안 "attention" 이 Σ_d q_c,d · exp(G_c,d − G_i,d) · k_i,d 가 된다.
    exp 인자는 항상 ≤ 0 이 되도록 mask 를 먼저 씌워 강한 망각에서도 안정적이다 ([C, C, K] 텐서를 만든다).
    """
    B, H, T, K = k.shape
    V, C = v.shape[-1], chunk_size
    pad = (C - T % C) % C                         # padding 칸: g=0(망각 없음), β=0(쓰기 없음)
    q, k, v, g = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, g))
    beta = F.pad(beta, (0, pad))
    N = (T + pad) // C
    q, k, v, g = (x.reshape(B, H, N, C, -1) for x in (q, k, v, g))
    beta = beta.reshape(B, H, N, C)

    G = g.cumsum(-2)                                                   # chunk 안 누적 log 망각
    i = torch.arange(C, device=q.device)
    strict, incl = i[:, None] > i[None, :], i[:, None] >= i[None, :]
    diff = G[..., :, None, :] - G[..., None, :, :]                     # [B, H, N, C(c), C(i), K] = G_c − G_i
    Dk = diff.masked_fill(~strict[..., None], float("-inf")).exp()
    Dq = diff.masked_fill(~incl[..., None], float("-inf")).exp()
    L = torch.einsum("bhncd,bhncid,bhnid->bhnci", k, Dk, k) * beta[..., None]   # delta rule 연쇄 계수
    Aqk = torch.einsum("bhncd,bhncid,bhnid->bhnci", q, Dq, k)                    # chunk 안 attention
    A = L + torch.eye(C, device=q.device, dtype=q.dtype)
    u = torch.linalg.solve_triangular(A, v * beta[..., None], upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(A, k * G.exp() * beta[..., None], upper=False, unitriangular=True)

    S = torch.zeros(B, H, K, V, device=v.device, dtype=v.dtype) if state is None else state
    out = torch.zeros_like(u)
    for n in range(N):
        v_new = u[:, :, n] - w[:, :, n] @ S                             # 이전 chunk 기억으로 예측한 부분 제거
        out[:, :, n] = (q[:, :, n] * G[:, :, n].exp()) @ S + Aqk[:, :, n] @ v_new
        gl = G[:, :, n, -1]                                             # chunk 전체 망각 [B, H, K]
        S = S * gl.exp()[..., None] + ((gl[:, :, None] - G[:, :, n]).exp() * k[:, :, n]).transpose(-1, -2) @ v_new
    return out.reshape(B, H, -1, V)[:, :, :T], S


class KDAOutNorm(nn.Module):
    """RMSNorm(o) · w · sigmoid(gate) (원본 fla FusedRMSNormGated, activation='sigmoid'). gate=None 이면 게이트 없음."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.gate_probe = nn.Identity()   # hook 지점: sigmoid 출력 게이트 [B, T, H, D]

    def forward(self, x, gate=None):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        if gate is not None:
            h = h * self.gate_probe(torch.sigmoid(gate.float()))
        return h.to(x.dtype)


class KimiDeltaAttention(nn.Module):
    def __init__(self, c: KimiK3Config):
        super().__init__()
        self.c = c
        self.h, self.d = c.kda_heads, c.kda_head_dim
        P = self.h * self.d
        self.q_proj = nn.Linear(c.d_model, P, bias=False)
        self.k_proj = nn.Linear(c.d_model, P, bias=False)
        self.v_proj = nn.Linear(c.d_model, P, bias=False)
        # 짧은 causal depthwise conv (원본 fla ShortConvolution = nn.Conv1d, weight [P, 1, K])
        self.q_conv1d = nn.Conv1d(P, P, c.conv_kernel, groups=P, bias=False, padding=c.conv_kernel - 1)
        self.k_conv1d = nn.Conv1d(P, P, c.conv_kernel, groups=P, bias=False, padding=c.conv_kernel - 1)
        self.v_conv1d = nn.Conv1d(P, P, c.conv_kernel, groups=P, bias=False, padding=c.conv_kernel - 1)
        self.A_log = nn.Parameter(torch.empty(self.h).uniform_(1, 16).log())
        self.f_a_proj = nn.Linear(c.d_model, self.d, bias=False)          # 망각 게이트: 저랭크 (rank = head_dim)
        self.f_b_proj = nn.Linear(self.d, P, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(P))
        self.b_proj = nn.Linear(c.d_model, self.h, bias=False)            # 쓰기 강도 β
        self.gated = c.use_output_gate
        if self.gated and c.full_rank_gate:
            self.g_proj = nn.Linear(c.d_model, P, bias=False)
        elif self.gated:
            self.g_a_proj = nn.Linear(c.d_model, self.d, bias=False)
            self.g_b_proj = nn.Linear(self.d, P, bias=False)
        self.o_norm = KDAOutNorm(self.d, c.rms_eps)
        self.o_proj = nn.Linear(P, c.d_model, bias=False)
        # hook 지점 (해석용): 채널별 log 망각 g [B, T, H, K], 쓰기 강도 β [B, T, H]
        self.decay_probe = nn.Identity()
        self.beta_probe = nn.Identity()

    def _conv(self, conv, x):
        T = x.shape[1]
        return F.silu(conv(x.transpose(1, 2))[..., :T]).transpose(1, 2)    # 앞쪽 padding → causal

    def forward(self, x, return_states: bool = False):
        B, T, _ = x.shape
        q = self._conv(self.q_conv1d, self.q_proj(x)).reshape(B, T, self.h, self.d)
        k = self._conv(self.k_conv1d, self.k_proj(x)).reshape(B, T, self.h, self.d)
        v = self._conv(self.v_conv1d, self.v_proj(x)).reshape(B, T, self.h, self.d)
        g = self.f_b_proj(self.f_a_proj(x)).reshape(B, T, self.h, self.d)
        beta = self.beta_probe(self.b_proj(x).float().sigmoid())
        g = self.decay_probe(kda_gate(g, self.A_log, self.dt_bias, self.c.gate_lower_bound))

        # [B, H, T, D] fp32, q/k L2 정규화 후 q 만 1/sqrt(K)
        q, k = l2norm(q.float()), l2norm(k.float())
        q = q * self.d ** -0.5
        q, k, v, g = (t.transpose(1, 2).float() for t in (q, k, v, g))
        beta = beta.transpose(1, 2)

        states = None
        if self.c.delta_mode == "recurrent" or return_states:
            o, _, states = kda_recurrent(q, k, v, g, beta, return_states=return_states)
        else:
            o, _ = kda_chunk(q, k, v, g, beta, self.c.chunk_size)
        o = o.transpose(1, 2).to(x.dtype)                                  # [B, T, H, D]
        gate = None
        if self.gated:
            gate = self.g_proj(x) if self.c.full_rank_gate else self.g_b_proj(self.g_a_proj(x))
            gate = gate.reshape(B, T, self.h, self.d)
        out = self.o_proj(self.o_norm(o, gate).reshape(B, T, -1))
        return (out, states) if return_states else out


# ---------------------------------------------------------------- gated MLA

class MLAAttention(nn.Module):
    """latent attention (NoPE). q·k = q_nope·k_nope(head 별) + q_rope·k_rope(head 공유)."""

    def __init__(self, c: KimiK3Config):
        super().__init__()
        self.c = c
        self.h, self.dn, self.dr, self.dv = c.n_head, c.qk_nope_dim, c.qk_rope_dim, c.v_head_dim
        self.dq = self.dn + self.dr
        if c.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(c.d_model, c.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(c.q_lora_rank, 1e-6)              # 원본도 eps 기본값(1e-6)
            self.q_b_proj = nn.Linear(c.q_lora_rank, self.h * self.dq, bias=False)
        else:
            self.q_proj = nn.Linear(c.d_model, self.h * self.dq, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(c.d_model, c.kv_lora_rank + self.dr, bias=False)
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank, 1e-6)
        self.kv_b_proj = nn.Linear(c.kv_lora_rank, self.h * (self.dn + self.dv), bias=False)
        self.o_proj = nn.Linear(self.h * self.dv, c.d_model, bias=False)
        self.gated = c.mla_output_gate and c.use_output_gate
        if self.gated:
            self.g_proj = nn.Linear(c.d_model, self.h * self.dv, bias=False)
        self.attn_probs = nn.Identity()   # hook 지점: [B, H, T, T]
        self.gate_probe = nn.Identity()   # hook 지점: sigmoid 출력 게이트 [B, T, H*Dv]

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))) if self.c.q_lora_rank is not None else self.q_proj(x)
        q = q.view(B, T, self.h, self.dq).transpose(1, 2)
        k_pass, k_rot = self.kv_a_proj_with_mqa(x).split([self.c.kv_lora_rank, self.dr], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(B, T, self.h, self.dn + self.dv).transpose(1, 2)
        k_nope, v = kv.split([self.dn, self.dv], dim=-1)
        k = torch.cat([k_nope, k_rot[:, None].expand(B, self.h, T, self.dr)], dim=-1)
        y = attention(q, k, v, causal=True, probs_hook=self.attn_probs, scale=self.dq ** -0.5)
        y = y.transpose(1, 2).reshape(B, T, -1)
        if self.gated:
            y = y * self.gate_probe(torch.sigmoid(self.g_proj(x)))
        return self.o_proj(y)


# ---------------------------------------------------------------- Attention Residuals

def attn_res(partial, blocks, proj: nn.Linear, norm: RMSNorm, probe: nn.Module):
    """[이전 블록들 [B,T,n,D], 현재 부분합 [B,T,D]] 를 softmax 가중합. 확률 [B, T, n+1] 은 probe 로 캡처."""
    v = torch.cat([blocks, partial.unsqueeze(-2)], dim=-2)
    vf = v.float()
    key = vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + norm.eps)
    query = norm.weight.float() * proj.weight.squeeze(0).float()          # 층마다 학습되는 query 하나
    probs = probe((key * query).sum(-1).softmax(-1))
    return (probs.unsqueeze(-2) @ vf).squeeze(-2).to(v.dtype)


# ---------------------------------------------------------------- 층 / 모델

class Layer(nn.Module):
    def __init__(self, c: KimiK3Config, i: int):
        super().__init__()
        self.c, self.i = c, i
        self.layer_type = c.layer_types[i]
        self.self_attn = KimiDeltaAttention(c) if self.layer_type == KDA else MLAAttention(c)
        if c.use_moe and c.n_experts is not None and i >= c.first_k_dense and i % c.moe_layer_freq == 0:
            self.block_sparse_moe = SparseMoE(c)
        else:
            self.mlp = MLP(c, c.d_model, c.d_ff)
        self.input_layernorm = RMSNorm(c.d_model, c.rms_eps)
        self.post_attention_layernorm = RMSNorm(c.d_model, c.rms_eps)
        if c.attn_res:
            self.self_attention_res_norm = RMSNorm(c.d_model, c.rms_eps)
            self.mlp_res_norm = RMSNorm(c.d_model, c.rms_eps)
            self.self_attention_res_proj = nn.Linear(c.d_model, 1, bias=False)
            self.mlp_res_proj = nn.Linear(c.d_model, 1, bias=False)
            self.self_attention_res_probe = nn.Identity()   # hook 지점: attention 입력의 깊이 방향 확률 [B, T, n+1]
            self.mlp_res_probe = nn.Identity()              # hook 지점: FFN 입력의 깊이 방향 확률

    def ffn(self, x):
        return self.block_sparse_moe(x) if hasattr(self, "block_sparse_moe") else self.mlp(x)

    def forward(self, h, blocks):
        """일반: h = residual stream. AttnRes: h = 현재 블록 부분합, blocks = 끝난 블록들 [B, T, n, D]."""
        if not self.c.attn_res:
            h = h + self.self_attn(self.input_layernorm(h))
            return h + self.ffn(self.post_attention_layernorm(h)), blocks
        partial = h
        x = partial if blocks.shape[-2] == 0 else attn_res(
            partial, blocks, self.self_attention_res_proj, self.self_attention_res_norm, self.self_attention_res_probe)
        if self.i % self.c.attn_res_block_size == 0:                     # 새 블록 시작: 지난 부분합을 목록에 추가
            blocks = torch.cat([blocks, partial.unsqueeze(-2)], dim=-2)
            partial = None
        a = self.self_attn(self.input_layernorm(x))
        partial = a if partial is None else partial + a
        x = attn_res(partial, blocks, self.mlp_res_proj, self.mlp_res_norm, self.mlp_res_probe)
        return partial + self.ffn(self.post_attention_layernorm(x)), blocks


class KimiK3(nn.Module):
    def __init__(self, c: KimiK3Config):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.n_layer))
        self.norm = RMSNorm(c.d_model, c.rms_eps)
        if c.attn_res:
            self.output_attn_res_norm = RMSNorm(c.d_model, c.rms_eps)
            self.output_attn_res_proj = nn.Linear(c.d_model, 1, bias=False)
            self.output_res_probe = nn.Identity()   # hook 지점: 최종 출력의 깊이 방향 확률
        self.lm_head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> LMOutput:
        x = self.embed_tokens(input_ids)
        B, T, D = x.shape
        blocks = x.new_zeros(B, T, 0, D)
        for layer in self.layers:
            x, blocks = layer(x, blocks)
        if self.config.attn_res:
            x = attn_res(x, blocks, self.output_attn_res_proj, self.output_attn_res_norm, self.output_res_probe)
        h = self.norm(x)
        logits = self.lm_head(h) if self.lm_head is not None else h @ self.embed_tokens.weight.T
        return LMOutput(h, logits)


# ---------------------------------------------------------------- 검증

def config_from_hf(c) -> KimiK3Config:
    """c = KimiLinearConfig (Kimi-K3 의 text_config)."""
    lac = c.linear_attn_config or {}
    g = lambda k, d=None: getattr(c, k, d)  # noqa: E731
    return KimiK3Config(
        vocab_size=c.vocab_size, d_model=c.hidden_size, n_layer=c.num_hidden_layers,
        layer_types=[KDA if c.is_kda_layer(i) else MLA for i in range(c.num_hidden_layers)],
        kda_heads=lac.get("num_heads", 1), kda_head_dim=lac.get("head_dim", 1),
        conv_kernel=lac.get("short_conv_kernel_size", 4), gate_lower_bound=lac.get("gate_lower_bound"),
        full_rank_gate=lac.get("use_full_rank_gate", False),
        n_head=c.num_attention_heads, q_lora_rank=c.q_lora_rank, kv_lora_rank=c.kv_lora_rank,
        qk_nope_dim=c.qk_nope_head_dim, qk_rope_dim=c.qk_rope_head_dim, v_head_dim=c.v_head_dim,
        mla_output_gate=bool(g("mla_use_output_gate", False)),
        n_experts=c.num_experts, top_k=c.num_experts_per_token or 1, moe_inter=c.moe_intermediate_size or 1,
        n_shared=c.num_shared_experts, latent_dim=g("routed_expert_hidden_size"),
        latent_norm=bool(g("latent_moe_use_norm", False)), router_act=c.moe_router_activation_func,
        renormalize=c.moe_renormalize, route_scale=c.routed_scaling_factor,
        n_group=g("num_expert_group", 1), topk_group=g("topk_group", 1),
        first_k_dense=c.first_k_dense_replace, moe_layer_freq=g("moe_layer_freq", 1),
        d_ff=c.intermediate_size, hidden_act=c.hidden_act,
        situ_beta=g("activation_situ_beta"), situ_linear_beta=g("activation_situ_linear_beta"),
        attn_res_block_size=g("attn_res_block_size"), rms_eps=c.rms_norm_eps)


def verify_pair(ref) -> dict:
    """ref = 원격 코드 KimiLinearModel (이름이 같아 name_map 불필요)"""
    mine = KimiK3(config_from_hf(ref.config))
    pairs = []
    for i, layer in enumerate(ref.layers):
        ffn = "block_sparse_moe" if hasattr(layer, "block_sparse_moe") else "mlp"
        pairs += [(f"layers.{i}.self_attn", f"layers.{i}.self_attn"),
                  (f"layers.{i}.{ffn}", f"layers.{i}.{ffn}"),
                  (f"layers.{i}", f"layers.{i}")]
    return dict(mine=mine, name_map=None, hook_pairs=pairs)
