"""A5b · MiniMax-M3 MSA (MiniMax Sparse Attention): GQA 위에 얹은 블록 top-k sparse attention.

원본
- 정답 함수: github MiniMax-AI/MSA (commit 80434d7) 의 순수 PyTorch reference.
  `sparse_attention_ref` (고른 블록만 보는 attention), `ref_tile_max_score` (블록별 최대 점수).
- 층 연결 (투영 이름, norm, RoPE, index 가지, top-k 규칙): sglang `models/minimax_m3.py`,
  `minimax_sparse_ops`, `kernels/ops/attention/minimax_sparse/prefill/flash_with_topk_idx.py`.
  HF MiniMaxAI/MiniMax-M3 에는 config 만 있고 modeling 코드가 없다.
파라미터 이름은 HF 체크포인트(language_model.model.layers.N.self_attn.*)와 같다.

MSA 층 하나의 흐름
1. Main 가지 = 보통 GQA. q 는 H 개 head, k/v 는 G 개 head. head 별 RMSNorm(gemma 식 1+w) → partial RoPE
   (앞 rotary_dim 채널만, half-rotate).
2. Index 가지 = 아주 가벼운 점수기. GQA 그룹마다 index query head 1 개(index_q_proj), 모든 그룹이 공유하는
   index key head 1 개(index_k_proj). main 과 같은 norm + RoPE 후 토큰 점수 s = q_idx·k_idx / sqrt(d_idx).
3. 블록 점수: 키를 block_size 개씩 묶고, 블록 안에서 causal 하게 보이는 토큰의 최대 점수 → [B, G, T, 블록 수].
4. top-k: 그룹마다 점수가 큰 블록 topk 개. 질의가 속한 local 블록(local_blocks 개)과 맨 앞 init 블록은
   점수와 무관하게 항상 포함 (M3: init 0, local 1). softmax 는 순서를 안 바꾸므로 exp 없이 raw 점수로 고른다.
5. Main 가지는 고른 블록 안의 (causal 하게 보이는) 토큰에만 softmax attention. 같은 그룹 q head 들은 블록 집합을 공유.
→ 질의당 attention 비용이 문맥 길이와 무관하게 topk · block_size 로 고정된다.

학습 (논문 3.2): top-k 는 미분이 안 되므로 index 가지는 KL 손실로만 배운다.
  teacher = 그룹 안 head 들의 main attention 확률 평균 (detach), student = 같은 토큰 집합 위 index 점수 softmax.
  index 가지 입력은 stopgrad(x) 라서 KL 이 backbone 으로 새지 않는다. index_kl=True 면 attention.kl_loss 에 기록.

MiniMax-M3 실제 구성: 60층 중 앞 3층은 dense GQA, 나머지 57층이 MSA. FFN 은 MoE(swigluoai) 지만
이 파일은 attention 연구용이라 dense SwiGLU 하나로 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput, rotate_half

FULL, SPARSE = "full_attention", "sparse_attention"


@dataclass
class MiniMaxMSAConfig:
    vocab_size: int = 128
    d_model: int = 256
    n_layer: int = 4
    layer_types: list[str] | None = None   # None → 앞 n_dense 층 full, 나머지 sparse (M3: 3 + 57)
    n_dense: int = 1
    # main 가지 (GQA)
    n_head: int = 8
    n_kv_head: int = 2
    head_dim: int = 32
    rotary_dim: int = 16                   # head 의 앞 rotary_dim 채널에만 RoPE (M3: 128 중 64)
    rope_theta: float = 10000.0
    # index 가지
    index_heads: int | None = None         # None → n_kv_head (M3: 4 = KV head 수, 그룹당 1개)
    index_dim: int = 32
    block_size: int = 128
    topk: int = 16                         # 그룹·질의마다 고르는 블록 수 (local/init 포함)
    init_blocks: int = 0                   # 맨 앞 블록 강제 포함 개수
    local_blocks: int = 1                  # 질의가 속한 블록부터 거꾸로 강제 포함 개수
    score_type: str = "max"                # 블록 점수: "max" (M3) / "lse"
    # 공통
    d_ff: int = 512
    rms_eps: float = 1e-6
    use_gemma_norm: bool = True            # norm 출력 × (1 + w), w 는 0 초기화
    tie_embeddings: bool = False
    # ablation 스위치 (기본값 = 원본)
    use_sparse: bool = True                # False: index 는 계산하되 main 은 dense causal (= indexer warmup 단계)
    use_qk_norm: bool = True               # False: q/k, index q/k 의 head 별 norm 제거
    # 학습용 (forward 값에는 영향 없음)
    index_kl: bool = False                 # True: 층마다 KL(teacher ‖ index) 를 attention.kl_loss 에 기록
    detach_index_input: bool = True        # 학습 중 index 가지 입력을 stopgrad

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [FULL] * min(self.n_layer, self.n_dense) + [SPARSE] * max(0, self.n_layer - self.n_dense)
        if self.index_heads is None:
            self.index_heads = self.n_kv_head
        assert len(self.layer_types) == self.n_layer
        assert self.index_heads % self.n_kv_head == 0
        assert self.init_blocks + self.local_blocks <= self.topk
        assert self.rotary_dim <= min(self.head_dim, self.index_dim) and self.rotary_dim % 2 == 0


# ---------------------------------------------------------------- 기본 부품

class RMSNorm(nn.Module):
    """gemma=True: norm(x) · (1 + w) (sglang GemmaRMSNorm). gemma=False: norm(x) · w."""

    def __init__(self, dim: int, eps: float = 1e-6, gemma: bool = True):
        super().__init__()
        self.eps, self.gemma = eps, gemma
        self.weight = nn.Parameter(torch.zeros(dim) if gemma else torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.gemma:
            return (h * (1.0 + self.weight.float())).to(dtype)
        return self.weight * h.to(dtype)


def rope_inv_freq(rotary_dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))


def rope_cos_sin(T: int, inv_freq: torch.Tensor):
    """→ cos, sin [T, 1, rotary_dim] (앞 절반/뒤 절반 같은 주파수, half-rotate 방식). [B, T, h, D] 에 broadcast."""
    f = torch.arange(T, device=inv_freq.device).float()[:, None] * inv_freq[None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos()[:, None, :], emb.sin()[:, None, :]


def apply_partial_rope(x, cos, sin):
    """x [B, T, h, D] 의 앞 rotary_dim 채널만 회전, 나머지는 위치 정보 없이 그대로."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos.to(x.dtype) + rotate_half(xr) * sin.to(x.dtype), xp], dim=-1)


# ---------------------------------------------------------------- MSA attention

class MSAAttention(nn.Module):
    """layer_type == SPARSE 면 MSA (index 가지 + 블록 top-k), FULL 이면 같은 투영의 dense GQA."""

    def __init__(self, c: MiniMaxMSAConfig, layer_type: str = SPARSE):
        super().__init__()
        self.c, self.sparse = c, layer_type == SPARSE
        H, G, D = c.n_head, c.n_kv_head, c.head_dim
        self.q_proj = nn.Linear(c.d_model, H * D, bias=False)
        self.k_proj = nn.Linear(c.d_model, G * D, bias=False)
        self.v_proj = nn.Linear(c.d_model, G * D, bias=False)
        self.o_proj = nn.Linear(H * D, c.d_model, bias=False)
        if c.use_qk_norm:   # head 별 (모든 head 가 같은 weight 공유)
            self.q_norm = RMSNorm(D, c.rms_eps, c.use_gemma_norm)
            self.k_norm = RMSNorm(D, c.rms_eps, c.use_gemma_norm)
        if self.sparse:
            self.index_q_proj = nn.Linear(c.d_model, c.index_heads * c.index_dim, bias=False)   # 그룹당 query 1개
            self.index_k_proj = nn.Linear(c.d_model, c.index_dim, bias=False)                   # 공유 key 1개
            if c.use_qk_norm:
                self.index_q_norm = RMSNorm(c.index_dim, c.rms_eps, c.use_gemma_norm)
                self.index_k_norm = RMSNorm(c.index_dim, c.rms_eps, c.use_gemma_norm)
            # hook 지점: 블록 점수 [B, Gi, T, NB] (안 보이는 블록 = -1e30), 고른 블록 [B, Gi, T, NB] (bool)
            self.index_score = nn.Identity()
            self.block_select = nn.Identity()
        self.attn_probs = nn.Identity()   # hook 지점: main attention 확률 [B, H, T, T] (안 고른 칸 = 0)
        self.register_buffer("inv_freq", rope_inv_freq(c.rotary_dim, c.rope_theta), persistent=False)
        self.kl_loss = None

    # ---- index 가지

    def index_scores(self, x, cos, sin):
        """→ 토큰 점수 [B, Gi, T, T] (causal 밖 -inf), 블록 점수 [B, Gi, T, NB]."""
        c = self.c
        B, T, _ = x.shape
        xi = x.detach() if (self.training and c.detach_index_input) else x
        iq = self.index_q_proj(xi).view(B, T, c.index_heads, c.index_dim)
        ik = self.index_k_proj(xi).view(B, T, 1, c.index_dim)
        if c.use_qk_norm:
            iq, ik = self.index_q_norm(iq), self.index_k_norm(ik)
        iq, ik = apply_partial_rope(iq, cos, sin), apply_partial_rope(ik, cos, sin)
        s = torch.einsum("btgd,bsd->bgts", iq.float(), ik[:, :, 0].float()) * c.index_dim ** -0.5
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        s = s.masked_fill(~causal, float("-inf"))
        nb = -(-T // c.block_size)
        sb = F.pad(s, (0, nb * c.block_size - T), value=float("-inf")).view(B, c.index_heads, T, nb, c.block_size)
        blk = sb.amax(-1) if c.score_type == "max" else torch.logsumexp(sb, dim=-1)
        return s, blk

    def select_blocks(self, blk):
        """블록 점수 [B, Gi, T, NB] → 고른 블록 mask (sglang _topk_index_kernel 과 같은 규칙)."""
        c = self.c
        T, nb = blk.shape[-2], blk.shape[-1]
        qb = torch.arange(T, device=blk.device) // c.block_size
        j = torch.arange(nb, device=blk.device)
        vis = j[None, :] <= qb[:, None]                                   # 질의 블록까지만 (causal)
        s = blk.masked_fill(~vis, float("-inf"))
        s = torch.where(vis & (j[None, :] < c.init_blocks), 1e30, s)      # init 블록 강제
        s = torch.where(vis & (j[None, :] > qb[:, None] - c.local_blocks), 1e29, s)   # local 블록 강제
        idx = s.topk(min(c.topk, nb), dim=-1).indices
        sel = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, True)
        return sel & vis                                                  # 보이는 블록이 topk 보다 적으면 그만큼만

    # ---- forward

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        c, x = self.c, hidden_states
        B, T, _ = x.shape
        H, G, D = c.n_head, c.n_kv_head, c.head_dim
        cos, sin = rope_cos_sin(T, self.inv_freq)
        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, G, D)
        v = self.v_proj(x).view(B, T, G, D)
        if c.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                 # [B, h, T, D]

        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        group_allow = causal.expand(B, G, T, T)                          # 그룹별로 볼 수 있는 토큰
        if self.sparse:
            s_tok, blk = self.index_scores(x, cos, sin)
            blk = self.index_score(blk.clamp(min=-1e30))
            sel = self.block_select(self.select_blocks(blk))             # [B, Gi, T, NB]
            tok = sel.repeat_interleave(c.block_size, dim=-1)[..., :T] & causal
            r = c.index_heads // G                                       # index head 가 더 많으면 그룹 안 합집합
            if c.use_sparse:
                group_allow = tok.view(B, G, r, T, T).any(2)

        rep = H // G
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        allow = group_allow.repeat_interleave(rep, dim=1)                # [B, H, T, T]
        scores = (q * D ** -0.5) @ k.transpose(-2, -1)
        scores = scores.masked_fill(~allow, float("-inf"))
        probs = self.attn_probs(scores.float().softmax(dim=-1).to(q.dtype))
        y = probs @ v

        self.kl_loss = None
        if self.sparse and c.index_kl:
            self.kl_loss = self._kl(probs, s_tok, group_allow)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, H * D))

    def _kl(self, probs, s_tok, group_allow):
        """KL(teacher ‖ index): teacher = 그룹 내 head 평균 main 확률 (detach), index = 같은 토큰 위 softmax(s_idx)."""
        c = self.c
        B, H, T, _ = probs.shape
        G, r = c.n_kv_head, c.index_heads // c.n_kv_head
        teacher = probs.detach().float().view(B, G, H // G, T, T).mean(2).repeat_interleave(r, dim=1)
        tok = group_allow.repeat_interleave(r, dim=1)
        logp = s_tok.masked_fill(~tok, float("-inf")).log_softmax(-1).masked_fill(~tok, 0.0)
        return (torch.xlogy(teacher, teacher) - teacher * logp).sum(-1).mean()


# ---------------------------------------------------------------- 블록 / 모델

class MLP(nn.Module):
    def __init__(self, c: MiniMaxMSAConfig):
        super().__init__()
        self.gate_proj = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.up_proj = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down_proj = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c: MiniMaxMSAConfig, layer_type: str):
        super().__init__()
        self.self_attn = MSAAttention(c, layer_type)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.d_model, c.rms_eps, c.use_gemma_norm)
        self.post_attention_layernorm = RMSNorm(c.d_model, c.rms_eps, c.use_gemma_norm)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class MiniMaxMSA(nn.Module):
    def __init__(self, c: MiniMaxMSAConfig):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Layer(c, t) for t in c.layer_types)
        self.norm = RMSNorm(c.d_model, c.rms_eps, c.use_gemma_norm)
        self.lm_head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.kl_loss = None
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> LMOutput:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        h = self.norm(x)
        logits = self.lm_head(h) if self.lm_head is not None else h @ self.embed_tokens.weight.T
        if self.config.index_kl:   # 층별 KL 합 (학습 루프에서 LM loss 에 계수를 곱해 더한다)
            kls = [l.self_attn.kl_loss for l in self.layers if l.self_attn.kl_loss is not None]
            self.kl_loss = torch.stack(kls).sum() if kls else h.new_zeros(())
        return LMOutput(h, logits)


# ---------------------------------------------------------------- 검증

def _get(c, k, default=None):
    v = c.get(k, default) if isinstance(c, dict) else getattr(c, k, default)
    return default if v is None else v


def config_from_hf(c) -> MiniMaxMSAConfig:
    """MiniMax-M3 config.json 의 text_config (또는 같은 키를 가진 객체) → 내 config."""
    c = _get(c, "text_config", c)
    sp = _get(c, "sparse_attention_config", {})
    n = _get(c, "num_hidden_layers")
    freq = list(_get(sp, "sparse_attention_freq", [1] * n))
    head_dim = _get(c, "head_dim", _get(c, "hidden_size") // _get(c, "num_attention_heads"))
    rotary_dim = _get(c, "rotary_dim", int(head_dim * _get(c, "partial_rotary_factor", 1.0)))
    theta = _get(c, "rope_theta", _get(_get(c, "rope_parameters", {}), "rope_theta", 10000.0))  # transformers 5 는 rope_parameters 로 옮김
    return MiniMaxMSAConfig(
        vocab_size=_get(c, "vocab_size"), d_model=_get(c, "hidden_size"), n_layer=n,
        layer_types=[SPARSE if f else FULL for f in freq],
        n_head=_get(c, "num_attention_heads"), n_kv_head=_get(c, "num_key_value_heads"), head_dim=head_dim,
        rotary_dim=rotary_dim, rope_theta=float(theta),
        index_heads=_get(sp, "sparse_num_index_heads"), index_dim=_get(sp, "sparse_index_dim", head_dim),
        block_size=_get(sp, "sparse_block_size", 128), topk=_get(sp, "sparse_topk_blocks", 16),
        init_blocks=_get(sp, "sparse_init_block", 0), local_blocks=_get(sp, "sparse_local_block", 1),
        score_type=_get(sp, "sparse_score_type", "max"),
        d_ff=_get(c, "dense_intermediate_size", _get(c, "intermediate_size")), rms_eps=_get(c, "rms_norm_eps", 1e-6),
        use_gemma_norm=bool(_get(c, "use_gemma_norm", False)), use_qk_norm=bool(_get(c, "use_qk_norm", True)),
        tie_embeddings=bool(_get(c, "tie_word_embeddings", False)))


def verify_pair(ref) -> dict:
    """ref = stages/a5b_minimax_msa.RefMSAAttention (MSA 저장소 reference 함수를 감싼 attention 층).

    원본 전체 모델 코드가 없어서 attention 층 단위로 비교한다. 이름이 같아 name_map 불필요.
    """
    mine = MSAAttention(config_from_hf(ref.config), SPARSE)
    return dict(mine=mine, name_map=None,
                hook_pairs=[("index_score", "index_score"), ("block_select", "block_select")])
