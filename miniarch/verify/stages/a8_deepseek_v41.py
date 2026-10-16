"""A8 · DeepSeek-V4.1-Flash (CSA2 Full/Reindex/Reuse · 2단계 indexer · Engram · mHC · MoE) 검증 케이스.

원본 = HF deepseek-ai/DeepSeek-V4.1-Flash `inference/model.py` + `engram.py` (커밋 고정, transformers main 에는 아직 없음).
원본은 TileLang 커널(inference/kernel.py)을 import 하므로, `kernel` 모듈만 같은 수식의 fp32 PyTorch 구현으로 바꿔 끼운다.
  - act_quant / fp4_act_quant (inplace=True): fp8·fp4 양자화 후 즉시 복원하는 QAT 흉내 → 항등 (fp32 기준값이므로 양자화 오차 제거)
  - sparse_attn: 인덱스로 KV 를 모아 sink 포함 softmax (커널과 같이 max 는 유효 칸만, 초기값 -1e30) → gather + softmax
  - hc_split_sinkhorn: pre/post/comb 분리 + Sinkhorn → 같은 식을 텐서 연산으로
  - fp8_gemm / fp4_gemm: dtype="bf16", expert_dtype=None 이라 호출되지 않음 (호출되면 에러)
커널 외에 바꾼 것 (dtype 저장 형식만, 수식 동일):
  - Engram 표(ParallelEngramEmbedding): fp8 weight + e8m0 scale → fp32 Parameter, 복원 후 bf16 으로 내리는 cast 제거
  - 그 밖의 bf16 가중치는 check_equivalence 의 .float() 로 fp32
원본 코드를 고치지 않고 붙인 것: forward 를 감싸 그동안만 torch 기본 device 를 입력 device 로 (원본은 window index 를 기본 device 에 만듦).
"""
from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F
from torch import nn

from ..cases import BATCH, VOCAB, Case

STAGE = "A8"
SLUG = "deepseek_v41"
TITLE = "DeepSeek-V4.1-Flash (CSA2 · Engram)"
INTRO = """
원본: HF `deepseek-ai/DeepSeek-V4.1-Flash`의 `inference/model.py` + `engram.py` (커밋 `5254448` 고정).
transformers main에는 아직 `deepseek_v41`이 없어서 DeepSeek 공식 추론 코드를 받아 씁니다.
이 코드는 TileLang 커널을 쓰므로 **커널 함수만** 같은 수식의 fp32 PyTorch 구현으로 바꿔 끼웁니다
(fp8/fp4 양자화 흉내는 항등으로 둡니다. 자세한 목록은 `miniarch/verify/stages/a8_deepseek_v41.py` 맨 위 설명).

내 구현: `miniarch/models/deepseek_v41.py`. 가중치 이름은 실제 체크포인트 키와 같습니다.

V4에서 달라진 부품
- CSA2: sliding 층 + 압축 층. 압축 층은 모드가 셋입니다.
  Full(압축 KV와 indexer key를 직접 만듦) / Reindex(KV·key는 재사용, top-k만 다시 고름) / Reuse(선택까지 재사용)
- 2단계 indexer: ratio 1 묶음에서 candidate 출처 층이 block 단위로 먼저 추리고, 뒤 층은 그 안에서 top-k
- 압축: 토큰 r개를 softmax gate 가중합 (겹침·위치 bias 없음), r = 1이면 투영만. 압축 층 RoPE는 YaRN
- Engram: 2..n-gram 해시로 큰 표에서 벡터를 꺼내 줄기마다 gate를 걸어 residual에 더함
- mHC의 pre 계수가 한 sublayer 늦게 쓰임, MoE는 hash 라우팅 없이 sqrt-softplus + bias

축소 config (모든 경로가 한 번씩 쓰이도록 고름)
- 7층: [sliding, r2 Full, r2 Reindex, r2 Reuse, r1 Full(candidate 출처), r1 Reindex(candidate 사용), r1 Reuse]
- Engram은 1, 4번 층. 3-gram, head 2개. 토큰 정규화 표는 대소문자만 다른 128개 토큰짜리 작은 tokenizer로 만듦
- 시퀀스 41: window 8 밖, 홀수 길이라 ratio 2의 덜 찬 마지막 묶음, candidate block(4칸) 끝 padding까지 지나감
- index top-6 < 볼 수 있는 칸 수, candidate top-3 block < 전체 block 수 → 실제로 고르고 버리는 경로가 쓰임
- YaRN: original_seq_len 4096, factor 16 → rope 채널 일부가 보간 구간에 들어감
- 마지막 위치 logits와, 전 위치의 층별 출력(engram/attn/ffn/층)과 최종 norm 출력을 비교합니다.
"""
PIP = []
HEADER = [("deepseek-ai/DeepSeek-V4.1-Flash", r"^layers\.(1|2|24)\.", "")]
CAUSAL = True
SELF_CHECKS = [
    ("""
### 커널 대체 확인

원본 비교의 기준값이 커널 대체에 달려 있으므로, 대체한 `sparse_attn`이 정의(인덱스로 고른 칸 + sink에 대한 softmax)와 같은지,
`hc_split_sinkhorn`의 comb가 이중확률행렬인지 따로 확인합니다.
""", """
K = S.torch_kernels()
V.set_deterministic(0)
b_, s_, h_, d_, n_, k_ = 2, 5, 3, 8, 12, 6
q_, kv_, sink_ = torch.randn(b_, s_, h_, d_), torch.randn(b_, n_, d_), torch.randn(h_)
idx_ = torch.stack([torch.randperm(n_)[:k_] for _ in range(b_ * s_)]).view(b_, s_, k_)
idx_[..., -2:] = -1
o1 = K.sparse_attn(q_, kv_, sink_, idx_.int(), 0.3)
mask_ = torch.zeros(b_, s_, n_ + 1, dtype=torch.bool).scatter_(-1, torch.where(idx_ >= 0, idx_, n_), True)[..., :n_]
lg_ = (torch.einsum("bshd,bnd->bshn", q_, kv_) * 0.3).masked_fill(~mask_[:, :, None], float("-inf"))
lg_ = torch.cat([lg_, sink_.view(1, 1, -1, 1).expand(b_, s_, h_, 1)], -1)
o2 = torch.einsum("bshn,bnd->bshd", lg_.softmax(-1)[..., :-1], kv_)
err = (o1 - o2).abs().max().item()
report("sparse_attn 대체 == 정의", err < 1e-5, err)

pre_, post_, comb_ = K.hc_split_sinkhorn(torch.randn(2, 3, 24), torch.rand(3) + 0.5, torch.randn(24) * 0.1, 4, 20, 1e-6)
err = max((comb_.sum(-1) - 1).abs().max().item(), (comb_.sum(-2) - 1).abs().max().item())
report("Sinkhorn comb 행/열 합 = 1", err < 1e-3, err)
"""),
    ("""
### Engram n-gram 해시 id

해시는 학습 파라미터가 없는 전처리라 가중치 복사로는 검사되지 않습니다. 원본 `NgramHashState`와 내 `NgramHash`가
같은 정수 id를 내는지 직접 비교합니다 (소수 bucket 배치, 곱셈 계수, XOR 누적, pad 처리).
""", """
from miniarch.models import deepseek_v41 as D41
ref, inp = S.ref()
m = D41.verify_pair(ref)["mine"]
ids = inp["input_ids"]
with torch.no_grad():
    a, b = ref.engram_hash(ids, 0), m.engram_hash(ids)
err = float((a - b).abs().max().item()) if a.shape == b.shape else float("nan")
print("hash ids", tuple(a.shape), "서로 다른 id 수:", a.unique().numel())
report("Engram 해시 id 일치", a.shape == b.shape and err == 0, err)
"""),
    ("""
### 층 모드 · attention map · sink · Engram gate hook 확인

- 층마다 모드(sliding/full/reindex/reuse)와, 질의 하나가 평균 몇 개의 압축 칸을 보는지 출력합니다.
- `attn_probs`: [B, H, T, T + 압축 칸]. 각 행 합 + sink 확률 = 1이어야 합니다.
- `gate_probe`: Engram gate [B, T, 줄기].
""", """
from miniarch.models import deepseek_v41 as D41
ref, inp = S.ref()
m = D41.verify_pair(ref)["mine"].to(dev).eval()
names = [n for n, _ in m.named_modules() if n.endswith(("attn_probs", "sink_probe", "gate_probe"))]
x = inp["input_ids"].to(dev)
T_ = x.shape[1]
with V.capture(m, names) as st, torch.no_grad():
    m(x)
for i, layer in enumerate(m.layers):
    p, s = st[f"layers.{i}.attn.attn_probs"], st[f"layers.{i}.attn.sink_probe"]
    err = (p.sum(-1) + s - 1).abs().max().item()
    nc = (p[..., T_:] > 0).float().sum(-1).mean().item()
    print(f"layer {i} {layer.attn.mode:<8} ratio={layer.attn.ratio}  probs={tuple(p.shape)}  압축 칸 평균 {nc:.1f}개  평균 sink={s.mean():.3f}")
    report(f"layer {i} 확률 합", err < 1e-5, err)
for n in names:
    if n.endswith("gate_probe"):
        print(n, "평균 gate", round(st[n].mean().item(), 3))
"""),
    ("""
### ablation 스위치

나중에 부품별 비교 실험에 쓸 스위치를 하나씩 끄고 forward가 정상(유한값)인지 확인합니다.
""", """
import dataclasses
from miniarch.models import deepseek_v41 as D41
ref, inp = S.ref()
base = D41.config_from_hf(ref.args)
x = inp["input_ids"].to(dev)
for sw in ["use_engram", "use_mhc", "use_compressed_kv", "use_indexer", "use_candidates", "use_sink", "use_moe"]:
    m = D41.DeepseekV41(dataclasses.replace(base, **{sw: False})).to(dev).eval()
    with torch.no_grad():
        out = m(x).logits
    report(f"{sw}=False forward", bool(torch.isfinite(out).all()), 0.0)
"""),
]

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "5254448103b6254fc646dacbca1cbbf20e9216c1"
_LOADED: dict = {}


# ---------------------------------------------------------------- kernel.py 대체 (fp32 PyTorch)

def torch_kernels() -> types.ModuleType:
    """inference/kernel.py 와 같은 이름·인자의 함수들. 양자화는 항등, 나머지는 같은 수식."""
    k = types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if inplace:      # fp8 양자화 → 복원 (QAT 흉내): fp32 기준값에서는 항등
            return x
        raise NotImplementedError("fp8 가중치 GEMM 경로는 쓰지 않음 (dtype='bf16')")

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
        if inplace:
            return x
        raise NotImplementedError("fp4 가중치 GEMM 경로는 쓰지 않음 (expert_dtype=None)")

    def fp8_gemm(*a, **kw):
        raise NotImplementedError("fp8_gemm 은 호출되지 않아야 함")

    def fp4_gemm(*a, **kw):
        raise NotImplementedError("fp4_gemm 은 호출되지 않아야 함")

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        """q [b,s,h,d], kv [b,n,d], topk_idxs [b,s,k] (-1 = 빈 칸) → o [b,s,h,d].
        커널: max 는 유효 칸에서만(초기 -1e30), 분모에 exp(sink - max) 추가."""
        b = q.shape[0]
        idx = topk_idxs.long().to(kv.device)
        valid = idx >= 0
        g = kv[torch.arange(b, device=kv.device)[:, None, None], idx.clamp_min(0)].float()   # [b,s,k,d]
        sc = torch.einsum("bshd,bskd->bshk", q.float(), g) * softmax_scale
        sc = sc.masked_fill(~valid[:, :, None, :], float("-inf"))
        mx = sc.amax(-1, keepdim=True).clamp_min(-1e30)
        p = torch.exp(sc - mx)
        den = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float().view(1, 1, -1, 1) - mx)
        return (torch.einsum("bshk,bskd->bshd", p, g) / den).to(q.dtype).contiguous()   # 원본은 o.view(...) 를 함

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        n = hc_mult
        pre = torch.sigmoid(mixes[..., :n] * hc_scale[0] + hc_base[:n]) + eps
        post = 2 * torch.sigmoid(mixes[..., n:2 * n] * hc_scale[1] + hc_base[n:2 * n])
        comb = (mixes[..., 2 * n:] * hc_scale[2] + hc_base[2 * n:]).unflatten(-1, (n, n))
        comb = torch.exp(comb - comb.amax(-1, keepdim=True))
        comb = comb / comb.sum(-1, keepdim=True) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    for f in (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn):
        setattr(k, f.__name__, f)
    return k


def _engram_embed_fp32(self, indices):
    """ParallelEngramEmbedding.forward 와 같고, fp8 저장 → fp32, 마지막 bf16 cast 만 뺐다."""
    mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
    local = (indices - self.vocab_start_idx).masked_fill(mask, 0)
    values = F.embedding(local, self.weight)
    scales = F.embedding(local, self.scale)
    values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
    return values.flatten(-2).masked_fill(mask.unsqueeze(-1), 0)


def load_reference():
    """원본 inference 코드(model.py, engram.py, vision.py, image_processor.py)를 받아 kernel 만 바꿔 import."""
    if _LOADED:
        return _LOADED["model"], _LOADED["engram"]
    import importlib.util
    from huggingface_hub import hf_hub_download
    paths = {n: hf_hub_download(REPO, f"inference/{n}.py", revision=REVISION)
             for n in ("model", "engram", "vision", "image_processor")}

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    bare = ("kernel", "engram", "vision", "image_processor")
    saved = {n: sys.modules.get(n) for n in bare}
    sys.modules["kernel"] = torch_kernels()
    try:
        eng = load("engram", paths["engram"])
        load("vision", paths["vision"])
        load("image_processor", paths["image_processor"])
        model = load("deepseek_v41_inference_model", paths["model"])
    finally:   # 흔한 이름이라 전역 sys.modules 는 원래대로 (model 모듈은 이미 객체를 잡고 있음)
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v
    model.ParallelEngramEmbedding.forward = _engram_embed_fp32
    _LOADED.update(model=model, engram=eng)
    return model, eng


def _tiny_tokenizer():
    """Engram 토큰 정규화 표용 작은 tokenizer: 'w5' 와 'W5' 처럼 대소문자만 다른 토큰 쌍 → 정규화 vocab 64."""
    from tokenizers import Tokenizer, models
    half = VOCAB // 2
    vocab = {("W" if i >= half else "w") + str(i % half): i for i in range(VOCAB)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="w0"))

    class Tok:
        backend_tokenizer = tk

        def __len__(self):
            return VOCAB
    return Tok()


def _device_hooks(m, M):
    """원본은 window index 를 torch 기본 device 에 만든다 → forward 동안만 기본 device 를 입력 device 로.
    hook 대신 forward 를 감싸 try/finally 로 되돌린다 (원본 forward 가 예외를 내도 기본 device 가 cuda 로 남지 않게)."""
    orig = m.forward

    def forward(input_ids, *args, **kwargs):
        prev = torch.get_default_device()
        torch.set_default_device(input_ids.device)
        M.get_window_topk_idxs.cache_clear()
        try:
            return orig(input_ids, *args, **kwargs)
        finally:
            torch.set_default_device(prev)
            M.get_window_topk_idxs.cache_clear()

    m.forward = forward


def ref():
    M, E = load_reference()
    base = dict(
        max_batch_size=BATCH, max_seq_len=64, temperature=0.0, dtype="bf16", expert_dtype=None,
        vocab_size=VOCAB, dim=64, moe_inter_dim=32, n_layers=7, n_mtp_layers=0, n_heads=4,
        n_routed_experts=4, n_shared_experts=1, n_activated_experts=2, score_func="sqrtsoftplus",
        route_scale=1.5, swiglu_limit=2.0,
        q_lora_rank=32, head_dim=32, rope_head_dim=16, norm_eps=1e-20, o_groups=2, o_lora_rank=16,
        window_size=8, compress_ratios=(0, 2, 2, 2, 1, 1, 1),
        kv_source_layers=(1, 4), index_source_layers=(1, 2, 4, 5),
        compress_rope_theta=160000.0, original_seq_len=4096, rope_theta=10000.0, rope_factor=16,
        beta_fast=32, beta_slow=1,
        index_n_heads=4, index_head_dim=32, index_topk=6,
        candidate_source_layer=4, candidate_topk_blocks=3, candidate_block_size=4,
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        engram_layer_ids=(1, 4), engram_max_ngram_size=3, engram_vocab_size=50,
        engram_n_heads=2, engram_head_dim=32, engram_pad_id=2, vision_n_layers=0)
    tok = _tiny_tokenizer()
    _, cvocab = E.build_compressed_token_map(tok)
    layout = E.EngramLayout.from_args(M.ModelArgs(**base, engram_num_embeddings=(0, 0),
                                                  engram_compressed_vocab_size=cvocab))
    rows = tuple(sum(sum(p) for p in layer) + 3 for layer in layout.primes)   # 소수 bucket 합 + 여분 3행
    args = M.ModelArgs(**base, engram_num_embeddings=rows, engram_compressed_vocab_size=cvocab)
    m = M.Transformer(args, tok)
    for layer in m.layers:   # Engram 표: fp8 + e8m0 scale → fp32
        if layer.engram is not None:
            e = layer.engram.embed
            n, d = e.weight.shape
            e.weight = nn.Parameter(torch.empty(n, d))
            e.scale = nn.Parameter(torch.empty(n, d // e.block_size))
    m.float()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():    # 원본은 torch.empty 로 만들기만 하므로 직접 초기화 (활성값이 O(1) 이 되도록)
        for name, p in m.named_parameters():
            r = torch.randn(p.shape, generator=g)
            if "hc_" in name and name.endswith("_base"):
                p.copy_(r * 0.1)
            elif "hc_" in name and name.endswith("_scale"):
                p.copy_(torch.rand(p.shape, generator=g) + 0.5)
            elif name.endswith(("norm.weight", "q_weight", "k_weight")):
                p.copy_(1 + 0.1 * r)
            elif name.endswith("embed.scale"):
                p.copy_(torch.rand(p.shape, generator=g) + 0.5)
            elif name.endswith("attn_sink"):
                p.copy_(r * 0.5)
            elif name.endswith("gate.bias"):
                p.copy_(r * 0.1)
            elif name == "embed.weight" or name.endswith("embed.weight"):
                p.copy_(r)
            else:
                p.copy_(r * p.shape[-1] ** -0.5)
    m.args = args
    _device_hooks(m, M)
    ids = torch.randint(0, VOCAB, (BATCH, 41), generator=torch.Generator().manual_seed(0))
    return m, {"input_ids": ids}


CASE = Case(STAGE, "DeepSeek-V4.1 전체: CSA2(sliding/Full/Reindex/Reuse) + 2단계 indexer + YaRN + Engram + mHC + MoE",
            "DeepSeek-V4.1-Flash", "HF deepseek-ai/DeepSeek-V4.1-Flash inference/model.py@5254448",
            ref, "miniarch.models.deepseek_v41", atol=1e-4)
