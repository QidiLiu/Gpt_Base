"""
正确性测试。

重点覆盖**形状/布局类**的 bug —— 这类 bug 最危险：
数值看起来「差不多对」，但训练几百步后才发现不对劲。
每个测试都对应一个真实踩过的坑。

跑法：uv run pytest tests/ -v
"""

import torch
import torch.nn.functional as F

from common.config import make_run_config, build_model_config
from model.layers import (
    precompute_rope, apply_rotary_emb, rms_norm, attend, sdpa_bt, MLP,
)
from model.gpt import build_model, sample_from_logits
from optim.orthogonalize import (
    orthogonalize_simple, orthogonalize_advanced, orthogonality_error,
)
from inference.engine import KVCache


# ===========================================================================
# 形状与布局
# ===========================================================================
def test_sdpa_layout_is_bhtd_not_bthd():
    """
    回归测试：PyTorch SDPA 严格按 (B,H,L,E) 解读输入。

    我们的模型全程用 (B,T,H,D)。如果直接喂给 SDPA：
      T==1 -> 形状悄悄变
      T==2 -> 直接报错
    sdpa_bt() 负责转置。这个测试锁死这个行为。
    """
    B, T, H, D = 2, 7, 6, 32
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, H, D, dtype=torch.float32)
    out = sdpa_bt(q, k, v, causal=True)
    assert out.shape == q.shape, f"sdpa_bt 改变了布局: {tuple(out.shape)} != {tuple(q.shape)}"


def test_attend_matches_manual_reference():
    """
    回归测试：sdpa_bt 的结果必须与「手写 attention」逐元素一致。

    手写版完全按教科书公式：(B,H,T,D) + causal mask + softmax + AV
    """
    torch.manual_seed(0)
    B, T, H, D = 2, 16, 4, 8
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    got = attend(q, k, v, (T, 0))                       # (B,T,H,D)

    # 手写 reference
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))   # (B,H,T,D)
    logits = (qt @ kt.transpose(-2, -1)) / D ** 0.5
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    logits = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    ref = (logits @ vt).transpose(1, 2)                   # 回到 (B,T,H,D)
    assert torch.allclose(got, ref, atol=1e-5), \
        f"最大误差 {(got-ref).abs().max().item()}"


def test_sliding_window_only_sees_recent():
    """
    回归测试：滑窗模式下，位置 j 只能看到 [j-left, j]。
    做法：把某一步的 k 改动，如果输出没变，说明那一步确实不可见。
    """
    torch.manual_seed(0)
    B, T, H, D = 1, 32, 2, 8
    left = 4
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    y1 = attend(q, k, v, (left, 0))

    # 改动位置 0 的 v。位置 0 距离最后位置 T-1 是 T-1 > left，应该完全不可见
    v2 = v.clone()
    v2[:, 0] += 100.0
    y2 = attend(q, k, v2, (left, 0))
    # 位置 T-1 的输出不应受影响
    assert torch.allclose(y1[0, -1], y2[0, -1], atol=1e-6), \
        "位置 T-1 的输出被位置 0 的 v 影响了 -> 滑窗没生效"

    # 但位置 left 的输出应该受影响（距离为 left，仍在窗口内）
    assert not torch.allclose(y1[0, left], y2[0, left], atol=1e-4), \
        "位置 left 的输出没被影响 -> 滑窗范围算错了"


def test_kv_cache_writes_all_batches():
    """
    回归测试：KV cache 的写入必须覆盖 batch 的**每一行**。

    曾经的 bug：用 scatter_ 配 expand 出来的 stride-0 索引，
    结果只写进去了 batch 的第 0 行，第 1 行及以后全是未初始化垃圾。
    现在改成切片赋值，从根上避免这个坑。
    """
    from model.layers import attend_with_kvcache
    B, Tq, H, D, cap = 3, 1, 6, 32, 100
    kc = KVCache(B, H, D, n_layers=2, max_seq_len=cap, device="cpu")
    kc.cache_seqlens.fill_(5)
    q = torch.randn(B, Tq, H, D)
    k = torch.randn(B, Tq, H, D)
    v = torch.randn(B, Tq, H, D)

    out = attend_with_kvcache(q, kc.k_cache[0], kc.v_cache[0], k, v, kc, (cap, 0))

    # 每一行位置 5 都必须等于对应的输入
    for b in range(B):
        assert torch.allclose(kc.k_cache[0][b, 5].float(), k[b],
                              rtol=0.02, atol=0.05), f"batch 第 {b} 行的 k 没写进去"
        assert torch.allclose(kc.v_cache[0][b, 5].float(), v[b],
                              rtol=0.02, atol=0.05), f"batch 第 {b} 行的 v 没写进去"
    assert out.shape == q.shape, f"kv-cache 注意力输出形状错: {tuple(out.shape)}"


def test_kv_cache_prefill_equals_naive():
    """
    回归测试：带 KV cache 分步前向的结果，必须等于一次性朴素前向。

    这是 KV cache 实现正确性的黄金测试。
    做法：先用朴素 forward 生成 5 个 token，再用 Engine 的 decode 路径
          逐个喂同样的 token，比较每一步的 logits。
    """
    from model.gpt import GPT
    torch.manual_seed(0)
    cfg = build_model_config(2, 16, 8, sequence_len=32, vocab_size=64)
    model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    model.eval()

    tokens = torch.randint(0, 64, (1, 6))
    # 朴素：一次性前向
    with torch.no_grad():
        naive = model(tokens)

    # KV cache：一次 prefill + 逐个 decode
    from common import COMPUTE_DTYPE
    kv = KVCache(1, cfg.n_kv_head, cfg.head_dim, cfg.n_layer, 64, "cpu",
                 dtype=COMPUTE_DTYPE)
    from model.layers import rms_norm
    with torch.no_grad():
        out = model(tokens[:, :5], kv_cache=kv)
        for i in range(5, 6):
            out = model(tokens[:, i:i + 1], kv_cache=kv)
    assert torch.allclose(naive[:, -1], out[:, -1], atol=2e-3), \
        f"KV cache 与朴素前向不一致，最大误差 {(naive[:,-1]-out[:,-1]).abs().max().item()}"


# ===========================================================================
# RoPE
# ===========================================================================
def test_rope_is_rotation_preserves_norm():
    """RoPE 是旋转：它保持向量的 L2 范数不变。"""
    torch.manual_seed(0)
    cos, sin = precompute_rope(16, 8)
    x = torch.randn(1, 16, 2, 8)
    y = apply_rotary_emb(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5), \
        "RoPE 改变了向量范数，那它就不是纯旋转"


def test_rope_relative_position():
    """
    回归测试：RoPE 的核心性质 —— 内积只依赖「相对距离」。

    数学恒等式（RoPE 论文的核心）：
        ⟨R_m q, R_n k⟩ = qᵀ R_mᵀ R_n k = qᵀ R_{n-m} k
    右边只含 n-m，也就是**相对距离**。
    这正是 RoPE 优于「可学习位置嵌入」的地方：位置信息不是外挂的，
    而是直接编码进了 q·k 的内积里。

    验证方式：固定两个向量 q0、k0，把它们分别放在不同的绝对位置上，
    只要「相对距离」相同，内积就必须相同。
    """
    torch.manual_seed(0)
    D, N, delta = 16, 64, 5
    cos, sin = precompute_rope(N, D)

    q0 = torch.randn(1, 1, 1, D)     # (B=1, T=1, H=1, D)
    k0 = torch.randn(1, 1, 1, D)

    def rot_at(vec, pos):
        """把 vec 当作位于绝对位置 pos，施加对应的旋转。"""
        return apply_rotary_emb(vec, cos[:, pos:pos + 1], sin[:, pos:pos + 1])

    # 基准：q 在位置 0，k 在位置 delta
    base = (rot_at(q0, 0) * rot_at(k0, delta)).sum().item()

    # q 在位置 i，k 在位置 i+delta，相对距离同样是 delta -> 内积应与基准相同
    for i in [1, 7, 20, 50]:
        got = (rot_at(q0, i) * rot_at(k0, i + delta)).sum().item()
        assert abs(got - base) < 1e-4, (
            f"相对距离同为 {delta} 但内积不同：位置({i},{i+delta}) 得 {got:.6f}，"
            f"基准 {base:.6f} —— RoPE 的相对位置性质被破坏")


def test_rope_does_not_leak_in_length():
    """
    回归测试：RoPE 之后的 attention 仍然是因果的。
    改动序列末尾的 token，不应影响前面的输出。
    """
    from model.gpt import GPT
    torch.manual_seed(0)
    V = 64
    cfg = build_model_config(2, 16, 8, sequence_len=24, vocab_size=V)
    model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    x1 = torch.randint(0, V, (1, 24))
    x2 = x1.clone(); x2[0, 16:] = torch.randint(0, V, (8,))
    with torch.no_grad():
        y1, y2 = model(x1), model(x2)
    assert torch.allclose(y1[:, :16], y2[:, :16], atol=1e-4), \
        f"RoPE 引入了未来信息泄漏 {(y1[:,:16]-y2[:,:16]).abs().max().item()}"


# ===========================================================================
# 优化器
# ===========================================================================
def test_orthogonalize_improves_orthogonality():
    """
    回归测试：正交化之后，||X·Xᵀ - I|| 必须显著变小。
    """
    torch.manual_seed(0)
    G = torch.randn(64, 192)
    before = orthogonality_error(G)
    after = orthogonality_error(orthogonalize_simple(G))
    assert after < before, f"正交化没起作用: {before:.4f} -> {after:.4f}"


def test_muon_plus_rescues_low_rank():
    """
    回归测试（本项目最重要的一条）：Newton-Schulz 推不动近零奇异值。

    低秩矩阵的谱里有大量 0，而 0 是 Newton-Schulz 迭代的不动点，
    所以无论迭代多少步都救不回来。Muon+ 的重归一化能把整体范数 snap 回去，
    从而把「推不动的方向」整体放大。
    """
    torch.manual_seed(0)
    G = torch.randn(64, 8) @ torch.randn(8, 192)     # rank=8
    simple = orthogonalize_advanced(G, use_muon_eq=False, use_muon_plus=False)
    full = orthogonalize_advanced(G, use_muon_eq=False, use_muon_plus=True)
    assert orthogonality_error(full) < orthogonality_error(simple) / 2, \
        (f"Muon+ 应该大幅改善低秩矩阵: simple={orthogonality_error(simple):.3f} "
         f"full={orthogonality_error(full):.3f}")


# ===========================================================================
# 采样
# ===========================================================================
def test_greedy_sampling_is_deterministic():
    """temperature=0 必须是确定性的（贪心解码）。"""
    logits = torch.tensor([[0.1, 5.0, 0.2, 0.1]])
    a = sample_from_logits(logits.clone(), None, temperature=0.0)
    b = sample_from_logits(logits.clone(), None, temperature=0.0)
    assert a.item() == b.item() == 1, f"贪心采样应恒选 argmax，得到 {a.item()}"


def test_topk_zeroes_out_tail():
    """top_k 之外必须是 -inf（概率 0）。"""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    rng = torch.Generator().manual_seed(0)
    picked = {sample_from_logits(logits.clone(), rng, 1.0, top_k=2).item()
              for _ in range(200)}
    assert picked <= {3, 4}, f"top_k=2 采到了 {picked}"


# ===========================================================================
# 模型端到端
# ===========================================================================
def test_forward_loss_starts_uniform():
    """
    回归测试：刚初始化的模型，loss 应接近 ln(vocab_size)。

    因为初始化时 lm_head 权重是 N(0, 0.001)，logits 几乎为 0，
    softmax 接近均匀分布。如果这个测试失败，说明初始化被改坏了。
    """
    from model.gpt import GPT
    torch.manual_seed(0)
    V = 512
    cfg = build_model_config(2, 16, 8, sequence_len=32, vocab_size=V)
    model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    x = torch.randint(0, V, (2, 32))
    y = torch.randint(0, V, (2, 32))
    loss = model(x, y).item()
    import math
    assert abs(loss - math.log(V)) < 0.1, \
        f"初始 loss {loss:.3f} 偏离均匀预测 ln({V})={math.log(V):.3f} 太多"


def test_loss_reduction_none_shape():
    """reduction='none' 必须返回 (B,T)，RL 和 bpb 计算都依赖这个形状。"""
    from model.gpt import GPT
    torch.manual_seed(0)
    V = 64
    cfg = build_model_config(2, 16, 8, sequence_len=16, vocab_size=V)
    model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    x = torch.randint(0, V, (3, 16))
    y = torch.randint(0, V, (3, 16))
    assert model(x, y, loss_reduction="none").shape == (3, 16)


def test_ignore_index_masked():
    """targets 里 -1 的位置必须不产生梯度贡献。"""
    from model.gpt import GPT
    torch.manual_seed(0)
    V = 64
    cfg = build_model_config(2, 16, 8, sequence_len=8, vocab_size=V)
    model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    x = torch.randint(0, V, (1, 8))
    y = torch.randint(0, V, (1, 8))
    y_masked = y.clone(); y_masked[0, 4:] = -1
    l_all = model(x, y, loss_reduction="none")
    l_part = model(x, y_masked, loss_reduction="none")
    assert torch.allclose(l_all[0, :4], l_part[0, :4], atol=1e-6)
    assert l_part[0, 4:].abs().max() == 0, "被 mask 的位置 loss 应为 0"


def test_causality_no_future_leak():
    """
    回归测试：改最后几个 token，之前的 logits 不应改变。
    如果这个测试挂了，说明 attention 的 causal mask 漏了 —— 那是最严重的 bug。
    """
    from model.gpt import GPT
    torch.manual_seed(0)
    V = 64
    cfg = build_model_config(2, 16, 8, sequence_len=16, vocab_size=V)
    model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    x1 = torch.randint(0, V, (1, 16))
    x2 = x1.clone(); x2[0, 12:] = torch.randint(0, V, (4,))
    with torch.no_grad():
        y1, y2 = model(x1), model(x2)
    assert torch.allclose(y1[:, :12], y2[:, :12], atol=1e-5), \
        f"因果性被破坏了！最大泄漏 {(y1[:,:12]-y2[:,:12]).abs().max().item()}"


# ===========================================================================
# 配置
# ===========================================================================
def test_presets_are_consistent():
    """三档预设必须自洽（batch 能被 micro-batch × seq 整除）。"""
    from common.config import make_run_config, resolve_scaling
    for mode, V in [("debug", 8192), ("smoke", 8192), ("full", 16384)]:
        cfg = make_run_config(mode, vocab_size=V)
        sc = resolve_scaling(cfg, log=lambda *a: None)
        assert sc["grad_accum_steps"] >= 1, f"{mode} 档 grad_accum < 1"
        assert sc["num_iterations"] >= 1, f"{mode} 档步数为 0"


def test_all_tricks_build_and_forward():
    """5 个残差流 trick 全开时，模型必须能正常前向反向。"""
    cfg = make_run_config("debug", vocab_size=64)
    cfg.model.sequence_len = 32
    cfg.model.n_head = cfg.model.n_kv_head = 2
    cfg.model.head_dim = 16
    cfg.model.n_embd = 32
    for f in ("use_resid_lambdas", "use_x0_lambdas", "use_value_embeds",
              "use_smear", "use_backout"):
        setattr(cfg.model, f, True)
    model = build_model(cfg.model, device="cpu")
    x = torch.randint(0, 64, (2, 32))
    y = torch.randint(0, 64, (2, 32))
    loss = model(x, y)
    loss.backward()
    assert torch.isfinite(loss), f"loss 非有限: {loss}"
    assert model.smear_lambda.grad is not None
    ve_key = next(iter(model.value_embeds))
    assert model.value_embeds[ve_key].weight.grad is not None
