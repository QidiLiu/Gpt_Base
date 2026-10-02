"""
正确性测试。

重点覆盖**形状/布局类**的 bug —— 这类 bug 最危险：
数值看起来「差不多对」，但训练几百步后才发现不对劲。
每个测试都对应一个真实踩过的坑。

跑法：uv run pytest tests/ -v
"""

import pytest
import torch

from common.config import make_run_config, build_model_config
from model.layers import (
    precompute_rope, apply_rotary_emb, rms_norm, attend, sdpa_bt, MLP,
)
from model.gpt import build_model, sample_from_logits
from optim.orthogonalize import (
    orthogonalize_simple, orthogonality_error,
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


# ===========================================================================
# Flash Attention 2 的保证
# ===========================================================================
def test_causal_path_can_select_flash_backend():
    """
    全上下文路径**必须能用** Flash Attention 2，且 attn_impl 不影响它。

    ── 为什么这条是判据而不只是注释？───────────────────────────────
    `F.scaled_dot_product_attention` 会在所有可用后端里**静默**挑一个。
    实测本项目全上下文路径一直都在用 FA2 —— 但从来没有任何东西验证过，
    只是碰巧如此。哪天某个 shape / dtype / torch 版本让它挑不满足，
    就会悄悄退回 mem-efficient 或 math，而且**不报错**。
    教程卷3 第20章的「性能悬崖」就是这么来的。

    这里用 `can_use_flash_attention(SDPAParams(...))` 直接问 PyTorch：
    给定真实形状，flash 后端到底可不可用。这是「保证」能被测量的形式。
    """
    if not torch.cuda.is_available():
        pytest.skip("没有 CUDA，无法探测 attention 后端")
    try:
        from torch.backends.cuda import SDPAParams, can_use_flash_attention
    except ImportError:
        pytest.skip("这个 torch 版本没有 torch.backends.cuda.can_use_flash_attention")

    # 覆盖三档真实形状 + GQA 场景
    cases = [
        # (n_head, head_dim, T, n_kv_head)  —— debug/smoke/ablation/full
        (1, 32, 128, 1), (4, 32, 512, 4), (6, 64, 1024, 6), (12, 64, 1024, 12),
        (12, 64, 1024, 4),   # full 档开 GQA 时
    ]
    for n_head, head_dim, T, n_kv_head in cases:
        q = torch.randn(1, n_head, T, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, n_kv_head, T, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        params = SDPAParams(q, k, v, None, 0.0, True, n_head != n_kv_head)
        assert can_use_flash_attention(params, True), (
            f"形状 (H={n_head}, kv={n_kv_head}, T={T}, D={head_dim}) 下 "
            f"flash 后端不可用 —— 全上下文路径会静默退回 mem-efficient。"
            f"本项目的 FA2 保证就失效了。"
        )


def test_sliding_window_paths_are_numerically_equivalent():
    """
    flex_attention（FA2 风格）与显式 mask 两条滑窗路径必须数值一致。

    ── 为什么这是必须的？────────────────────────────────────────
    `attn_impl` 是个消融开关，两条路径都在真实训练里跑。它们的数值必须
    相同，否则「同一模型换个开关跑出不同的 bpb」就无法区分是
    「开关的架构差异」还是「两条 kernel 算错了」—— 消融结论就废了。

    实测两条路的差异在 bf16 舍入量级（相对误差 < 1e-3），因为
    flex 走的是块级累加、SDPA 走的是逐元素累加，累加顺序不同。
    所以判据用**相对容差**而不是 allclose 的默认绝对容差。
    """
    if not torch.cuda.is_available():
        pytest.skip("flex_attention 需要 CUDA 才能编译")
    from model.layers import attend_sliding_flex, attend_sliding_sdpa, _flex_attention
    if _flex_attention() is None:
        pytest.skip("flex_attention 编译不可用（缺 C 编译器？），只剩显式 mask 路径")

    torch.manual_seed(0)
    for B, H, T, D, left in [(2, 4, 256, 32, 64), (1, 2, 128, 64, 32),
                             (3, 6, 512, 64, 128)]:
        q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
        got = attend_sliding_flex(q, k, v, left).float()
        ref = attend_sliding_sdpa(q, k, v, left).float()
        scale = ref.abs().max().clamp_min(1e-6)
        rel = (got - ref).abs().max() / scale
        assert rel < 1e-2, (
            f"B{B} H{H} T{T} D{D} left={left}: 两条滑窗路径相对误差 {rel:.5f} "
            f"过大（阈值 1e-2）。bf16 舍入应在 1e-3 量级。"
        )


def test_attn_impl_does_not_change_full_context_result():
    """
    全上下文（left >= T）的结果与 attn_impl 无关 —— 它永远走 SDPA flash。

    这条钉住一个设计决定：flex_attention 在全上下文上没有收益
    （is_causal=True 本来就选中 flash），所以不应该把它也换掉。
    若哪天真换成「全上下文也走 flex」，这个测试会提醒你两者的差异。
    """
    torch.manual_seed(0)
    B, T, H, D = 2, 64, 4, 16
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    a = attend(q, k, v, (T, 0), attn_impl="flex")
    b = attend(q, k, v, (T, 0), attn_impl="sdpa")
    assert torch.equal(a, b), "attn_impl 影响了全上下文路径的结果（它不该影响）"


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


# ===========================================================================
# 逐章的粒度判据（卷2 第08/13/16 章、卷3 第17/18/19 章）
# ===========================================================================
# 为什么要这些：progress.sh 之前让第 08/13/16 章共用 `-k uniform`、
# 第 17/18/19 章共用 `-k all_tricks`，学习者无法知道**具体哪一章**做完了。
# 下面每个测试只锁一个组件，-k 选择器可以一一对应到章节。

def _tiny_all_tricks_model(only=None):
    """建一个开启了全部 5 个残差流 trick 的极小模型（device='cpu'）。"""
    cfg = make_run_config("debug", vocab_size=64)
    cfg.model.sequence_len = 32
    cfg.model.n_head = cfg.model.n_kv_head = 2
    cfg.model.head_dim = 16
    cfg.model.n_embd = 32
    for f in ("use_resid_lambdas", "use_x0_lambdas", "use_value_embeds",
              "use_smear", "use_backout"):
        setattr(cfg.model, f, True)
    if only is not None:
        for f in ("use_resid_lambdas", "use_x0_lambdas", "use_value_embeds",
                  "use_smear", "use_backout"):
            if f != only:
                setattr(cfg.model, f, False)
    model = build_model(cfg.model, device="cpu")
    # ★ 打破「block 初始是恒等映射」：init_weights 把两个 c_proj 都置零，
    #   所以刚建好的模型里每个 block 都是恒等映射，残差流 trick 的影响
    #   会被完整地「吃掉」——实测零差异。必须先给 c_proj 一点随机值。
    with torch.no_grad():
        for n, prm in model.named_parameters():
            if "c_proj" in n:
                prm.normal_(0.0, 0.05)
    return model


def test_rmsnorm_preserves_direction_per_row():
    """
    ch08 RMSNorm 的粒度判据。

    三个必须成立的性质：
      1) **逐行**方向不变（归一化只改长度，不改该行的方向）
      2) 输出长度固定（每行 RMS = 1），与输入长度无关
      3) 不减均值 —— 这正是它比 LayerNorm 便宜的地方
    """
    torch.manual_seed(0)
    x = torch.randn(3, 5, 64)
    y = rms_norm(x)
    # 1) 逐行方向不变：RMSNorm 对最后两维整体乘一个标量
    cos_row = torch.nn.functional.cosine_similarity(
        x[..., 0, :], y[..., 0, :], dim=-1)
    assert torch.allclose(cos_row, torch.ones_like(cos_row), atol=1e-4), \
        f"RMSNorm 改变了该行的方向，余弦相似度 {cos_row}"
    # 2) 输出 RMS 恒为 1
    got = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(got, torch.ones_like(got), atol=1e-3), \
        f"输出 RMS 应为 1，实际 {got}"
    # 3) 不减均值：对零均值的输入，输出也该零均值
    z = x - x.mean(-1, keepdim=True)
    zy = rms_norm(z)
    assert abs(float(zy.mean())) < 1e-5, "对零均值输入做了中心化 -> 就不该叫 RMSNorm"


def test_mlp_activation_relu2_vs_gelu():
    """
    ch13 MLP 的粒度判据：**同一份权重**下，relu2 与 gelu 必须给出不同结果。

    ReLU²(x) = max(0,x)²：x>0 时导数就是 2x，极其便宜。
    GELU(x) = x·Φ(x)：对负数不是恒 0。

    ★ 与旧版的区别：旧版对每种激活各建一个 MLP，然后比较
      `MLP(cfg).c_fc(x)` 的输出 —— 那有两个致命问题：
        1) 两次的 MLP 是**不同随机初始化**，输出当然不同，跟激活无关；
        2) c_fc 只是投影，**根本没施加激活**。
      结果：把 MLP.forward 改成「无条件 relu2」，测试照样通过。
      现在只建一个 MLP，只改 self.activation，权重完全不变。
    """
    from common.config import build_model_config
    torch.manual_seed(0)
    cfg = build_model_config(2, 16, 16, sequence_len=32, vocab_size=64)
    mlp = MLP(cfg)
    x = torch.randn(2, 8, cfg.n_embd) * 3.0        # 放大到有明显的负数

    # 同一份权重，只切 activation
    mlp.activation = "relu2"
    out_r2 = mlp(x).clone()
    mlp.activation = "gelu"
    out_gl = mlp(x).clone()
    assert not torch.allclose(out_r2, out_gl), (
        "同一份权重下 relu2 与 gelu 输出完全一样 -> forward 里的 "
        "activation 分支没生效")

    # 未知激活必须报错（else 分支不能省）
    mlp.activation = "swiglu"
    try:
        mlp(x)
        raise AssertionError("未知 activation 应该报错")
    except ValueError:
        pass

    # ── 直接验激活函数的语义（这里才是 relu2 与 gelu 的分水岭）──
    # 构造一个「c_fc 输出恒为负」的场景：输入全 -1，c_fc 权重全为正
    # => c_fc(x) = -sum(w) < 0。此时 relu2 应输出 0，gelu 应保留负值。
    # 注意 use_bias=False 时 c_fc.bias 是 None，所以只用权重构造。
    neg_in = torch.full((1, 4, cfg.n_embd), -1.0)
    with torch.no_grad():
        mlp.c_fc.weight.fill_(1.0 / cfg.n_embd)      # c_fc(-1) = -1
        mlp.c_proj.weight.fill_(1.0)                  # c_proj(v) = sum(v)
        mlp.activation = "relu2"
        z_r2 = mlp(neg_in).clone()
        mlp.activation = "gelu"
        z_gl = mlp(neg_in).clone()
    assert torch.allclose(z_r2, torch.zeros_like(z_r2)), \
        f"relu2 对全负输入应输出 0，实际 {z_r2}"
    assert (z_gl < 0).all(), f"gelu 对负输入应保留负值，实际 {z_gl}"


def test_meta_device_weights_are_all_finite():
    """
    ch16 meta device 三步法的粒度判据。

    ★ 本项目最重要的一个 NaN 坑：__init__ 里算的 cos/sin 是在 meta device
      上创建的（只有形状壳）。to_empty() 只分配**未初始化的内存**，
      不填任何值。漏掉「在 init_weights 里重算 RoPE」这一步，整个模型就废了。

    ⚠ 这里有个比「NaN」更阴险的坑，写这个判据时才发现：
      to_empty() 留下的是什么，**取决于是哪块内存**，实测有三种：

        | 场景                       | max|cos| | isfinite |
        |----------------------------|----------|----------|
        | 全新进程的 CPU 分配         | NaN      | False    |
        | 内存 churn 后的 CPU 回收块   | 4.4e+35  | **True** |
        | CUDA 全新页                 | 0.0      | **True** |

      只有第一种能被 `isfinite` 抓到。另两种是**有限垃圾** ——
      「NaN 检查」根本不会红，但 RoPE 依然是错的。

      所以判据不能只看 isfinite，必须验「内容对不对」：
        · cos 是余弦，取值必然落在 [-1, 1]
        · 位置 0 处 cos 恒为 1（全角度）
      这两条对真正的 NaN、有限垃圾、全零，三种情况一视同仁。
    """
    cfg = make_run_config("debug", vocab_size=64)
    cfg.model.sequence_len, cfg.model.n_embd = 32, 32
    cfg.model.n_head = cfg.model.n_kv_head = 2
    cfg.model.head_dim = 16
    model = build_model(cfg.model, device="cpu")

    # (1) 有限值（挡第一种情况：全新 CPU 分配的 NaN）
    assert torch.isfinite(model.cos).all(), "cos 表含 NaN/Inf -> 没在 init_weights 重算"
    assert torch.isfinite(model.sin).all(), "sin 表含 NaN/Inf -> 没在 init_weights 重算"

    # (2) 内容正确 —— 这两条才真正抓得住「有限垃圾」和「全零」
    #     位置 0 的旋转角恒为 0，所以 cos(0)=1、sin(0)=0，一个字不差。
    #     注意索引：cos 形状是 (1, rotary_seq_len, 1, head_dim/2)，
    #     所以「位置 0」是 cos[0, 0]（第 0 维是 broadcast 的 batch/head）。
    assert model.cos[0, 0].eq(1.0).all(), (
        f"cos(位置0) 应恒为 1，实测 min={model.cos[0, 0].min():.4g} "
        f"max={model.cos[0, 0].max():.4g} -> 表是垃圾内存")
    assert model.sin[0, 0].eq(0.0).all(), (
        f"sin(位置0) 应恒为 0，实测 min={model.sin[0, 0].min():.4g} "
        f"max={model.sin[0, 0].max():.4g} -> 表是垃圾内存")

    # (3) 取值范围：余弦不可能越界。这一条挡「有限但离谱」的垃圾
    #     （实测见过 4.4e+35），(2) 抓不到的那种。
    for name, tbl in (("cos", model.cos), ("sin", model.sin)):
        assert tbl.abs().max() <= 1.0, (
            f"{name} 是三角函数值，不可能 >1，实测 max={tbl.abs().max():.4g}")

    # (4) 表必须覆盖 rotary_seq_len 且不是全零（全零表会被 (2) 抓到，
    #     这里只确认长度对得上）
    assert model.cos.shape[1] == model.rotary_seq_len, \
        f"RoPE 表长度 {model.cos.shape[1]} != rotary_seq_len {model.rotary_seq_len}"

    # 所有参数也必须有限
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert not bad, f"这些参数含 NaN/Inf: {bad}"
    # 参数必须在真实设备上（不是 meta）
    assert model.transformer.wte.weight.device.type == "cpu", "参数还留在 meta device"
    # 初始 loss ≈ ln(vocab)
    x = torch.randint(0, 64, (2, 32))
    with torch.no_grad():
        loss = model(x, x)
    ref = torch.log(torch.tensor(64.0))
    assert abs(float(loss) - float(ref)) < 0.1, \
        f"初始 loss {float(loss):.4f} 应 ≈ ln(64)={float(ref):.4f}"


def test_resid_lambdas_scale_the_residual_stream():
    """
    ch17 resid_lambdas / x0_lambdas 的粒度判据。

    resid_lambdas[i]：第 i 层入口把残差流整体乘一个可学标量。
    ★ 关键：它必须真的影响前向结果。把它们手动置 0，
      同一批输入的输出必须变化 —— 否则 forward 里根本没消费它们。
    """
    torch.manual_seed(0)
    model = _tiny_all_tricks_model(only="use_resid_lambdas")
    x = torch.randint(0, 64, (2, 32))
    with torch.no_grad():
        base = model(x).clone()
        model.resid_lambdas.zero_()            # 每层入口把残差流清零
        off = model(x).clone()
    assert not torch.allclose(base, off, atol=1e-6), (
        "resid_lambdas 置 0 后输出没变 -> forward 根本没消费它")
    # 它们是可学参数，必须能收到梯度
    model(x, x).backward()
    assert model.resid_lambdas.grad is not None, "resid_lambdas 没有梯度"
    assert torch.isfinite(model.resid_lambdas.grad).all()


def test_x0_lambdas_add_back_the_initial_embedding():
    """
    ch17 x0_lambdas 的粒度判据：把初始嵌入加回来必须改变输出。
    """
    torch.manual_seed(0)
    model = _tiny_all_tricks_model(only="use_x0_lambdas")
    x = torch.randint(0, 64, (2, 32))
    with torch.no_grad():
        base = model(x).clone()
        model.x0_lambdas.fill_(1.0)           # 明显加回初始嵌入
        on = model(x).clone()
    assert not torch.allclose(base, on, atol=1e-6), (
        "x0_lambdas 变化后输出没变 -> forward 根本没消费它")
    model(x, x).backward()
    assert model.x0_lambdas.grad is not None, "x0_lambdas 没有梯度"
    assert torch.isfinite(model.x0_lambdas.grad).all()


def test_backout_subtracts_mid_layer_residual():
    """
    ch19 Backout 的粒度判据：末层前减去中层残差。

    backout_lambda 默认 0.2。关掉它（置 0）必须改变输出。
    """
    torch.manual_seed(0)
    model = _tiny_all_tricks_model(only="use_backout")
    x = torch.randint(0, 64, (2, 32))
    with torch.no_grad():
        base = model(x).clone()
        model.backout_lambda.zero_()
        off = model(x).clone()
    assert not torch.allclose(base, off, atol=1e-6), (
        "backout_lambda 置 0 后输出没变 -> forward 根本没消费它")
    model(x, x).backward()
    assert model.backout_lambda.grad is not None, "backout_lambda 没有梯度"
    assert torch.isfinite(model.backout_lambda.grad).all()


def test_value_embeds_live_on_alternate_layers():
    """
    ch18 Value Embeddings 的粒度判据。

    ★ 位置约定：只加在「隔一层」和「最后一层」上（ResFormer 的折中）。
      残差流的容量有限，每加一路信号就多一份要维护的信息。
      has_value_embed(i, n) 的规则是 i % 2 == (n-1) % 2。
    """
    from model.layers import has_value_embed
    for n in (2, 4, 6, 8):
        wanted = [i for i in range(n) if has_value_embed(i, n)]
        assert n - 1 in wanted, f"n={n}：最后一层必须有 value embed"
        assert len(wanted) == (n + 1) // 2, \
            f"n={n}：应该正好一半，实际 {wanted}"

    # 真实模型里的层数必须与规则一致
    model = _tiny_all_tricks_model(only="use_value_embeds")
    n = model.config.n_layer
    got = sorted(int(k) for k in model.value_embeds.keys())
    assert got == [i for i in range(n) if has_value_embed(i, n)], \
        f"value_embeds 挂在 {got}，规则要求 {[i for i in range(n) if has_value_embed(i, n)]}"

    # 每一层都必须真的消费 ve —— 直接比对「传 ve」与「不传 ve」的
    # 单层 attention 输出。（不用端到端：init 时 c_proj 全零，
    #   每个 block 都是恒等映射，trick 的影响会被完全吃掉。）
    T = 6
    x = torch.randn(1, T, model.config.n_embd)
    cos, sin = model.cos[:, :T], model.sin[:, :T]
    block = model.transformer.h[n - 1]          # 必然带 value_embed 的那一层
    ve = model.value_embeds[str(n - 1)](torch.randint(0, 64, (1, T)))
    with torch.no_grad():
        # c_proj 初始是全零（每个 block 一开始是恒等映射），
        # 那样 attn 的输出恒为 0，ve 有没有被消费都看不出来。先给它一点值。
        block.attn.c_proj.weight.normal_(0.0, 0.05)
        with_ve = block.attn(x, (cos, sin), model.window_sizes[n - 1], None, ve)
        without = block.attn(x, (cos, sin), model.window_sizes[n - 1], None, None)
    assert (with_ve - without).abs().max() > 1e-7, \
        "传了 ve 但输出完全一样 -> attention 根本没消费 ve（ve_gate 没接上）"

    # 且必须能收到梯度
    model(torch.randint(0, 64, (1, 32)), torch.randint(0, 64, (1, 32))).backward()
    emb = model.value_embeds[str(n - 1)]
    assert emb.weight.grad is not None, "value_embeds 权重没有梯度"


def test_smear_mixes_previous_embedding():
    """
    ch19 Smear 的粒度判据：把前一个 token 的嵌入混进当前 token。

        x[i] = x[i] + lambda * sigmoid(gate) * x[i-1]

    ★ 直接单测 _apply_smear，而不是端到端。
      端到端测不出来：残差流本来就通过 attention 携带 token 身份，
      改位置 0 会让**所有**位置的输出都变，smear 的贡献被淹没。

    三条必须成立：
      1) lambda = 0（初始值）时必须是恒等映射
      2) 位置 0 没有前驱，必须原样保留
      3) 位置 t 的增量必须由 x[t-1] 决定
    """
    torch.manual_seed(0)
    model = _tiny_all_tricks_model(only="use_smear")
    B, T, D = 2, 8, model.config.n_embd
    x = torch.randn(B, T, D)

    with torch.no_grad():
        # 1) 初始 lambda 就是 0（刻意关闭），此时必须是恒等映射
        assert float(model.smear_lambda.detach()) == 0.0, "smear_lambda 初始应为 0"
        assert torch.allclose(model._apply_smear(x.clone(), None), x, atol=1e-7), \
            "lambda=0 时 _apply_smear 应是恒等映射"

        model.smear_lambda.fill_(1.0)
        on = model._apply_smear(x.clone(), None)
        # 2) 位置 0 原样保留（没有前驱）
        assert torch.allclose(on[:, 0], x[:, 0], atol=1e-7), \
            "位置 0 没有前驱，不该被混合"
        # 3) 位置 1..T-1 必须被改变
        delta = (on - x)
        assert delta[:, 1:].abs().max() > 1e-7, "位置 1..T-1 没有任何变化 -> smear 没生效"
        assert delta[:, 0].abs().max() == 0.0, "位置 0 不该有增量"

    # 4) 增量必须由**前一个位置**驱动：改 x[0] 只能影响位置 1 的 smear 项
    x2 = x.clone()
    x2[:, 0] += 5.0
    with torch.no_grad():
        on2 = model._apply_smear(x2, None)
    assert not torch.allclose(on[:, 1], on2[:, 1], atol=1e-7), \
        "改 x[0] 没有改变位置 1 的输出 -> smear 项不是来自 x[0]"

    # 5) 是可学参数，必须有梯度
    model(torch.randint(0, 64, (1, 32)), torch.randint(0, 64, (1, 32))).backward()
    assert model.smear_lambda.grad is not None, "smear_lambda 没有梯度"
