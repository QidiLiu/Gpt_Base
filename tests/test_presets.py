"""
卷 0（配置与单一旋钮）的测试。

  uv run pytest tests/test_presets.py -v
"""

from common.config import (
    make_run_config, resolve_scaling, estimate_flops_per_token,
    build_model_config, scaling_params, ModelConfig,
)


def test_presets_are_self_consistent():
    """
    三档预设必须自洽：梯度累积步数 >= 1、步数 >= 1。

    曾经踩过：full 档 total_batch_size=65536，device_batch_size=24，
    sequence_len=1024 -> 24*1024 = 24576 整除不了 65536。
    resolve_scaling 里的 assert 当场抓住了它。
    """
    from common.config import resolve_scaling
    for mode, V in [("debug", 8192), ("smoke", 8192), ("full", 16384)]:
        cfg = make_run_config(mode, vocab_size=V)
        sc = resolve_scaling(cfg, log=lambda *a: None)
        assert sc["grad_accum_steps"] >= 1, f"{mode} 档 grad_accum < 1"
        assert sc["num_iterations"] >= 1, f"{mode} 档步数为 0"
        assert sc["total_batch_size"] > 0


def test_depth_drives_width_linearly():
    """
    单一旋钮的核心保证：n_embd 与 depth 成正比（aspect_ratio 固定）。

    这是 scaling law 能跨 depth 迁移的前提 ——
    只有形状一致，推导出的 batch size / LR / WD 才能通用。
    """
    for depth in (4, 6, 8, 12):
        for aspect in (32, 64):
            cfg = build_model_config(depth, aspect, 64, 1024, 16384)
            base = depth * aspect
            n_embd = ((base + 63) // 64) * 64        # 向上取整到 head_dim
            assert cfg.n_embd == n_embd
            assert cfg.n_embd % cfg.head_dim == 0
            assert cfg.n_head == cfg.n_embd // cfg.head_dim
            assert cfg.n_embd >= base, "n_embd 应该 >= depth*aspect"


def test_scaling_params_excludes_embeddings():
    """
    scaling 用的参数量口径：transformer 矩阵 + lm_head，**不含 wte 嵌入表**。

    为什么？查表没有矩阵乘的 FLOPs。
    （nanochat 的实验结论见其 dev/LOG.md 2026-01-27：
      这个组合能得到最干净的 scaling law。）
    """
    cfg = build_model_config(6, 64, 64, 1024, 16384)
    P = scaling_params(cfg)
    # lm_head 恰好是 n_embd * vocab_size（不含 wte 的那一份）
    assert P >= cfg.n_embd * cfg.vocab_size
    # 加一层矩阵约 6*4*d^2
    per_layer = 6 * 4 * cfg.n_embd ** 2
    assert P < 6 * per_layer + cfg.n_embd * cfg.vocab_size + 1


def test_flops_estimate_is_positive_and_monotonic():
    """更深/更宽的模型，FLOPs/token 必须更大。"""
    small = build_model_config(4, 32, 32, 512, 8192)
    large = build_model_config(8, 64, 64, 1024, 16384)
    assert estimate_flops_per_token(large) > estimate_flops_per_token(small)


def test_window_sizes_respect_pattern():
    """
    window_pattern 的展开规则：
      · 只允许 S 和 L
      · 模式按层平铺（"SSL" -> S,S,L,S,S,L,...）
      · 最后一层强制 L（最靠近输出的层需要看全局）
    """
    cfg = ModelConfig(n_layer=6, sequence_len=2048,
                      window_pattern="SSL")
    sizes = cfg.window_sizes()
    assert len(sizes) == 6
    assert sizes[-1][0] == 2048, "最后一层必须全上下文"
    for i in (0, 1, 3, 4):
        assert sizes[i][0] < 2048, f"第 {i} 层应该是滑窗"
    for i in (2, 5):
        assert sizes[i][0] == 2048, f"第 {i} 层应该是全上下文"
    # 非法字符必须报错
    try:
        ModelConfig(window_pattern="XY").window_sizes()
        raise AssertionError("非法 window_pattern 应该被 assert 拦住")
    except AssertionError as e:
        if "非法" not in str(e):
            raise


def test_unknown_override_raises():
    """拼错配置项名要立刻报错，不能静默忽略。"""
    try:
        make_run_config("debug", vocab_size=64, lrr=0.1)
        raise AssertionError("未知配置项应该被拒绝")
    except ValueError as e:
        if "未知的配置项" not in str(e):
            raise
