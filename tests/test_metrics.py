"""
`src/evaluation/metrics.py` 的正确性测试（卷7 只读代码的回归护栏）。

背景：这个模块此前**零测试覆盖**，导致两个缺陷长期潜伏：
  1. compute_pass_at_k 把 c 算成「前 k 个里的正确数」，公式用错
  2. evaluate_bpb / evaluate_multiple_choice 的 train 模式还原是死代码

所以本文件不测「要你敲的函数」，只测只读代码本身不能悄悄坏掉。

跑法：uv run pytest tests/test_metrics.py -v
"""

import math

import pytest
import torch

from evaluation.metrics import (
    compute_pass_at_k, evaluate_bpb,
)


# ===========================================================================
# pass@k —— 最容易写错的一处
# ===========================================================================
def _reference_pass_at_k(outcomes, k):
    """独立实现的正确答案：pass@k = 1 - C(n-c, k)/C(n, k)，c 为总正确数。"""
    n = len(outcomes)
    if n == 0:
        return 0.0
    c = sum(outcomes)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def test_pass_at_k_matches_reference():
    """对照独立实现，穷举小规模组合。

    这一条直接锁住 P0 那个 bug：写成 `c = sum(outcomes[:k])` 时，
    正确答案若不在前 k 位，c 被算成 0，pass@k 归零。
    """
    cases = [
        ([True, False, False, False], 1),
        ([False, False, False, True], 1),      # ← 正确样本在最后，旧代码返回 0
        ([False, False, True, True], 2),       # ← 同上
        ([True, True, False, False], 2),
        ([False, False, False, False], 1),
        ([True, True, True, False], 3),
        ([True] * 5, 2),
        ([False] * 5, 3),
        ([True, True, False, False, False, False], 3),
    ]
    for outcomes, k in cases:
        expect = _reference_pass_at_k(outcomes, k)
        got = compute_pass_at_k(outcomes, k)
        assert abs(got - expect) < 1e-12, (
            f"pass@k({outcomes}, k={k}) 应为 {expect:.6f}，实际 {got:.6f}"
        )


def test_pass_at_k_c_is_total_not_prefix():
    """c 必须是「总正确数」，与正确样本在列表里的位置无关。

    同一组正确样本数，只是换个位置，pass@k 不该变。
    """
    a = compute_pass_at_k([False, False, True], k=1)
    b = compute_pass_at_k([True, False, False], k=1)
    assert abs(a - b) < 1e-12, (
        f"pass@1 只取决于对了几个，与位置无关：得到 {a:.4f} vs {b:.4f}")
    assert abs(a - 1 / 3) < 1e-12, "1/3 正确时 pass@1 应为 1/3"


def test_pass_at_k_boundaries():
    """边界：空样本、k 超过样本数、全对、全错。"""
    assert compute_pass_at_k([], k=1) == 0.0, "空样本应返回 0"

    # 样本不够挑 k 个时，「至少一个对」的概率按定义取 1
    assert compute_pass_at_k([False, False], k=5) == 1.0
    assert compute_pass_at_k([False, True], k=5) == 1.0

    assert compute_pass_at_k([True, True], k=1) == 1.0, "全对应为 1"
    assert compute_pass_at_k([False, False], k=1) == 0.0, "全错应为 0"


def test_pass_at_k_monotone_in_k():
    """固定样本数，k 越大 pass@k 不减（能挑的样本多了，命中概率只会更高）。"""
    outcomes = [True, False, False, True, False, False, False, True]
    vals = [compute_pass_at_k(outcomes, k) for k in range(1, 6)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-12, f"pass@k 应随 k 单调不减：{vals}"


def test_pass_at_1_equals_correct_ratio():
    """k=1 时 pass@1 恒等于 c/n —— 这是 eval_sft 用 n 条估 pass@1 的依据。"""
    for n in (1, 2, 3, 4, 8):
        for c in range(n + 1):
            outcomes = [True] * c + [False] * (n - c)
            assert abs(compute_pass_at_k(outcomes, 1) - c / n) < 1e-12


def test_more_samples_lower_variance_of_pass_at_1_estimator():
    """用 n 个样本估 pass@1，比只看 1 个样本方差更低。

    这条锁住 eval_sft 那个「生成 4 条却只用 1 条」的无谓浪费：
    两者是同一个量，但后者抖动大一倍。

    ★ 与旧版的区别：旧版拿 random 自己模拟两个估计量算方差，
      **从头到尾没有 import 过 compute_pass_at_k** —— 也就是说
      metrics.py 整个坏掉它照样绿。这里改为驱动真实的实现：
      对同一个「每题独立、正确率 p」的题目模型，
      枚举所有可能的 outcome 组合，算 pass@1 估计量的真实方差。
    """
    import itertools
    p, trials = 0.3, 4000

    def variance_of_pass1_estimator(n):
        """用**真实的 compute_pass_at_k** 估计 n 样本 pass@1 的方差。

        对每个 outcome 向量，按二项分布的权重取期望与二阶矩。
        """
        acc = acc2 = 0.0
        for outcomes in itertools.product([False, True], repeat=n):
            weight = 1.0
            for ok in outcomes:
                weight *= p if ok else (1.0 - p)
            est = compute_pass_at_k(list(outcomes), k=1)
            acc += weight * est
            acc2 += weight * est * est
        mean = acc
        return acc2 - mean * mean

    # 先锁住「k=1 时 pass@k 化简成 c/n」这个恒等式 ——
    # 它是上面方差结论成立的前提。
    for outcomes in ([True], [False], [True, True, False, True],
                     [False, False, False, True], [True] * 4, [False] * 4):
        c = sum(outcomes)
        assert compute_pass_at_k(outcomes, k=1) == pytest.approx(c / len(outcomes))

    var1 = variance_of_pass1_estimator(1)
    var4 = variance_of_pass1_estimator(4)
    # 理论值：n=1 -> p(1-p)=0.21；n=4 -> 0.0525（正好 1/4，因为 0.21/4=0.0525）
    assert var1 == pytest.approx(0.21, abs=1e-9), f"n=1 的方差应等于 p(1-p)，实际 {var1:.4f}"
    assert var4 == pytest.approx(var1 / 4, rel=0.02), (
        f"n=4 的方差应约为 n=1 的 1/4，实际 {var4:.5f} vs {var1/4:.5f}")
    # 采样估计也要稳定（固定 seed，避免 flaky）
    import random
    random.seed(0)
    stds = {}
    for n in (1, 4):
        ests = []
        for _ in range(trials):
            c = sum(1 for _ in range(n) if random.random() < p)
            ests.append(compute_pass_at_k([True] * c + [False] * (n - c), k=1))
        m = sum(ests) / len(ests)
        stds[n] = math.sqrt(sum((e - m) ** 2 for e in ests) / len(ests))
    assert stds[4] < stds[1] * 0.75, (
        f"4 样本的标准差应明显低于 1 样本，实际 {stds[4]:.4f} vs {stds[1]:.4f}")


# ===========================================================================
# 共享的模型/数据夹具
# ===========================================================================
def _tiny_model():
    """建一个极小模型。

    main 分支上 `build_model` 还是骨架（卷2-3 的手抄目标），此时这些
    「只读代码的回归护栏」没有可测对象 —— skip 而不是 fail，
    否则会给「你还没敲的函数」再添一批噪音失败。
    """
    from common.config import make_run_config
    from model.gpt import build_model
    cfg = make_run_config("debug", vocab_size=64)
    cfg.model.sequence_len, cfg.model.n_embd = 32, 32
    cfg.model.n_head = cfg.model.n_kv_head = 2
    cfg.model.head_dim = 16
    try:
        model = build_model(cfg.model, device="cpu")
    except NotImplementedError:
        pytest.skip("模型还是骨架（main 分支），卷2-3 完成后再跑这些只读代码的护栏")
    return model, cfg


def test_evaluate_multiple_choice_counts_and_restores_mode():
    """多选题评测：分数最高的选项即预测；结束必须还原 train 模式。

    覆盖的是此前零测试的 evaluate_multiple_choice，
    同时守住 was_training 的取值顺序（先读状态再 model.eval()）。
    """
    import evaluation.metrics as M

    model, _ = _tiny_model()
    model.train()
    n = 3
    items = [{"question": f"q{i}", "choices": ["a", "b", "c", "d"],
              "gold": i % 4} for i in range(n)]

    calls = []

    def fake_score(model_, tok, question, choices, device):
        calls.append(question)
        return [0.0, 0.0, 0.0, 0.0]      # 分数全同 -> max() 取下标 0

    orig = M.score_choices
    M.score_choices = fake_score
    try:
        out = M.evaluate_multiple_choice(model, None, items,
                                         device="cpu", max_examples=n)
    finally:
        M.score_choices = orig

    assert len(calls) == n, f"应打分 {n} 次，实际 {len(calls)} 次"
    assert out["n"] == n
    # 分数全相同 -> 预测恒为选项 0 -> 只有 gold==0 的那题算对
    assert out["accuracy"] == pytest.approx(1 / n)
    assert model.training, "结束后必须还原成 train 模式"


def test_evaluate_multiple_choice_respects_max_examples():
    """max_examples 应真的截断题量。"""
    model, _ = _tiny_model()
    import evaluation.metrics as M
    items = [{"question": "q", "choices": ["a", "b"], "gold": 0}] * 10
    orig = M.score_choices
    M.score_choices = lambda *a, **k: [1.0, 0.0]
    try:
        out = M.evaluate_multiple_choice(model, None, items,
                                         device="cpu", max_examples=4)
    finally:
        M.score_choices = orig
    assert out["n"] == 4, f"max_examples 应生效，实际 n={out['n']}"
    assert out["accuracy"] == 1.0


def test_evaluate_bpb_restore_mode_both_directions():
    """从 train 进、从 eval 进，都要还原成进来时的状态。"""
    model, cfg = _tiny_model()
    tok_batch = torch.randint(0, cfg.vocab_size, (2, 32))
    tb = torch.full((cfg.vocab_size,), 4.0)
    loader = [(tok_batch, tok_batch.clone(), {})]

    for initial in (True, False):
        model.train(initial)
        evaluate_bpb(model, loader, tb, max_batches=1)
        assert model.training is initial, (
            f"进来时 training={initial}，出来时应还原，实际 {model.training}")


def test_evaluate_bpb_empty_loader_is_finite():
    """一个 batch 都没有时不能崩，也不能出 NaN。"""
    model, _ = _tiny_model()
    tb = torch.full((64,), 4.0)
    val = evaluate_bpb(model, [], tb, max_batches=4)
    assert not math.isnan(val), f"空 loader 不该得到 NaN：{val}"
    assert val == 0.0, f"空 loader 应得 0（0 总字节被 max(...,1e-9) 兜住），得到 {val}"


def test_evaluate_bpb_zero_byte_token_does_not_divide_by_zero():
    """token_bytes 全为 0 时不能除零。"""
    model, cfg = _tiny_model()
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    tb = torch.zeros(cfg.vocab_size)
    val = evaluate_bpb(model, [(x, x.clone(), {})], tb, max_batches=1)
    assert not math.isnan(val), f"总字节为 0 时不该 NaN：{val}"
