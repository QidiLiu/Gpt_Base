"""
正交化：Muon 的核心。

▓▓ 这一章要你敲的部分 ▓▓
    orthogonalize_simple     ★ 20 行，本教程最重要的 20 行（卷4 第 23 章）
    orthogonalize_advanced   分层给骨架（卷4 第 24-25 章）
    nor_muon_scale          卷4 第 25 章
    orthogonality_error      验证工具（自己写一个）

────────────────────────────────────────────────────────────────
为什么需要正交化？
────────────────────────────────────────────────────────────────
Muon 的思路是：SGD 的更新量 g 方向很差 —— 矩阵不同维度上的梯度尺度
差异巨大，直接用会让「高频方向」主导。

如果对 g 做 SVD：g = U·Σ·Vᵀ，最理想的做法是只保留方向 U·Vᵀ（丢掉 Σ），
这叫「完全正交化」。但 SVD 太慢。

Newton-Schulz 迭代是 SVD 的快速近似：反复做
    X ← a·X + (b·X·Xᵀ + c·(X·Xᵀ)²)·X
只用矩阵乘（可在 tensor core 上跑），几步就让 X·Xᵀ 逼近单位矩阵。

「正交化」这个名字来自这里：让更新量变成「各向同性」的，
每个方向的步长一样，条件数从 O(κ) 压到 O(1)。

────────────────────────────────────────────────────────────────
Newton-Schulz 为什么不迭代到收敛？
────────────────────────────────────────────────────────────────
经典的五次多项式 (3.4445, -4.7750, 2.0315) 在 |x|≤1 上是收敛的，
但代价是**在 0 附近斜率太小**，收敛慢。modded-nanogpt 的观察是：
斜率不够大不要紧，可以「不过度收敛」—— 让结果落在 US′Vᵀ，
其中 S′ 的对角线元素落在 [0.5, 1.5] 的均匀分布里。
这已经足够好，而且收敛快得多。

Polar Express（arXiv 2505.16932）把这个「选择合适的每步系数」
变成一个正经的优化问题：在保证不过度收敛的前提下，
最大化原点处的斜率。
"""

import torch

# ===========================================================================
# 🔶 规格部分：系数表（直接抄，理解来源即可）
# ===========================================================================
# 经典 Newton-Schulz 五次多项式 x·(a + b·x² + c·x⁴) 的系数。
# 优点是短、久经考验；缺点是 0 点斜率偏小，收敛慢。
NEWTON_SCHULZ_COEFFS = (3.4445, -4.7750, 2.0315)

# Polar Express 的系数（num_iters=5, safety_factor=2e-2, cushion=2）
# 来自 https://arxiv.org/pdf/2505.16932
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


# ===========================================================================
# ❗ 1. 最简正交化（卷4 第 23 章）★ 全教程最重要的 20 行
# ===========================================================================
def orthogonalize_simple(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    对 G 的最后两维做正交化。G 可以是 (..., m, n)，前面的维度是 batch。

    ── 要点 1：分「高矩阵 / 宽矩阵」两种情况 ──────────────────
    我们想做的是 X → X·(XᵀX)^{-1/2}（宽矩阵，m ≤ n）
    或     X → (XXᵀ)^{-1/2}·X（高矩阵，m > n）。

    它们的迭代形式不同：
      宽矩阵：算 A = X·Xᵀ （小，m×m）
      高矩阵：算 A = Xᵀ·X （小，n×n）
    统一做法：如果矩阵是高的，先转置，处理完再转回来。
    提示：.mT 是 PyTorch 转置的快捷方式（等价于 .transpose(-2,-1)）。

    ── 要点 2：归一化的位置很关键 ────────────────────────────
    每轮迭代前不归一化的话，X 的尺度会指数增长或衰减，
    在 bf16 下会溢出/下溢。所以迭代前先除掉整体 Frobenius 范数
    （带 1.01 的裕度），把 X 拉到单位球附近。

    ── 要点 3：迭代里的运算顺序 ──────────────────────────────
        A = X @ X.mT                    # (..., m, m)
        B = b * A + c * (A @ A)         # 三次多项式的中间量
        X = a * X + B @ X               # ⚠ 注意是 B @ X 不是 X @ B
    最后那个顺序很重要，写反了会算成另一个东西
    （宽矩阵情况下的标准写法就是 B @ X）。

    ── 你要写的 ──
    1) X = G
    2) transposed = X.size(-2) > X.size(-1)；if transposed: X = X.mT
    3) X = X / (X.norm(dim=(-2,-1), keepdim=True) * 1.01 + 1e-6)
    4) a, b, c = NEWTON_SCHULZ_COEFFS
       for _ in range(steps): 跑上面那三行
    5) if transposed: X = X.mT
    6) return X

    ── 怎么验证你写对了 ──────────────────────────────────────
    uv run pytest -k orthogonalize -v
      · test_orthogonalize_improves_orthogonality
        正交误差 ||X·Xᵀ - I||_F / ||X·Xᵀ||_F 必须显著变小
    uv run python scratch/ortho_demo.py   （自己写，见 tutorial 第 23 章）
    """
    raise NotImplementedError(
        "待实现：orthogonalize_simple —— 五步，见 docstring\n"
        "  参考实现：git show solution:src/optim/orthogonalize.py\n"
        "  验证：uv run pytest -k orthogonalize -v")


# ===========================================================================
# ❗ 2. 生产版正交化（卷4 第 24-25 章）
# ===========================================================================
def orthogonalize_advanced(G: torch.Tensor, steps: int = 5,
                           use_muon_eq: bool = True,
                           use_muon_plus: bool = True,
                           use_polar_express: bool = True) -> torch.Tensor:
    """
    相比 simple 版多了三件事，每件都修一个具体问题。
    三个开关可以逐个关掉做消融（`--muon-advanced` 之后加这些参数）。

    ── (1) MuonEq 行均衡（use_muon_eq）────────────────────
    问题：矩阵不同行的范数可能差几个数量级，直接送进正交化会让
          谱严重不均衡，迭代收敛变慢甚至发散。
    做法：把每一行 rescale 到「平均行范数」，让进入正交化时
          每一行量级一致。
    代价：多做一次 norm 和一次乘法。
    出处：MuonEq, arXiv 2603.28254

    ── (2) Polar Express（use_polar_express）──────────────
    问题：Newton-Schulz 的固定系数在 0 点斜率偏小。
    做法：换用针对「最大化 0 点斜率」优化出来的 5 组系数。
    出处：Polar Express, arXiv 2505.16932
    实测收益：smoke 档 d4 随机矩阵上，正交误差从 0.47（simple）降到 0.17。

    ── (3) Muon+ 重归一化（use_muon_plus）──────────────────
    问题：有限步迭代不会精确收敛到正交矩阵（奇异值停在 [0.5,1.5] 附近），
          而且如果谱里有极小奇异值，迭代**根本推不动**它
          （Newton-Schulz 在 0 附近斜率低 -> x=0 是不动点）。
    做法：算完之后强行把 Frobenius 范数 snap 到 √min(m,n)
          —— 那是一个「恰好正交」矩阵应有的范数。
          这能让「推不动的极小奇异值」对应的方向整体被放大回来。
    出处：Muon+, arXiv 2602.21545

    ★ Muon+ 的效果实测（这是本教程最能说明问题的一个数字）：
        对一个 rank=8 的低秩矩阵（128×192）做正交化，
            simple（5 步 Newton-Schulz）  正交误差 3.34
            Polar Express 单独            正交误差 3.73   （也救不回来）
            Polar Express + MuonEq        正交误差 3.68
            完整（含 Muon+）              正交误差 0.97   ← 唯一有效的
      原因：rank=8 的矩阵有 128-8=120 个零奇异值，而 0 是
      Newton-Schulz 迭代的**不动点**，迭代多少步都推不动它。
      跑 pytest -k muon_plus 会验证这一点。

    ── 你要写的 ──
    顺序：MuonEq 行均衡 → 归一化 → [转置] → 迭代 → [转置] → MuoN+ 重归一化

    MuonEq 的公式（target 那行要想清楚）：
        整张矩阵的 RMS 行范数 = 1，也就是
        target = ||X||_F / sqrt(m)           （m = 行数）
        row_norm = 每行的 L2 范数
        X = X * (target / row_norm)           （逐行缩放）
    提示：行范数要 clamp_min(1e-6) 防 0 除。

    Muon+ 的公式：
        target_norm = min(m, n) ** 0.5      （半正交矩阵的 Frobenius 范数）
        current_norm = ||X||_F
        X = X * (target_norm / current_norm)

    迭代部分的分支（高矩阵 vs 宽矩阵）：
        高矩阵（transposed=True）:  A = X.mT @ X;  B = b*A + c*(A@A);  X = a*X + X @ B
        宽矩阵                  :  A = X @ X.mT;  B = b*A + c*(A@A);  X = a*X + B @ X
    ⚠ 两个分支的**乘法顺序不同**，别写混。

    验证：uv run pytest -k "orthogonalize or muon_plus" -v
    """
    raise NotImplementedError(
        "待实现：orthogonalize_advanced —— 四个阶段，见 docstring\n"
        "  参考实现：git show solution:src/optim/orthogonalize.py")


# ===========================================================================
# ❗ 3. NorMuon 方差缩减（卷4 第 25 章）
# ===========================================================================
def nor_muon_scale(G: torch.Tensor, state: torch.Tensor, beta2: float,
                   red_dim: int) -> torch.Tensor:
    """
    NorMuon（arXiv 2510.05491）：正交化之后还要做一次「逐神经元自适应缩放」。

    动机：Muon 的输出虽然整体正交，但**不同神经元的更新尺度仍然不同**。
          某些神经元会收到远大于其他神经元的更新量，导致训练不稳定。

    做法：估计每个「列/行」上的更新能量 v，用 1/√v 归一化，
          并且把归一化后的整体尺度还原回去（否则范数会变）。

    ── 你要想清楚的三点 ──────────────────────────────────────

    (1) state 存的是什么形状？
        提示：只存 (chunk, m, 1) 或 (chunk, 1, n) —— 被规约维度的均值，
        不存完整的 (chunk, m, n)。省 100~1000 倍显存。
        这就是 NorMuon 名字里 "Nor"（Normalized）的来源。
        规约维度由调用方给出（red_dim = -1 或 -2，取决于矩阵是高是宽）。

    (2) 为什么最后要把范数「还原」回去？
        提示：1/√v 归一化会把整体范数也缩小了。
        要用归一化前的全局范数除以归一化后的全局范数来补偿。
        不还原的话，Muon 的有效步长会随训练漂移。

    ── 提示（这不是答案，是路标）──
        v_mean    = G.float().square().mean(dim=red_dim, keepdim=True)
        v_norm    = 归一化前的全局尺度（见上面的 (2)）
        state     .lerp_(v_mean, 1 - beta2)      # EMA
        step_size = state.clamp_min(1e-10).rsqrt()
        v_norm_new= 归一化后的全局尺度
        final     = step_size * (v_norm / v_norm_new)
    """
    raise NotImplementedError(
        "待实现：nor_muon_scale —— 见 docstring 的三点 + 提示\n"
        "  验证：uv run pytest -k muon -v\n"
        "  参考实现：git show solution:src/optim/orthogonalize.py")


# ===========================================================================
# ❗ 4. 自检工具（卷4 第 23 章的验证实验要用）
# ===========================================================================
def orthogonality_error(X: torch.Tensor) -> float:
    """
    衡量一个矩阵「有多正交」。

    理想情况 X·Xᵀ = I，所以 ||X·Xᵀ - I||_F 应该接近 0。
    返回一个 0 附近的数，越小越正交。

    提示：
      A = X.float() @ X.float().mT              （转 fp32 因为要算范数）
      eye = 单位矩阵，尺寸 = X.size(-2)
      return (A - eye).norm() / (A.norm() + 1e-9)
    """
    raise NotImplementedError(
        "待实现：orthogonality_error —— 三行，见 docstring")


def condition_number(X: torch.Tensor) -> float:
    """
    条件数 = 最大奇异值 / 最小奇异值。正交矩阵的条件数是 1。
    用来观察 MuonEq 和 Muon+ 各自改善了什么。

    提示：torch.linalg.svdvals(X.float()) 给出奇异值（降序）。
    """
    raise NotImplementedError(
        "待实现：condition_number —— \n"
        "  s = torch.linalg.svdvals(X.float())\n"
        "  return (s.max() / s.min().clamp_min(1e-12)).item()")
