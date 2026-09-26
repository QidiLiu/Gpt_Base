"""
正交化：Muon 的核心。

教程卷4 的核心章节会逐行拆这里。两个版本：

  orthogonalize_simple    最简版。5 步 Newton-Schulz 迭代，约 20 行。
  orthogonalize_advanced  nanochat 生产版。Polar Express + MuonEq + Muon+。

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
最大化原点处的斜率。它给出的 5 组系数每步都能用满预算。
"""

import torch

# ---------------------------------------------------------------------------
# 系数表
# ---------------------------------------------------------------------------
# 经典 Newton-Schulz 五次多项式 x·(a + b·x² + c·x⁴) 的系数。
# 优点是短、久经考验；缺点是 0 点斜率偏小，收敛慢。
NEWTON_SCHULZ_COEFFS = (3.4445, -4.7750, 2.0315)

# Polar Express 的系数（num_iters=5, safety_factor=2e-2, cushion=2）
# 来自 https://arxiv.org/pdf/2505.16932
# 每一项是一次迭代的三次多项式 x·(a + b·x² + c·x⁴)，系数逐步变化。
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


# ===========================================================================
# 版本一：最简正交化（教学默认）
# ===========================================================================
def orthogonalize_simple(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    对 G 的最后两维做正交化。G 可以是 (..., m, n)，前面的维度是 batch。

    ── 为什么要分「高矩阵 / 宽矩阵」两种情况？─────────────────────
    我们想做的是 X → X·(XᵀX)^{-1/2}（宽矩阵，m ≤ n）
    或     X → (XXᵀ)^{-1/2}·X（高矩阵，m > n）。

    它们的迭代形式不同：
      宽矩阵：算 A = X·Xᵀ （小，m×m）
      高矩阵：算 A = Xᵀ·X （小，n×n）
    统一做法：如果矩阵是高的，先转置，处理完再转回来。

    ── 归一化的位置很关键 ────────────────────────────────────────
    每轮迭代前不归一化的话，X 的尺度会指数增长或衰减，
    在 bf16 下会溢出/下溢。所以迭代前先除掉整体 Frobenius 范数
    （带 1.01 的裕度），把 X 拉到单位球附近。
    """
    X = G
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    # 归一化到单位球（1.01 留一点裕度）
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)

    a, b, c = NEWTON_SCHULZ_COEFFS
    for _ in range(steps):
        A = X @ X.mT                    # (..., m, m)
        B = b * A + c * (A @ A)         # 三次多项式的中间量
        X = a * X + B @ X               # 注意顺序：B @ X，不是 X @ B

    if transposed:
        X = X.mT
    return X


# ===========================================================================
# 版本二：nanochat 生产版
# ===========================================================================
def orthogonalize_advanced(G: torch.Tensor, steps: int = 5,
                           use_muon_eq: bool = True,
                           use_muon_plus: bool = True,
                           use_polar_express: bool = True) -> torch.Tensor:
    """
    相比 simple 版多了三件事，每件都修一个具体问题：

    1) MuonEq 行均衡（use_muon_eq）
       问题：矩阵不同行的范数可能差几个数量级，直接送进正交化会让
             谱严重不均衡，迭代收敛变慢甚至发散。
       做法：把每一行 rescale 到「平均行范数」，让进入正交化时
             每一行量级一致。
       代价：多做一次 norm 和一次乘法。
       出处：MuonEq, arXiv 2603.28254

    2) Polar Express（use_polar_express）
       问题：Newton-Schulz 的固定系数在 0 点斜率偏小。
       做法：换用针对「最大化 0 点斜率」优化出来的 5 组系数。
       出处：Polar Express, arXiv 2505.16932

    3) Muon+ 重归一化（use_muon_plus）
       问题：有限步迭代不会精确收敛到正交矩阵（奇异值停在 [0.5,1.5] 附近），
             而且如果谱里有极小奇异值，迭代**根本推不动**它
             （Newton-Schulz 在 0 附近斜率低 -> x=0 是不动点）。
       做法：算完之后强行把 Frobenius 范数 snap 到 √min(m,n)
             —— 那是一个「恰好正交」矩阵应有的范数。
             这能让「推不动的极小奇异值」对应的方向整体被放大回来。
       出处：Muon+, arXiv 2602.21545
    """
    X = G

    # ---- 1) MuonEq 行均衡 -------------------------------------------------
    if use_muon_eq:
        # 目标：整张矩阵的 RMS 行范数 = 1
        target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
        row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        X = X * (target / row_norm).to(X.dtype)

    # ---- 2) 归一化 + 正交化迭代 -------------------------------------------
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)

    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    if use_polar_express:
        coeffs = POLAR_EXPRESS_COEFFS[:steps]
    else:
        # 用经典的固定系数重复 steps 次
        a, b, c = NEWTON_SCHULZ_COEFFS
        coeffs = [(a, b, c)] * steps

    for a, b, c in coeffs:
        if transposed:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
        else:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    if transposed:
        X = X.mT

    # ---- 3) Muon+ 重归一化 -------------------------------------------------
    if use_muon_plus:
        # 一个 m×n 的半正交矩阵（m≤n），Frobenius 范数恰好是 √m
        target_norm = min(X.size(-2), X.size(-1)) ** 0.5
        current_norm = X.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        X = X * (target_norm / current_norm).to(X.dtype)

    return X


# ===========================================================================
# 方差缩减（NorMuon）
# ===========================================================================
def nor_muon_scale(G: torch.Tensor, state: torch.Tensor, beta2: float,
                   red_dim: int) -> torch.Tensor:
    """
    NorMuon（arXiv 2510.05491）：正交化之后还要做一次「逐神经元自适应缩放」。

    动机：Muon 的输出虽然整体正交，但**不同神经元的更新尺度仍然不同**。
          某些神经元会收到远大于其他神经元的更新量，导致训练不稳定。
    做法：估计每个「列/行」上的更新能量 v，用 1/√v 归一化，
          并且把归一化后的整体尺度还原回去（否则范数会变）。

    state 是「因子化的二阶矩」：只存 (chunk, m, 1) 或 (chunk, 1, n)，
    而不是完整的 (chunk, m, n) —— 省 100~1000 倍显存。
    这就是 NorMuon 名字里 "Nor"（Normalized）的来源。

    返回值是一个乘法系数，形状 (..., 1) 或 (..., m, 1)，直接乘到 G 上。
    """
    beta2_t = torch.as_tensor(beta2, dtype=G.dtype, device=G.device)
    v_mean = G.float().square().mean(dim=red_dim, keepdim=True)
    red_size = G.size(red_dim)

    # 归一化前的全局尺度，用来在最后把总范数还原
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_size).sqrt()

    # 更新因子化二阶矩（EMA）
    state.lerp_(v_mean.to(state.dtype), 1 - beta2_t)
    step_size = state.clamp_min(1e-10).rsqrt()             # 1/√v

    # 归一化之后的全局尺度
    scaled_sq_sum = (v_mean * red_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()

    # 最终系数 = 逐元素步长 × 全局还原因子
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    return final_scale.to(G.dtype)


# ===========================================================================
# 自检工具：教程第 23 章的验证实验会用到
# ===========================================================================
def orthogonality_error(X: torch.Tensor) -> float:
    """
    衡量一个矩阵「有多正交」。

    理想情况 X·Xᵀ = I，所以 ||X·Xᵀ - I||_F 应该接近 0。
    返回一个 0~1 之间的数，越小越正交。
    """
    Xf = X.float()
    m = Xf.size(-2)
    A = Xf @ Xf.mT
    eye = torch.eye(m, device=Xf.device, dtype=A.dtype)
    return ((A - eye).norm() / (A.norm() + 1e-9)).item()


def condition_number(X: torch.Tensor) -> float:
    """
    条件数 = 最大奇异值 / 最小奇异值。正交矩阵的条件数是 1。
    用来观察 MuonEq 和 Muon+ 各自改善了什么。
    """
    s = torch.linalg.svdvals(X.float())
    return (s.max() / s.min().clamp_min(1e-12)).item()
