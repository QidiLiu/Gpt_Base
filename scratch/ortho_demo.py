"""正交化可视化：看 Newton-Schulz 怎么把一个病态矩阵「拉正」。

第 23 章（教程卷4）的动手验证。跑：

    uv run python scratch/ortho_demo.py

看三件事：
  1. 正交化前，各向异性矩阵的谱差几个数量级（条件数 κ 很大）
  2. 5 步 Newton-Schulz 之后，奇异值全部落在 [0.5, 1.5]（谱被压平）
  3. 为什么 5 步就够 —— 它**不需要**迭代到精确收敛

这个脚本只用 torch（不需要数据、不需要 tokenizer、不需要 GPU），
所以在任何机器上都能秒级跑完。
"""
import torch

from optim.orthogonalize import (
    orthogonalize_simple, orthogonalize_advanced,
    orthogonality_error, condition_number,
    NEWTON_SCHULZ_COEFFS,
)


def spectrum(X):
    """返回排序后的奇异值（降序）。"""
    return torch.linalg.svdvals(X.float())


def report(name, X):
    s = spectrum(X)
    print(f"  {name:22s} 条件数 {condition_number(X):8.2f}   "
          f"正交误差 {orthogonality_error(X):.4f}   "
          f"σ ∈ [{s.min():.3f}, {s.max():.3f}]")
    return s


def main():
    torch.manual_seed(0)
    m = n = 96

    # ── 1) 一个「病态」矩阵：对角线从 1 拉到 100 ──
    # 真实梯度矩阵的谱就是这样不均衡的，这也是 Muon 存在的理由。
    U, _ = torch.linalg.qr(torch.randn(m, n))
    scales = torch.logspace(0, 2, min(m, n))          # 1 .. 100
    G = U @ torch.diag(scales) @ U.T

    print("── 1) 一个各向异性矩阵 ──")
    print(f"  构造：正交基 × diag(1..100) × 正交基ᵀ   尺寸 {m}×{n}")
    s_before = report("原始 G", G)
    print(f"  最大/最小奇异值比 = {s_before.max()/s_before.min():.1f}x"
          f"  -> 直接用 SGD 更新的话，'高频方向'会主导")

    # ── 2) 5 步 Newton-Schulz ──
    print("\n── 2) 5 步 Newton-Schulz（系数 "
          f"{tuple(round(c,4) for c in NEWTON_SCHULZ_COEFFS)}）──")
    O = orthogonalize_simple(G, steps=5)
    s_after = report("正交化后", O)
    print(f"  谱从 [{s_before.min():.2f}, {s_before.max():.2f}] "
          f"压到 [{s_after.min():.2f}, {s_after.max():.2f}]")
    print(f"  目标范数 sqrt(min(m,n)) = {min(m, n) ** 0.5:.2f}，"
          f"实测 {float(O.norm()):.2f}（半正交矩阵的 Frobenius 范数）")

    # ── 3) 逐步数：为什么 5 步就够？──
    print("\n── 3) 迭代步数 vs 正交质量（不需要迭代到精确收敛）──")
    print("  步数   正交误差   条件数")
    for steps in (1, 2, 3, 4, 5, 6, 8):
        X = orthogonalize_simple(G, steps=steps)
        print(f"  {steps:4d}   {orthogonality_error(X):8.4f}   "
              f"{condition_number(X):8.2f}")
    print("  注意：奇异值停在 [0.5, 1.5] 而不是精确的 1 —— 这是**刻意**的。")
    print("  经典的五次多项式在 0 附近斜率太小，继续迭代收益很低；")
    print("  modded-nanogpt 的观察是「不过度收敛」已经足够好。")

    # ── 4) 低秩矩阵：Muon+ 为什么是必需的 ──
    print("\n── 4) 低秩矩阵：只有 Muon+ 救得回来 ──")
    torch.manual_seed(0)
    rank = 8
    L = torch.randn(128, rank) @ torch.randn(rank, 192)   # rank-8，128×192
    print(f"  构造：128×192，rank={rank} -> 谱里有 {128-rank} 个零奇异值")
    variants = {
        "simple (5 步 NS)": orthogonalize_simple(L, 5),
        "advanced 全关": orthogonalize_advanced(L, 5, use_muon_eq=False,
                                                 use_muon_plus=False,
                                                 use_polar_express=False),
        "Polar Express": orthogonalize_advanced(L, 5, use_muon_eq=False,
                                                use_muon_plus=False,
                                                use_polar_express=True),
        "+ MuonEq": orthogonalize_advanced(L, 5, use_muon_eq=True,
                                            use_muon_plus=False,
                                            use_polar_express=True),
        "完整（含 Muon+）": orthogonalize_advanced(L, 5),
    }
    print("  变体                      正交误差")
    for name, X in variants.items():
        print(f"  {name:24s}   {orthogonality_error(X):.4f}")
    print("  原因：0 是 Newton-Schulz 迭代的**不动点** ——")
    print("        谱里的零奇异值迭代多少步都推不动。")
    print("        只有 Muon+ 强行把 Frobenius 范数 snap 回 sqrt(min(m,n))，")
    print("        把「推不动的方向」整体放大，才真正有效。")

    print("\n  验证对应的测试：uv run pytest -k 'orthogonalize or muon_plus' -v")


if __name__ == "__main__":
    main()
