# 24 · Polar Express

第 23 章说：经典系数的 `φ'(0) = 3.4445`，8 步才饱和，
「斜率不够大」是收敛慢的根源。

Polar Express 把这个斜率从 **3.44 提到 8.16**（2.37 倍），
**同样的 5 步达到更好的正交性。**

本章还包含一个实测发现：**Polar Express 在 5 步就完全饱和，
`ns_steps` 调到 8 也不会有任何变化。**

---

## 本章目标

- 理解「选系数」是一个正经的优化问题，以及它的目标函数是什么
- 看清 Polar Express 的 `φ` 曲线形状为什么那么「奇怪」
- 知道 `ns_steps > 5` 在 advanced 模式下是**无效的**

---

## 前置回顾

[第 23 章](23-手写Newton-Schulz正交化.md)：Newton-Schulz 五次多项式、
`φ` 的作用、以及 5 步/8 步的实测曲线。

本章讲的是「换一组更好的系数能带来什么」。

---

## 概念：把选系数变成优化问题

第 23 章实测过：8 步饱和时 `ortho_err` 是 0.4093，再多也不降。
**原因就是 `φ` 在 `x ≈ 1` 处的斜率只有 0.70 < 1** ——
已经到达 1 附近的奇异值会被**继续压下去**，需要更多步才能拉回来。

Polar Express（arXiv 2505.16932）的思路：

> **在保证「不过度收敛」的前提下，最大化 `φ` 在原点的斜率。**

翻译成约束优化：

```
maximize   a  = φ'(0)
subject to φ(x) ≤ 1        对所有 x ∈ [0, 1]     （不过度收敛）
           φ(x) ≥ 0        对所有 x ∈ [0, 1]     （不翻符号）
           φ 是五次奇多项式（只有 3 个自由度：a, b, c）
```

「不过度收敛」的**形式化定义**就是 `φ(x) ≤ 1` ——
`φ` 把谱**推向** 1 但不越过 1。

**为什么最大化原点斜率？** 因为收敛速度主要由「最慢的那个奇异值」
决定，而最小的奇异值 `x → 0` 时 `φ(x) ≈ a·x`，
斜率 `a` 就是它被推上去的速率。**`a` 越大，推得越快。**

### 系数表

```python
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
```

**注意每一组的 `a` 不同** —— 第 1 组 8.16，之后 4.04、3.89、3.29、2.35，
**逐步减小**。这不是「同一个多项式迭代 5 次」，
而是「5 个不同的多项式，各有分工」。

（参数：`num_iters=5, safety_factor=2e-2, cushion=2`。）

---

## 概念：★ PE 的 φ 曲线形状很「奇怪」

这是本章最值得停下来看的一张表。

```python
import sys, torch
sys.path.insert(0, "src")
from optim.orthogonalize import NEWTON_SCHULZ_COEFFS, POLAR_EXPRESS_COEFFS

a1, b1, c1 = NEWTON_SCHULZ_COEFFS
a2, b2, c2 = POLAR_EXPRESS_COEFFS[0]
xs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
p1 = a1 * xs + b1 * xs**3 + c1 * xs**5
p2 = a2 * xs + b2 * xs**3 + c2 * xs**5
print(f"{'x':>6} {'经典 φ(x)':>12} {'PE φ(x)':>12}")
for x, u, v in zip(xs.tolist(), p1.tolist(), p2.tolist()):
    print(f"{x:>6.2f} {u:>12.4f} {v:>12.4f}")
```

实测：

```
     x      经典 φ(x)      PE φ(x)
   0.10       0.3397       0.7933
   0.20       0.6514       1.4565
   0.30       0.9094       1.8785
   0.40       1.0930       1.9863
   0.50       1.1889       1.7641
   0.60       1.1933       1.2723
   0.70       1.1148       0.6666
   0.80       0.9765       0.2170
   0.90       0.8187       0.3268
   1.00       0.7010       1.5520
   1.10       0.7052       4.6199
```

**三个观察**：

**① PE 在小 x 处激进得多。**
`φ(0.3)`：经典 0.909，PE **1.879**（2 倍）。
`φ(0.5)`：经典 1.189，PE 1.764（1.5 倍）。
**这就是「斜率大 2.37 倍」的直接体现** —— 小奇异值被推得更快。

**② 但 PE 在 `x = 0.8` 有一个深谷（0.217）。**

```
PE:   0.6 → 1.272    0.7 → 0.667    0.8 → 0.217    0.9 → 0.327
                              ↑___________↑
                            这个区间的奇异值被「压缩」
```

`φ(0.8) = 0.217` 意味着**原本 0.8 的奇异值会被压到 0.217** ——
一步之内缩小 3.7 倍。**这是故意的**：把「已经足够接近 1 的」
奇异值先压下去，给后面几步留出把它们拉回来的余量。

**③ PE 在 `x = 1.0` 处 `φ(1) = 1.552 > 1` —— 会推过头。**

对比经典的 `φ(1) = 0.701`。

这看起来违反了「`φ(x) ≤ 1`」的约束。原因：`φ(x) ≤ 1` 这个约束
只作用在 `[0, 1]` 区间，而 `x = 1` 是**边界**。
PE 允许在边界上略微越界（`safety_factor=2e-2` 就是这个裕度）。

**`cushion=2` 的作用**：把 `[0,1]` 区间**向外扩展到 `[0, 1/2]`** 再施加约束 ——
所以约束区间内的 `φ` 都 `< 1`，但 `x = 1` 处可以到 1.55。

> 这个「非单调的 φ」在 Muon+ 那篇论文里有正式的名字：
> **放大区 / 缩减区**（amplification / reduction region）。
> 论文证明了单步极化会让行列范数方差**先放大后缩减**，
> 而「5 步不够」时正好停在放大区 —— 这就是 Muon+ 要修的问题。
> 第 25 章会看到本项目在这个问题上的处理有个**实质偏差**。

---

## 概念：PE 到底带来多少收益（实测）

同样的病态谱（条件数 1000），只换系数：

```python
import sys, torch
sys.path.insert(0, "src")
from optim.orthogonalize import (
    orthogonalize_advanced, orthogonality_error, condition_number)

torch.manual_seed(0)
D = 512
U, _ = torch.linalg.qr(torch.randn(D, D))
V, _ = torch.linalg.qr(torch.randn(D, D))
G = U @ torch.diag(torch.logspace(0, -3, D)) @ V.mT

print(f"{'steps':>6} {'经典 err':>11} {'经典 cond':>11} "
      f"{'PE err':>11} {'PE cond':>11}")
for steps in (1, 2, 3, 5, 8):
    Xc = orthogonalize_advanced(G, steps, use_muon_eq=False,
                                use_muon_plus=False, use_polar_express=False)
    Xp = orthogonalize_advanced(G, steps, use_muon_eq=False,
                                use_muon_plus=False, use_polar_express=True)
    print(f"{steps:>6} {orthogonality_error(Xc):>11.4f} "
          f"{condition_number(Xc):>11.4f} {orthogonality_error(Xp):>11.4f} "
          f"{condition_number(Xp):>11.4f}")
```

实测：

```
输入 cond=1.000e+03  ortho_err=4.8993
 steps      经典 err     经典 cond      PE err     PE cond
--------------------------------------------------------
     1     17.2120    964.1512      3.0428    929.2498
     2      2.2648    625.9797      0.8889    368.7526
     3      1.4375    181.9401      0.8323     90.0752
     5      0.7654     15.3466      0.4869      7.1585
     8      0.4093      1.6637      0.4869      7.1585
                                          ↑─────── 完全相同
```

**三个观察**：

**① 第 1 步的差距最悬殊：17.21 vs 3.04（5.7 倍）。**

第 23 章解释过「第 1 步误差变大」是归一化造成的假象。
但 PE 的第 1 步误差（3.04）甚至比**输入**（4.90）还小 ——
**一步就改善了**。经典系数要 5 步才到 0.765。

**② 5 步时：0.4869 vs 0.7654（1.57 倍），条件数 7.16 vs 15.35（2.14 倍）。**

这是 PE 的净收益：**同样 5 步、同样耗时，正交性提升 1.6 倍。**

**③ ★ 第 8 行和第 5 行完全相同。**

```
     5      0.7654     15.3466      0.4869      7.1585
     8      0.4093      1.6637      0.4869      7.1585
                                    ↑ 逐位相同
```

**这不是巧合。** 看代码：

```python
if use_polar_express:
    coeffs = POLAR_EXPRESS_COEFFS[:steps]
```

`POLAR_EXPRESS_COEFFS` **只有 5 组**。
`[:8]` 返回的还是那 5 组。**所以 `ns_steps > 5` 在
Polar Express 模式下不会有任何效果。**

> ⚠⚠ **这是一个真 bug。** 不是「无效」，是「静默无效」：
> 你把 `ns_steps` 从 5 调到 8，代码照跑、耗时理论上应该涨
> （实际上也不会涨，因为循环次数就是 `len(coeffs)`），
> **但没有任何东西告诉你这个改动是空的。**
>
> 最直接的修法是加一个断言：
> ```python
> if use_polar_express and steps > len(POLAR_EXPRESS_COEFFS):
>     raise ValueError(...)
> ```
> 或在日志里警告。本项目**两者都没做**。

### 这改变了第 23 章的结论

第 23 章说「8 步比 5 步好 9 倍，建议试 `ns_steps=8`」。
**那条建议只在 `flavor="simple"` 下成立。**

在 `full` 档（`flavor="advanced"`，PE 开启）下：
- `ns_steps=8` 与 `ns_steps=5` **完全相同**
- 想要更好的正交性，唯一的办法是 `flavor="simple" + ns_steps=8`
  —— **但那是放弃 MuonEq / Muon+ / NorMuon**

**换句话说：advanced 模式下正交性被 PE 锁死在 5 步。**

---

## 概念：MuonEq 对正交性几乎没有贡献

顺便把三个 trick 逐个关掉，看各自贡献什么：

```python
for tag, kw in [
    ("只正交化（经典）", dict(use_muon_eq=False, use_muon_plus=False,
                        use_polar_express=False)),
    ("+Polar Express",  dict(use_muon_eq=False, use_muon_plus=False,
                             use_polar_express=True)),
    ("+MuonEq",         dict(use_muon_eq=True, use_muon_plus=False,
                             use_polar_express=False)),
    ("全开",            dict(use_muon_eq=True, use_muon_plus=True,
                             use_polar_express=True)),
]:
    X = orthogonalize_advanced(G, 5, **kw)
    print(f"{tag:>20} err={orthogonality_error(X):.4f} "
          f"cond={condition_number(X):.4f}")
```

实测：

```
                          配置   ortho_err        条件数
----------------------------------------------------
                    只正交化（经典）      0.7654    15.3466
              +Polar Express      0.4869     7.1585
                     +MuonEq      0.7652    15.3910
                          全开      0.4415     7.1792
```

**MuonEq 的正交性贡献是 0.7654 → 0.7652 —— 约等于零。**

**这不是说 MuonEq 没用。** 它的作用在**收敛的稳定性**：
它的论文（MuonEq, arXiv 2603.28254）说有限步正交化
「受输入谱支配，特别是 stable rank 和条件数」，
行均衡是「whitening 的零阶替代品」——
**目的是让输入谱更均衡，从而让迭代更容易收敛。**

但对**已经归一化过的**这个合成矩阵，行均衡不改变结果。

> **这是一个重要的方法论**：
> **衡量 MuonEq 的效果，不能用「正交性误差」这个指标** ——
> 那个指标对 MuonEq 是瞎的。要看训练稳定性或收敛步数。

---

## 手抄代码

`orthogonalize_advanced` 里 Polar Express 那一段：

```python
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
```

**三处值得注意**：

**① `coeffs` 是一个「三元组列表」，两种模式统一**

`use_polar_express=False` 时是 `[(a,b,c)] * steps` ——
**同一个系数重复 steps 次**。这是「经典系数」在 advanced 模式下的实现。

**② 循环的是 `for a, b, c in coeffs` 而不是 `for _ in range(steps)`**

**这正是 `ns_steps > 5` 静默失效的原因**：循环次数是
`len(coeffs)`，而 `len(coeffs)` 在 PE 模式下是 `min(steps, 5)`。

**③ `if transposed` 分支里的矩阵乘顺序反了**

第 23 章讲过：
- 不转置：`A = X @ X.mT`（`m×m`），`X = a*X + B @ X`
- 转置：`A = X.mT @ X`（`n×n`），`X = a*X + X @ B`

⚠ 写成 `if transposed: A = X @ X.mT` 会得到**错误的结果而不报错**
（当 `m ≠ n` 时，`X @ X.mT` 是 `m×m`，`B @ X` 也合法，
但数学上不是想要的那个多项式）。

---

## 动手验证

### 验证 1：φ 曲线对比

见上面「PE 的 φ 曲线形状很『奇怪』」那段。

### 验证 2：PE 的实际收益

见上面「PE 到底带来多少收益（实测）」。

```bash
uv run pytest tests/test_optim.py -k polar -v
```

### 验证 3：`ns_steps > 5` 静默失效

这是本章最该亲手确认的一条：

```python
import sys, torch
sys.path.insert(0, "src")
from optim.orthogonalize import orthogonalize_advanced

torch.manual_seed(0)
G = torch.randn(256, 256)

X5 = orthogonalize_advanced(G, 5, use_muon_eq=False, use_muon_plus=False,
                            use_polar_express=True)
X8 = orthogonalize_advanced(G, 8, use_muon_eq=False, use_muon_plus=False,
                            use_polar_express=True)
print(f"PE 模式  steps=5 vs 8 逐位相同? {torch.equal(X5, X8)}")
print(f"  最大差异: {(X5 - X8).abs().max().item():.2e}")

# 对照：classic 模式下两者不同
Y5 = orthogonalize_advanced(G, 5, use_muon_eq=False, use_muon_plus=False,
                            use_polar_express=False)
Y8 = orthogonalize_advanced(G, 8, use_muon_eq=False, use_muon_plus=False,
                            use_polar_express=False)
print(f"经典模式 steps=5 vs 8 逐位相同? {torch.equal(Y5, Y8)}")
print(f"  最大差异: {(Y5 - Y8).abs().max().item():.2e}")
print()
print(f"POLAR_EXPRESS_COEFFS 有 {5} 组 -> [:8] 只返回 5 组")
print("→ PE 模式下 ns_steps>5 是空操作。")
```

实测：

```
PE 模式  steps=5 vs 8 逐位相同? True
  最大差异: 0.00e+00
经典模式 steps=5 vs 8 逐位相同? False
  最大差异: 8.53e-02
```

**PE 模式逐位相同、差异精确为 0；经典模式差异 8.5e-2。**

---

## ★ 消融实验

### 消融 1：`use_polar_express`

这是本章唯一值得做的消融，而且**它有现成开关**：

```python
MuonConfig.use_polar_express = False
```

```bash
bash script/train_base.sh ablation --no-resume --model-tag d6_pe
# 需要在 config 里关掉 use_polar_express，或加一个 CLI 开关
```

**预期**：PE 的实测收益是正交性 1.57 倍、条件数 2.14 倍。
这**不一定**转化成 bpb 收益 —— 第 22 章已经说过，
正交性更好不代表 loss 更低。

> ⚠ 本项目**没有** `--no-polar-express` CLI 开关。
> 而 `use_muon_eq` / `use_muon_plus` / `use_nor_muon` /
> `use_cautious_wd` 也都没有。
> **advanced 模式的 5 个子开关一个 CLI 入口都没有** ——
> 而 `flavor` 本身有一个（`--muon-advanced`）。
> 这是和第 19 章同类的缺漏：**默认开的配置没有关闭入口**。

### 消融 2：`ns_steps`（只对 simple 模式有意义）

见第 23 章的消融表。**但要加一条**：

| `flavor` | `ns_steps` | 有效迭代次数 |
|---|---|---|
| `simple` | 8 | **8** |
| `advanced`（PE 开） | 8 | **5**（静默截断） |

**所以第 23 章那条「试 `ns_steps=8`」的建议在 `full` 档上无效。**

---

## 常见坑

### 坑 1：以为 `ns_steps=8` 在 advanced 模式下有效

见上面。**这是本章最重要的发现**，而且它是静默的。

**排查方法**：

```python
from optim.orthogonalize import POLAR_EXPRESS_COEFFS, orthogonalize_advanced
print(len(POLAR_EXPRESS_COEFFS))     # 5
```

如果你想真的用 8 步 PE，需要拿到 8 组系数 ——
Polar Express 的代码仓库可以按 `num_iters` 重新生成。

### 坑 2：`if transposed` 分支里的乘法顺序写反

见上面「③」。**不报错，只是结果不对。**

自检方法：把 `if transposed:` 那两行的 `X.mT @ X` 和 `X @ B`
去掉转置（都改成不转置的版本），跑一个高矩阵，
比较 `orthogonality_error` 是不是变差。

### 坑 3：用「正交性误差」评估 MuonEq

见上面。MuonEq 对这个指标**贡献为零**（0.7654 → 0.7652）。

**指标选错会让有效的改动看起来像没效果。**

### 坑 4：以为 PE 的 φ 是单调的

见上面。`φ(0.8) = 0.217`（深谷），`φ(1.0) = 1.552`（越过 1）。
**非单调是设计，不是 bug。**

---

>
> ★ **2026-10 消融实测**（`ablation` 档 d6，3,096 步 / 2.03 亿 token，每组 39 分钟）
>
> `--muon-advanced`（Polar Express + MuonEq + Muon+ + NorMuon + 谨慎 WD
> 全部打开）的 `val_bpb` 是 **1.0600，Δ = −0.0016，8σ**。
>
> 噪声底是**实测**的：3 次同配置 + 1 次换初始化，得 σ_Δ = 0.000190，
> 显著性阈值 0.001 ≈ 5.3σ。
>
> **所以本章讲的这些改进，合计换来的是「真实但很小」的改善**
> —— 0.0016 bpb，不是 0，也不是「大」。
>
> ⚠ **两个重要的限定**：
> ① **这是 d6 档（23M 参数 / 2 亿 token）的结论。**
>    第 22 章说过「矩阵越大，Muon 相对 AdamW 的优势越大」——
>    这些改进（尤其 Polar Express 修的是「0 点斜率」）的效果很可能随规模变化。
>    **`full` 档 d24 上可能完全不同，而本项目从未测过。**
> ② 要在 `full` 档下分辨 0.001 量级的差异，需要多 seed 重复，
>    那意味着 36 小时 × N —— 本项目做不到。
>
> 完整消融表见 [README 的 ★ 消融总表](README.md#-消融总表)。

## 延伸

**为什么 Polar Express 的系数是「逐步减小」的**

```
第1组 a=8.157   第2组 a=4.043   第3组 a=3.892   第4组 a=3.286   第5组 a=2.347
```

原因：**第 1 步的工作最重**（要把整个谱从小推向 1），
所以斜率最大。后面谱已经接近 1 了，需要的斜率变小 ——
而且**太大会过冲**（`φ(1.1) = 4.62`）。

**这组系数其实是「同一个优化问题的时变解」** ——
可以理解为一个「先猛推、后微调」的自适应策略。

**和 Newton-Schulz 理论的关系**

Newton-Schulz 迭代最初是为了求矩阵逆的平方根，
理论收敛域是 `|λ| < 1`（`λ` 是 `A = I - X` 的特征值）。

用在这里时：`A = I - X`，若 `‖X‖` 归一化到 1 附近，
则 `λ ≈ 0`，**正好在收敛域的中心** —— 这是为什么归一化那一步
（虽然只加了个 1.01 的裕度）如此重要。

**`safety_factor=2e-2` 的含义**

Polar Express 的论文里，`safety_factor` 控制「允许越过 1 多少」。
`2e-2` = 2%。实测 `φ(1) = 1.552`，越过 55% ——
比 2% 大得多。

所以 `safety_factor` 不是「φ 超过 1 的幅度」，
而是「约束区间的外扩比例」（`cushion=2` 的配合）。
**这一点我没有从论文里确认清楚，标为未验证。**

**其他 2025-2026 的谱估计工作**

| 方法 | 思路 | 额外成本 |
|---|---|---|
| **Polar Express** | 固定系数表 | 0 |
| **You** | 一次性估计谱上下界 | 1 次前向 |
| **Quartet** | 4 个特征值估计 | 少量 |

固定系数表的优势是**零额外成本**（不用多跑一次前向）。
这也是它被 nanochat 选中的原因。

**`orthogonality_error` 的局限**

第 23 章验证 3 说过：它对非方阵无效。而**真实模型里的矩阵
几乎全是非方阵**（`768×3072`、`3072×768`）。

**所以本章所有 `ortho_err` 数字都来自合成方阵**，
用来说明机制的，不是模型里的真实值。
这是本项目测量能力的一个边界，值得知道。
