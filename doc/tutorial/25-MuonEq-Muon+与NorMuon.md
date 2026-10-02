# 25 · MuonEq / Muon+ / NorMuon

三个修正，各修一个具体问题。

**这也是全项目「论文说 X，代码做的是 Y」的最完整案例** ——
本章会详细讲 `use_muon_plus` 的实现和它引用的论文
**根本不是一回事**，并给出实测证据。

---

## 本章目标

- 说清 MuonEq / Muon+ / NorMuon 各自修什么问题（以及各自是否真的修了）
- 量化因子化二阶矩省下的显存（`full` 档实测 647 MiB）
- 搞清 `use_muon_plus` 到底实现了什么，以及它和论文的差别

---

## 前置回顾

[第 24 章](24-Polar-Express.md)：Polar Express 换系数带来的收益，
以及 `ns_steps > 5` 静默失效的问题。

本章讲 `orthogonalize_advanced` 里的另外两个修正，以及
`nor_muon_scale`（在 Muon 路径里，不在正交化里）。

---

## 概念：三个修正各自的位置

```
┌─────────────────────────────────────────────────┐
│  orthogonalize_advanced(G, steps, ...)          │
│                                                 │
│  1. MuonEq  行均衡          ← 正交化「之前」    │
│         ↓                                      │
│  2. 归一化 + Newton-Schulz 迭代（5 步）          │
│         ↓                                      │
│  3. Muon+   重归一化        ← 正交化「之后」    │
└─────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────┐
│  nor_muon_scale(G, state, beta2, red_dim)       │
│  逐神经元自适应缩放         ← 正交化之后、Muon 路径里 │
└─────────────────────────────────────────────────┘
            ↓
        谨慎权重衰减（第 26 章）
```

**前两个在正交化内部，第四个在正交化外部。**
这个位置差别很重要 —— 它决定了两者能不能同时生效。

---

## 概念：MuonEq —— 正交化之前先把行拉平

```python
if use_muon_eq:
    # 目标：整张矩阵的 RMS 行范数 = 1
    target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
    row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    X = X * (target / row_norm).to(X.dtype)
```

三行做了三件事：

1. 算整张矩阵的范数，除以 `√m` → 得到「每行平均应该多大」
2. 算每行的范数
3. **每行 rescale 到平均行范数**

```
第 i 行的范数 → 全矩阵的平均行范数
```

**出处**：MuonEq（arXiv 2603.28254）。论文摘要说：

> 我们证明有限步正交化**受输入谱支配**，特别是 stable rank 和条件数，
> 而行/列归一化是 **whitening 的零阶替代品**。
> 对隐藏层权重，**R（行归一化）是默认变体**。

论文报告在 LLaMA2/C4 上，130M/350M/1B 都优于 Muon。

### ⚠ 但它对「正交性」的贡献是零

第 24 章实测过：

```
                          配置   ortho_err        条件数
                    只正交化（经典）      0.7654    15.3466
                     +MuonEq      0.7652    15.3910
```

**0.7654 → 0.7652。** MuonEq 对最终正交性**没有任何改善**。

**这不代表它没用。** 它的作用在**收敛过程** ——
让输入谱更均衡，从而让有限步迭代更容易到达目标。

> ⚠ **但我在本项目里没能测出这个收益。** 我能测的指标
> （`orthogonality_error`、`condition_number`、条件数）
> **对 MuonEq 都是瞎的**。要验证它的真实效果需要看
> 训练稳定性或收敛步数，那是 `ablation` 档 44 分钟起步的实验。
>
> **诚实结论：MuonEq 的机制清楚，但本项目未验证其收益。**

---

## 概念：NorMuon —— 逐神经元自适应缩放

`nor_muon_scale` 在正交化**之后**：

```python
def nor_muon_scale(G, state, beta2, red_dim):
    beta2_t = torch.as_tensor(beta2, dtype=G.dtype, device=G.device)
    v_mean = G.float().square().mean(dim=red_dim, keepdim=True)
    red_size = G.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_size).sqrt()
    state.lerp_(v_mean.to(state.dtype), 1 - beta2_t)
    step_size = state.clamp_min(1e-10).rsqrt()             # 1/√v
    scaled_sq_sum = (v_mean * red_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    return final_scale.to(G.dtype)
```

**和 AdamW 的思想完全一样**：
`1/√v`（v 是梯度平方的 EMA）→ 梯度大的神经元步长自动变小。

**唯一的区别是「在哪个维度上归约」**：

- AdamW：逐**元素**（每个元素一个 `v`）
- **NorMuon：逐行或逐列**（`red_dim = -1` 或 `-2`）

`muon_step` 里选哪个：

```python
m, n = g.size(-2), g.size(-1)
red_dim = -1 if m >= n else -2
```

**高矩阵按行归约，宽矩阵按列归约** —— 总是沿「被归约的维度较短」那个方向。

### 实测：它把行范数方差压到精确的 0

```python
import sys, torch
sys.path.insert(0, "src")
from optim.orthogonalize import orthogonalize_advanced, nor_muon_scale

torch.manual_seed(0)
m, n = 512, 256
U, _ = torch.linalg.qr(torch.randn(m, m))
V, _ = torch.linalg.qr(torch.randn(n, n))
G = (U[:, :n] @ torch.diag(torch.logspace(0, -2, n)) @ V.mT).unsqueeze(0)

X0 = orthogonalize_advanced(G, 5, use_muon_eq=True, use_muon_plus=True)
sc = nor_muon_scale(X0, torch.zeros(1, m, 1), 0.9, -1)
X1 = X0 * sc

def row_var(X):
    return X[0].float().square().sum(dim=-1).var(unbiased=False).item()

print(f"正交化后:        row-norm 方差 = {row_var(X0):.4e}")
print(f"再 NorMuon 缩放: row-norm 方差 = {row_var(X1):.4e}")
```

实测：

```
正交化后:        row-norm 方差 = 2.0073e-03
再 NorMuon 缩放: row-norm 方差 = 9.6589e-15
```

**从 2.0e-3 降到 9.7e-15 —— 压到机器精度的 0。**
（相当于「每一行的 L2 范数完全相同」。）

### ⚠ 它明确地破坏了正交性

这**不是 bug，是取舍**。

数学上：`X1 = D·X0`（D 是对角矩阵）。那么
```
X1·X1ᵀ = D·(X0·X0ᵀ)·D
```
如果 `X0·X0ᵀ = I`，那么 `X1·X1ᵀ = D² ≠ I`。

**「各向同性」（正交）和「逐行等长」是两个互相冲突的目标。**
NorMuon 选了后者，因为**逐神经元等步长比全局各向同性更贴近
「每个神经元学得一样快」这个目标**。

（这和第 25 章开头的表格一致：NorMuon 在正交化**之后**，
所以它可以自由地破坏正交结果而不影响迭代过程。）

### 缩放范围的实测

```python
G = torch.randn(1, 512, 256)
G[0, :10, :] *= 100          # 前 10 行大 100 倍
scale = nor_muon_scale(G, torch.zeros(1, 512, 1), 0.9, -1)
```

实测：

```
scale 形状: (1, 512, 1)
scale 范围: [0.1296, 16.4261]  比值 126.76x
前 10 行均值: 0.1406   其余行均值: 14.1327
  -> 大行拿到 更小 的缩放
```

**梯度大 100 倍的行，缩放系数小 100 倍**（0.14 vs 14.13）。
这就是「Adam 思想用在行维度」。

---

## 概念：★ 因子化二阶矩 —— 省 647 MiB

NorMuon 的状态不是完整的 `(K, m, n)`，而是**因子化**的：

```python
shape = (k, m, 1) if m >= n else (k, 1, n)
st["second_moment_buffer"] = torch.zeros(shape, dtype=p0.dtype, ...)
```

只存「被归约维度上的均值」。

### 省多少（full 档实测）

```python
# 遍历 Muon 组，算完整 vs 因子化
```

实测：

```
       (1, 24) x1    完整二阶矩           24  因子化         24  省 1x
      (12, 12) x12   完整二阶矩        1,728  因子化        144  省 12x
    (768, 768) x96   完整二阶矩   56,623,104  因子化     73,728  省 768x
   (768, 3072) x24   完整二阶矩   56,623,104  因子化     73,728  省 768x
   (3072, 768) x24   完整二阶矩   56,623,104  因子化     73,728  省 768x

Muon 状态总量：完整 339,742,128 vs 实际 170,092,416 -> 省 2.0x  = 647.2 MiB (fp32)
```

**省 647 MiB** —— 这是 16 GB 卡的 **4%**。

注意 5 个分组里**只有 3 个大组能省**（768x-3072x）。
`(1,24)` 那个（`smear_gate`）省不了 —— 它本来就是 24 个元素。

**总倍率只有 2.0x** 是因为 momentum buffer（`K×m×n`，必须完整）
占了 170,092,416 里的绝大部分：

```
momentum: 157 个张量的完整大小 = 169,871,040
二阶矩（因子化）           =    221,376
```

**所以 NorMuon 的因子化只省了二阶矩那一半，而二阶矩本来只占
状态的 0.5%。** 真正的 647 MiB 来自「不存二阶矩」这个决定，
而不是「因子化」。

> 这是个容易被忽略的算术：**因子化让二阶矩从 169M 降到 221K
> （省 768 倍），但因为 momentum 本来就要 170M，
> 总量只省了 2 倍。**
> 换句话说，**因子化 vs 完整二阶矩的差别很小；
> 「要不要有二阶矩」的差别才是大的。**

---

## 概念：★★ Muon+ 的实现和论文不是一回事

这是本章最重要的一节，也是我写这一卷时最想报告的发现。

### 论文说什幺

Muon+（arXiv 2602.21545）的摘要：

> 我们指出这在实践中**不成立**：实际的极化步会**显著放大**
> 行列范数的失衡。我们称之为 **post-polar imbalanced update** 问题。

论文的解法是**在正交化之后插入一个归一化步骤**，
而且是**沿列或沿行**的：

```
Norm_col(X) := X · D_col^{-1},   D_col[j] = sqrt(Σ_i x_ij²)
Norm_row(X) := D_row^{-1} · X,   D_row[i] = sqrt(Σ_j x_ij²)
```

论文的消融（Table 6，LLaMA-350M 验证 ppl）：

| | None | Col | Row | Col-Row | Row-Col |
|---|---|---|---|---|---|
| 350M | 14.02 | 13.73 | 13.46 | **13.41** | 13.44 |

**而且论文明确说 `None`（不归一化）那一列的方差是「被放大」的** ——
列范数方差（`Var(s(X))`）就是衡量失衡的指标。

### 代码里是什么

```python
# ---- 3) Muon+ 重归一化 ---------------------------------------------
if use_muon_plus:
    # 一个 m×n 的半正交矩阵（m≤n），Frobenius 范数恰好是 √m
    target_norm = min(X.size(-2), X.size(-1)) ** 0.5
    current_norm = X.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    X = X * (target_norm / current_norm).to(X.dtype)
```

**这是把全局 Frobenius 范数 snap 到 `√min(m,n)`。**

### 为什么这两个不是一回事

**一个全局标量乘数无法改变行与行之间的相对范数。**

```
X_new = c · X_old        （c 是一个标量）
=> ||X_new[i]|| = c · ||X_old[i]||        对所有 i
=> 行范数的方差 = c² · 原方差            ← 只是整体缩放，方差比例不变
```

**而论文要消除的正是「行与行之间的差异」。**
全局缩放对它**完全无效**。

### 实测证据

用一个列范数跨约 100 倍的人造失衡矩阵，对比三种实现：

```python
import sys, torch
sys.path.insert(0, "src")
from optim.orthogonalize import orthogonalize_advanced

torch.manual_seed(0)
m, n = 256, 128
X = torch.randn(m, n) * torch.logspace(0, -2, n).unsqueeze(0)

def row_var(Y):
    return Y[0].float().square().sum(dim=-1).var(unbiased=False).item()

def col_var(Y):
    return Y[0].float().square().sum(dim=-2).var(unbiased=False).item()

base = orthogonalize_advanced(X, 5, use_muon_eq=False, use_muon_plus=False)

# 论文版的两种
def paper_col(Y):
    d = Y[0].float().square().sum(dim=-2).sqrt().clamp_min(1e-12)
    return Y * (1.0 / d).to(Y.dtype)

def paper_row(Y):
    d = Y[0].float().square().sum(dim=-1).sqrt().clamp_min(1e-12)
    return Y * (1.0 / d.unsqueeze(-1)).to(Y.dtype)

for tag, Y in [("仅正交化（Muon 基线）", base),
               ("本项目的 Muon+", orthogonalize_advanced(
                   X, 5, use_muon_eq=False, use_muon_plus=True)),
               ("论文版 Norm_col", paper_col(base)),
               ("论文版 Norm_row", paper_row(base))]:
    print(f"{tag:>22}  行方差={row_var(Y):.3e}  列方差={col_var(Y):.3e}")
```

实测：

```
      仅正交化（Muon 基线）  行方差=2.358e-03  列方差=8.041e-03
         本项目的 Muon+  行方差=2.235e-03  列方差=7.622e-03
              论文版 Norm_col  行方差=2.200e-03  列方差=1.274e-14
              论文版 Norm_row  行方差=7.591e-15  列方差=2.988e-02
```

**结论一目了然**：

| 实现 | 行方差 | 列方差 | 相对基线的改善 |
|---|---|---|---|
| 仅正交化 | 2.358e-3 | 8.041e-3 | — |
| **本项目的 Muon+** | 2.235e-3 | 7.622e-3 | **5%（形同虚设）** |
| 论文版 `Norm_col` | 2.200e-3 | **1.274e-14** | 列方差 **630 万倍** |
| 论文版 `Norm_row` | **7.591e-15** | 2.988e-2 | 行方差 **31 万倍** |

**论文版把其中一个方向精确归零；本项目的实现只改善了 5%。**

**所以：`use_muon_plus` 不是 Muon+。**
它是一个「全局 Frobenius 范数归一化」，而且实测几乎不做任何事
（5% 的变化来自全局缩放与有限步迭代残留谱的交互，不是设计意图）。

### 代码 docstring 里的动机也是错的

```python
3) Muon+ 重归一化（use_muon_plus）
   问题：有限步迭代不会精确收敛到正交矩阵（奇异值停在 [0.5,1.5] 附近），
         而且如果谱里有极小奇异值，迭代**根本推不动**它
         （Newton-Schulz 在 0 附近斜率低 -> x=0 是不动点）。
   做法：算完之后强行把 Frobenius 范数 snap 到 √min(m,n)
         —— 那是一个「恰好正交」矩阵应有的范数。
   这能让「推不动的极小奇异值」对应的方向整体被放大回来。
```

三个问题：

1. **「推不动的极小奇异值」这个问题是真的**（第 23 章实测：
   `logspace(0,-3)` 的谱，5 步后最小奇异值只有 0.078），
   但「整体放大」对**所有**方向一视同仁，
   **不会改变极小方向相对于其他方向的比例** ——
   所以它解决不了「极小方向学得慢」这个问题。
2. **论文说的动机完全不同**：post-polar **行列失衡被放大**。
3. **「恰好正交的矩阵应有的范数 = √min(m,n)」是对的**
   （半正交矩阵 `U·Vᵀ` 的 Frobenius 范数确实是 `√min(m,n)`），
   但那只是在说「正交结果的范数是多少」，
   **不是说「把任意矩阵的范数调到这个值就变正交了」**。
   实测确认：本项目版本输出的正交性误差是 **1.0000**，
   而同样输入下 `orthogonality_error` 最好的配置（PE + 全开）是 0.4415。

### 我没改它，理由

这是**行为改动**，会改变 `full` 档的所有训练结果。
而且「共享表 vs 每层表」这类问题我倾向于给出开关让用户决定
（用户上次也是这么定的）。

但**文档和注释的假陈述必须修**。我在本章记录了这个发现，
并把 `orthogonalize.py` 的 docstring 改成如实描述。

### 修的话怎么修

```python
def muon_plus_norm(X, mode="row_col"):
    """
    论文（arXiv 2602.21545）的 Muon+：沿指定方向做 L2 归一化。
    实测能把行/列范数方差压到机器精度（1e-14），而全局 Frobenius
    缩放只改善 5%。
    """
    if "col" in mode:
        d = X.float().square().sum(dim=-2).sqrt().clamp_min(1e-12)
        X = X * (1.0 / d).to(X.dtype)
    if "row" in mode:
        d = X.float().square().sum(dim=-1).sqrt().clamp_min(1e-12)
        X = X * (1.0 / d.unsqueeze(-1)).to(X.dtype)
    return X
```

⚠ 注意它会**改变更新量的整体范数**（不再等于 `√min(m,n)`），
所以配套的形状补偿 `lr·√(m/n)` 可能也要调。
论文那边也是单独扫 lr 的。

---

## 手抄代码

`nor_muon_scale` 的四行核心：

```python
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
```

**两处是「配对」的，必须一起看**：

**① `v_norm` 和 `v_norm_new`**

`v_norm` 是**归一化之前**的全局尺度，`v_norm_new` 是**归一化之后**的。
`v_norm / v_norm_new` 是「还原系数」。

为什么需要还原？因为 `1/√v` 归一化之后，**整体范数会变**——
如果不管，AdamW 式的归一化会把整体步长也改掉，
那就不是「逐神经元自适应」而是「整体缩放 + 逐神经元」了。

论文的做法是：归一化之后**保持总范数不变**，
只改变行列之间的相对分布。

**② `red_size` 出现在两处，作用不同**

```python
v_norm = (v_mean.sum(...) * red_size).sqrt()      # 乘
scaled_sq_sum = (v_mean * red_size) * step_size^2  # 也乘
```

因为 `v_mean` 是 `mean(dim=red_dim)` —— **已经被除以 `red_size` 了**。
要还原成「平方和」就得乘回去。

⚠ 漏掉这两处，得到的范数会差 `red_size` 倍。
`full` 档的 `red_size` 是 256 或 3072 —— 差几千倍。

**③ `clamp_min(1e-6)` 和 `clamp_min(1e-10)` 是两个不同的地板**

- `state.clamp_min(1e-10)`：防 `1/√0 = inf`
- `row_norm.clamp_min(1e-6)`（MuonEq 里）：防除零

两个数不能互换 —— 量级差 4 个数量级。

---

## 动手验证

### 验证 1：Muon+ 的实现与论文的差距

见上面「实测证据」那张表。**这是本章最重要的验证** ——
它把「实现和论文不一致」从推测变成了实测数字。

```bash
uv run pytest tests/test_optim.py -k "orthogonalize or nor_muon" -v
```

### 验证 2：NorMuon 把行范数方差压到 0

见上面。

### 验证 3：因子化省多少显存

见上面。

### 验证 4：`beta2` 的作用

⚠ **我第一版的实验设计错了**：`beta2` 是 EMA 的平滑系数，
它的效果体现在**过渡过程**，而我固定输入测的是**不动点** ——
不动点与 `beta2` 无关（都收敛到 `1/v`）。

实测（固定输入，20 步）：

```
beta2= 0.90  最终 scale [0.8897, 1.1828]  20 步内平均变化 0.00000
beta2= 0.95  最终 scale [0.8897, 1.1828]  20 步内平均变化 0.00000
beta2= 0.99  最终 scale [0.8897, 1.1828]  20 步内平均变化 0.00000
```

**三个完全一样 —— 因为测的是不动点。**

正确的实验要**让输入随时间变化**：

```python
import sys, torch
sys.path.insert(0, "src")
from optim.orthogonalize import nor_muon_scale

torch.manual_seed(0)
m, n = 64, 32
base = torch.randn(1, m, n)

for beta2 in (0.9, 0.95, 0.99):
    st = torch.zeros(1, m, 1)
    Xs = [base * (1.0 + 0.5 * i) for i in range(10)]   # 输入在变
    traj = [nor_muon_scale(x.clone(), st, beta2, -1)[0].flatten().clone()
            for x in Xs]
    lag = sum(float((traj[i] - traj[i - 1]).abs().max())
              for i in range(1, len(traj))) / (len(traj) - 1)
    print(f"  beta2={beta2:5.2f} 逐步最大变化 {lag:.5f}")
print("  -> beta2 越大，scale 对输入变化的跟随越慢（更平滑）")
```

**我没有跑完这个实验**，所以 `beta2` 在 NorMuon 里的实际取值
（`muon_step` 里是 `beta2 = group.get("beta2", 0.9)`）
在本项目里**未验证**。

---

## ★ 消融实验

**五个子开关，一个 CLI 入口都没有。**

```python
MuonConfig.use_polar_express   # 第 24 章
MuonConfig.use_muon_eq         # 本章
MuonConfig.use_muon_plus       # 本章
MuonConfig.use_nor_muon        # 本章
MuonConfig.use_cautious_wd     # 第 26 章
```

只有 `flavor` 有 `--muon-advanced`。

**这是和第 19 章同类的缺漏，而且更严重** ——
`flavor=advanced` 本身就是 `full` 档的默认，
所以这 5 个开关决定着 `full` 档的实际行为，
却一个都无法从命令行触达。

```bash
# 基线（5 个全开）
bash script/train_base.sh full --no-resume --model-tag full_adv

# 需要改 config 的 use_* 字段
bash script/train_base.sh full --no-resume --model-tag full_no_nor
```

**预期（诚实版）**：

- **Muon+ 的消融大概率「测不出差别」** ——
  因为实测它的效果只有 5%（见上面）。
  **这个消融会告诉你「现在这个实现基本没用」，
  但不会告诉你论文版本有没有用。**
- **NorMuon 的消融是三个里最值得做的** ——
  实测它把行范数方差从 2.0e-3 压到 9.7e-15，效果最实在。
- **MuonEq 的消融最难解读** ——
  正交性指标对它不敏感（第 24 章实测贡献 0.7654 → 0.7652），
  得看训练稳定性。

---

## 常见坑

### 坑 1：以为 `use_muon_plus` 是论文的 Muon+

见上面。**本章的全部内容就是关于这个的。**

### 坑 2：用 `orthogonality_error` 评价 MuonEq

见上面。实测贡献 0.7654 → 0.7652。

### 坑 3：以为 NorMuon 保留了正交性

见上面。`X1 = D·X0` ⇒ `X1X1ᵀ = D²`，**正交性被破坏是设计**。

### 坑 4：`red_size` 漏乘

见上面「②」。差 `red_size` 倍（256~3072）。

### 坑 5：以为因子化省了 768 倍显存

**总倍率只有 2.0x**（647 MiB），因为 momentum buffer 必须完整，
它占 170M 中的 169.9M。

「省 768x」是**二阶矩这一个张量**的倍率，
不是 Muon 状态总量的倍率。

---

## 延伸

**三篇论文的定位**

| 论文 | arXiv | 修什么 | 本项目实现 |
|---|---|---|---|
| Polar Express | 2505.16932 | 收敛慢（斜率小） | ✅ 一致 |
| MuonEq | 2603.28254 | 输入谱不均衡 | ✅ 一致 |
| **Muon+** | **2602.21545** | **post-polar 行列失衡** | **❌ 不是论文的** |
| NorMuon | 2510.05491 | 逐神经元步长不均 | ✅ 一致 |

**三个修正的位置与相互作用**

```
MuonEq    ：正交化之前 —— 改变输入
Polar Expr：正交化之中 —— 改变迭代轨迹
Muon+     ：正交化之后 —— 改变输出（本项目实现有偏差）
NorMuon   ：正交化之后、优化器路径里 —— 再次改变输出
```

**Muon+ 和 NorMuon 在同一个位置**，理论上会互相抵消 ——
一个把行范数归一（Muon+，论文版），一个把行范数归一（NorMuon）。
**两篇论文的作者是重叠的**（Muon+ 的 Ruijie Zhang 也是 MuonEq 的作者之一），
所以它们大概是**二选一**，不是叠加。

⚠ 本项目默认**两个都开**。如果论文版 Muon+ 和 NorMuon 真的
目标相同，那其中一个是多余的 —— 但既然本项目的 Muon+ 基本无效，
现在的配置实际上是「只有 NorMuon 在起作用」。

**NorMuon 的状态为什么能因子化**

因为 `v_mean = G.square().mean(dim=red_dim)` ——
它对 `red_dim` 之外的维度**完全不敏感**（只依赖和，不依赖具体位置）。
所以存 `(m,1)` 就够，不需要存 `(m,n)`。

这是**精确**的，不是近似：
```
v_mean[i] = (1/n)·Σ_j G[i][j]²
```
存 `v_mean`（m 个数）和存 `G²`（m×n 个数）能算出同样的 `v_mean`。

---

## 下一章

[第 26 章：谨慎权重衰减](26-谨慎权重衰减.md) ——

本章讲的是**第 25 章那个 `ortho_err = 1.0000`** ——
NorMuon 破坏正交性的后果之一。

第 26 章讲 Muon 里另一个「会改变更新方向」的东西：
**权重衰减**。而「谨慎」版本只在梯度与参数同号时才衰减。
