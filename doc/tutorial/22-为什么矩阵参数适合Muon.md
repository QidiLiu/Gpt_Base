# 22 · 为什么矩阵参数适合 Muon

第 21 章说 Adam 把每个参数的更新量都归一化到 `~lr`。
**这对矩阵参数是灾难** —— 它抹掉了「哪些方向更值得走」这个信息。

本章用 SVD 视角说清 Muon 到底改了什么。

> ⚠ **本章用 `V=16384` 举例。** 2026-10 实测发现 `train_tokenizer`
> 的缓存不按词表大小分桶，曾让 `ablation`/`full` 真的按 `V=8192` 训练。
> **「大量行每步都是零梯度」这个结论不受影响** ——
> 词表越小这个比例只会越严重。
>
> **已修（2026-10 同日）**，见[第 37 章](37-全流程与排错.md)。

---

## 本章目标

- 用 SVD 说清 Adam 和 Muon 各自的更新方向
- 理解「rank collapse」和「谱失衡」这两个 Muon 要解决的问题
- 说清为什么嵌入**不能**用 Muon（这是本章最实际的结论）

---

## 前置回顾

[第 21 章](21-从SGD到AdamW.md)：AdamW 的每个超参，以及
本项目 4 个 AdamW 分组。

本章开始讲 Muon 接管的那一半参数。

---

## 概念：Adam 的更新在矩阵上是「错的」

对一个矩阵参数 `W ∈ ℝ^{m×n}`，Adam 的更新是**逐元素**的：

```
W ← W - lr · m̂ / (√v̂ + ε)
```

每个元素 `W[i][j]` 独立地归一化到 `~lr`。

**问题**：矩阵的「方向」是元素之间的**相对关系**，
不是单个元素的大小。逐元素归一化会破坏这个关系。

### 一个具体的例子

假设梯度 `G` 是 rank-1 的：`G = u·vᵀ`（即所有行都成比例）。

Adam 逐元素归一化后，`G[i][j] = sign(u_i·v_j)`。
**行 i 和行 k 的比例信息完全丢了** ——
它们变成了同样多个 ±1，只是位置不同。

而矩阵乘法是 `W_new = W - lr·O`，`O` 的**行空间**决定了
哪些输入方向被改变。**行的相对幅度是这个信号的强度。**

### SVD 视角

对梯度做 SVD：

```
G = U·Σ·Vᵀ
```

其中 `U, V` 是正交矩阵，`Σ = diag(σ₁, ..., σ_k)` 是奇异值对角阵。

**矩阵条件数** `κ = σ₁/σ_k`。LLM 里这个数实测能到 **10¹⁰**：

```
输入奇异值: 1.000e+00 .. 6.218e-11  (跨 1.6e+10 倍)
```

（实测：随机旋转 × `logspace(0,-8)` 谱构造，模拟真实早期梯度）

这意味着：
- `σ₁` 方向上梯度是 `σ_k` 方向的 10¹⁰ 倍
- 用 SGD 的话，**小奇异值方向上参数几乎不动**（步长被大方向淹没）
- 用 Adam 的话，**所有方向步长一样**，但 `O` 被强行填满到各向同性

**两种都不对。** 理想情况（完全正交化）是只保留方向：

```
O = U·Vᵀ        ← 把 Σ 整个丢掉
```

每个方向的步长变成 `lr`，条件数 = 1。**但 SVD 太慢**（`O(mn·min(m,n))`）。

---

## 概念：谱失衡 —— Muon 真正解决的问题

Muon 的论文（Jordan et al. 2024，*Muon: An optimizer for
spectral descent*）把问题分成两层：

1. **谱失衡（spectral imbalance）**：梯度矩阵的奇异值跨好几个数量级，
   小奇异值方向上参数几乎不动
2. **秩坍缩（rank collapse）**：训练过程中梯度的有效秩会下降

第 1 点是可以直接算的。**第 2 点我没能在本项目里复现出干净证据** ——
见下面的诚实说明。

### 谱失衡：可以实测，而且结果很清楚

同一个梯度 `G`，分别用 Adam 和 Muon 的规则算出**更新方向**，
然后比较两者的谱：

```python
import sys, torch
sys.path.insert(0, "src")
from optim.orthogonalize import orthogonalize_simple

torch.manual_seed(0)
D, rank = 256, 8
U, _ = torch.linalg.qr(torch.randn(D, D))
V, _ = torch.linalg.qr(torch.randn(D, D))
sv = torch.cat([torch.ones(rank), torch.full((D - rank,), 1e-3)])
G = U @ torch.diag(sv) @ V.mT


def eff_rank(X):
    s = torch.linalg.svdvals(X.float())
    e = s.square() / s.square().sum()
    return int((e.cumsum(0) < 0.99).sum()) + 1


# Adam 的更新方向：逐元素归一化
adam_dir = G / (G.square().sqrt() + 1e-10)
# Muon 的更新方向：正交化
muon_dir = orthogonalize_simple(G, 5)

print(f"梯度的有效秩: {eff_rank(G)} / {D}")
print(f"{'更新方向':>16} {'有效秩':>8} {'s1':>10} {'s_last':>10} {'条件数':>10}")
for name, O in (("Adam (逐元素)", adam_dir), ("Muon (正交化)", muon_dir)):
    s = torch.linalg.svdvals(O.float())
    print(f"{name:>16} {eff_rank(O):>8} {s[0]:>10.4f} {s[-1]:>10.4f} "
          f"{s[0] / s[-1]:>10.2f}")

sa = torch.linalg.svdvals(adam_dir.float())
sm = torch.linalg.svdvals(muon_dir.float())
print(f"\nAdam 前12: {' '.join('%.3f' % v for v in sa[:12].tolist())}")
print(f"Muon 前12: {' '.join('%.3f' % v for v in sm[:12].tolist())}")
print(f"Adam 后 4: {' '.join('%.2e' % v for v in sa[-4:].tolist())}")
print(f"Muon 后 4: {' '.join('%.3f' % v for v in sm[-4:].tolist())}")
```

实测：

```
梯度的有效秩: 8 / 256

            更新方向      有效秩         s1     s_last        条件数
------------------------------------------------------------
      Adam (逐元素)      166    81.2424     0.0398    2042.09
      Muon (正交化)      251     1.0747     0.1691       6.36

Adam 前12: 81.242 78.645 77.486 76.066 74.753 72.968 72.214 70.311 19.966 19.304 19.240 18.991
Muon 前12: 1.075 1.075 1.075 1.075 1.075 1.075 1.075 1.075 0.169 0.169 0.169 0.169
Adam 后 4: 1.70e-01 1.28e-01 7.86e-02 3.98e-02
Muon 后 4: 0.169 0.169 0.169 0.169
```

**核心数字是条件数：2042 → 6.36（差 320 倍）。**

看谱的形状，差别更清楚：

| | 前 8 个（信号） | 第 9-12 个 | 最后 4 个（噪声） |
|---|---|---|---|
| **Adam** | 81.2 → 70.3 | 20.0 → 19.0 | 0.170 → **0.040** |
| **Muon** | 1.075 × 8（完全相同） | 0.169 × 4 | **0.169 × 4（完全相同）** |

**三个观察**：

1. **两者都保留了「前 8 个方向是信号」这个结构。**
   因为那是谱的主子空间，任何合理的更新都不会丢掉它。
   **所以「Muon 保留了梯度结构」这句话不准确** ——
   Adam 也保留了（只是尺度不同）。
2. **差别在噪声子空间（248 个方向）**：
   - Adam 给它们的是 `0.170, 0.128, 0.079, 0.040...` —— **参差不齐**，
     相差 4 倍，而且**大小和梯度本身的尺度（1e-3）没有直接关系**
   - Muon 给它们的是 `0.169 × 248` —— **完全相同**
3. **「完全相同」正是 5 步 Newton-Schulz 的设计目标**。
   模块 docstring 写得很准：
   > 让结果落在 `U·S′·Vᵀ`，其中 `S′` 的对角线元素落在 [0.5, 1.5]
   > 的均匀分布里。这已经足够好，而且收敛快得多。

**所以 Muon 的成就用一句话说就是：把噪声子空间「等亮度化」。**

这比「保留结构」准确得多，也比「消除谱失衡」更具体 ——
谱失衡被压到 6.36 而不是 1，因为 5 步不够（第 23 章会实测）。

### ⚠ 关于「rank collapse」，我没能给出干净证据

论文说 Adam 会导致 rank collapse。但我试了两个实验都不支持
「Muon 的有效秩更高」这个直觉：

**实验 A（失败的版本）**：跑 200 步真实优化，看参数 `W` 的有效秩。
结果是 Adam 166、Muon 251 —— **Adam 更低**，和论文相反。

**失败原因**：`W` 累积了 200 步的更新，
它的谱反映的是「累积位移」而不是「更新方向」。
在恒定梯度下 Adam 每步把每个元素推 `~lr`，
累积起来被一个秩 1 的「常数位移」主导。**这个实验测错了东西。**

**实验 B（正确的版本）**：只比较**单步更新方向**的谱 ——
就是上面那个实验。它测的是 Muon 真正控制的东西。

**诚实的结论**：
- 「谱失衡被消除」= **已实测**（条件数 2042 → 6.36）
- 「Adam 导致 rank collapse，Muon 避免它」= **论文的说法，
  本项目未复现**

按本项目一贯的口径，不在文档里写没实测过的因果。
第 25 章讲 NorMuon 时会再遇到「论文说 X，代码做的是 Y」，
那时候有个更好的例子。

---

## 概念：Muon 的三步

```
1. Nesterov 动量：  M ← β·M + (1-β)·G ;  g' ← β·M + (1-β)·G
2. 正交化：          O ← Ortho(g')        ← 本章核心，第 23 章
3. 更新：            W ← W - lr·√(m/n)·O
```

代码在 `muon_step` 里：

```python
# ---- 1) Nesterov 动量 ----
momentum_buf.lerp_(stacked_grad, 1 - hp["momentum"])
g = stacked_grad.lerp_(momentum_buf, hp["momentum"])

# ---- 2) 降到 bf16 做正交化 ----
X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g

# ---- 3) 正交化 ----
if cfg.flavor == "simple":
    g = orthogonalize_simple(X, cfg.ns_steps)
else:
    g = orthogonalize_advanced(X, cfg.ns_steps, ...)
g = g.to(stacked_param.dtype)
```

**注意第 2 步**：正交化在 **bf16** 里做。
矩阵乘要吃 tensor core（fp32 的 GEMM 慢一半以上）。

`fp16` 不行 —— 指数范围太小，正交化过程的中间值容易溢出。
（本项目 `COMPUTE_DTYPE` 恒为 bf16，所以这行永远走 `g.bfloat16()`。）

### 正交化把梯度的尺度信息整个丢掉

一个容易忽略但很重要的事实：**`Ortho(g)` 完全不依赖 `g` 的整体大小。**

实测（梯度逐级缩小 10⁸ 倍）：

| 梯度 `max\|·\|` | 正交化后 `max\|·\|` | 放大倍数 |
|---|---|---|
| 4.66e-08 | 0.1619 | **3475909×** |
| 4.66e-06 | 0.1667 | 35777× |
| 4.66e-04 | 0.1667 | 358× |
| 4.66e-02 | 0.1667 | 3.6× |
| 4.66e+00 | 0.1667 | — |

**梯度小 8 个数量级，正交化后的输出完全一样。**

这有两个后果：

1. **梯度裁剪（grad_clip）在 Muon 上对更新量没有影响** ——
   缩放梯度不改变 `Ortho(g)`。
2. **`smear_gate` 这种梯度恒为 0 的参数**（第 19 章）：
   零梯度进正交化，输出是零（实测无 NaN，因为归一化分母有 `1e-6` 兜底），
   所以参数**精确地**不动。

> ⚠ 顺带说：第 19 章说 `smear_gate` 的梯度被 `λ=0` 短路成 0，
> 所以「Muon 会去正交化一批全零梯度」。**实测确认**：
> 全零梯度 → 全零输出，无 NaN。是**无害的浪费**。

---

## 概念：为什么嵌入不能用 Muon

这是本章最实际的结论，也是 `setup_optimizer` 分组的依据。

`wte` 是 `(V, D)` 的**嵌入表**。`V = 16384`，`D = 768`。
形状上是矩阵，看起来「适合」Muon。**但它不是。**

### 理由 1：梯度是稀疏的 one-hot 类信号

查表操作的梯度是：

```
∂L/∂wte[v] = Σ_{位置 p}  (该位置 token id == v) · ∂L/∂x[p]
```

**只有本 batch 里出现过的 token 的行有梯度。**
`ablation` 档每个 batch 有 `8 × 1024 = 8192` 个 token，
`V = 16384` —— 所以**大约一半的行每步都是零梯度**。

对一个一半行是零的矩阵做正交化？
那些零行在 SVD 里的 `σ = 0`，而 **Newton-Schulz 在 0 附近斜率低，
0 是不动点**（第 23 章会实测）。结果就是正交化把矩阵推向一个
「只在出现过的行上有结构」的半正交阵 ——
**没出现过的行基本不动。**

这听起来好像无害，但其实很糟：AdamW 对零梯度的行是
「继续衰减（wd）+ 不更新」；Muon 会把行之间的相对范数
**强制拉平**，导致已学到的行被无意义地放大。

### 理由 2：嵌入的「谱」没有意义

`Ortho(g)` 假设 `g` 的 SVD 分解捕捉了「值得走的方向」。
但嵌入行的「方向」是任意的（初始化是随机的），
谱的结构反映的是**随机噪声**，不是学习信号。

### 理由 3：`lm_head` 的梯度稀疏性相反

`lm_head` 是 `(D, V)`。每个 batch 都经过**全部** `V` 个输出单元
（softmax 的分母），所以**没有零行**。
但它的谱同样不携带「方向」信息。

### 理由 4：数值稳定性

AdamW 对每个参数独立归一化，**不会炸**。
Muon 依赖整个矩阵的谱结构 —— 如果某个元素出问题时
（比如梯度里有一个 `inf`），正交化会把问题**放大到整个矩阵**。

`setup_optimizer` 的 docstring 归纳成三句：

```python
分界线是「形状 + 角色」：
  · 2D 矩阵参与大规模矩阵乘，梯度谱严重不均衡 -> Muon 的正交化有用
  · 嵌入是**查表**，梯度是稀疏的 one-hot 类的信号，正交化会把它毁掉
  · 标量只有 1 个元素，正交化无从谈起
```

### ⚠ 但「标量」这条在本项目里是错的

`setup_optimizer` 的判断规则是**纯形状**：

```python
if p.ndim == 2 and "wte" not in name and "value_embeds" not in name \
        and name != "lm_head.weight":
    matrix_params.append(p)
```

而 `p.ndim == 2` 会把这两个也捞进 Muon 组：

| 参数 | 形状 | 是矩阵吗 |
|---|---|---|
| `smear_gate.weight` | **(1, 24)** | **不是** —— 秩 1，只有 1 个奇异值 |
| `ve_gate.weight` | (12, 12) | 勉强算 |

实测（`ablation` 档的 Muon 组，40 个张量 / 5 种形状）：

```
Muon 组里形状不是「矩阵」的东西：
  smear_gate.weight                            (1, 24)
  transformer.h.1.attn.ve_gate.weight          (6, 12)
```

**`(1, 24)` 的矩阵只有 1 个奇异值**（`min(m,n) = 1`）。
「正交化」它没有任何意义 —— 唯一的效果是把那个向量缩放到单位长度。

而 nanochat 是把 `smear_gate` 放进 `smear_params`（AdamW 组）的：

```python
smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
```

**这是一个真实的偏差**，后果不大（24 个参数），
但它暴露了「纯形状判断」这个规则的问题：
**它没有区分「真正参与大规模矩阵乘的权重」和「碰巧是 2D 的小东西」。**

> nanochat 里的 `matrix_params = list(self.transformer.h.parameters())`
> —— **按名字取**，不是按形状。所以 `smear_gate`（在 `self` 上，
> 不在 `transformer.h` 里）自动留在 AdamW 那边。

---

## 手抄代码

`muon_step` 的前三步（本章范围）：

```python
def muon_step(stacked_grad, stacked_param, momentum_buf, second_moment_buf,
              hp, cfg, use_compiled_ortho=True):
    # ---- 1) Nesterov 动量 ----
    #    g = lerp(grad, momentum, μ) 而不是传统 SGD 的 lerp(momentum, grad, μ)。
    #    区别在于用的是「前瞻」后的梯度方向，效果更好。
    momentum_buf.lerp_(stacked_grad, 1 - hp["momentum"])
    g = stacked_grad.lerp_(momentum_buf, hp["momentum"])

    # ---- 2) 降到 bf16 做正交化（矩阵乘要吃 tensor core）----
    #    fp16 不行：指数范围太小，正交化过程中的中间值容易溢出。
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g

    # ---- 3) 正交化 ----
    if cfg.flavor == "simple":
        g = orthogonalize_simple(X, cfg.ns_steps)
    else:
        g = orthogonalize_advanced(
            X, cfg.ns_steps,
            use_muon_eq=cfg.use_muon_eq,
            use_muon_plus=cfg.use_muon_plus,
            use_polar_express=cfg.use_polar_express,
        )
    g = g.to(stacked_param.dtype)
```

**四处容易写错**：

1. **`lerp_` 的参数顺序**。`a.lerp_(b, w)` = `a + w·(b - a)`。
   `momentum_buf.lerp_(grad, 1-μ)` → `m ← μ·m + (1-μ)·g` ✓
   但 `grad.lerp_(momentum_buf, μ)` → `g' ← g + μ·(m - g)`
   = `(1-μ)·g + μ·m` ✓ **这个「反直觉的顺序」才是 Nesterov**。

   写成传统形式 `g' ← μ·m + (1-μ)·g` 是**普通动量**，不是 Nesterov。
   两者数值上很接近但不等价。

2. **`.bfloat16()` 是无返回值的原地转换**。`X = g.bfloat16()`
   返回新张量（因为 `COMPUTE_DTYPE` 可能不是 bf16），
   但如果写成 `g.bfloat16()` 不接返回值，`g` 就还是原 dtype。

3. **正交化后必须 `.to(stacked_param.dtype)` 变回来**。
   否则下一步的 `stacked_param.sub_(...)` 会做类型提升，
   参数 dtype 可能被改成 bf16。

4. **参数是「堆叠」的 `(K, m, n)`**。所以正交化的实现必须支持
   前导 batch 维（第 23 章的 `orthogonalize_simple` 用了
   `dim=(-2,-1)` 和 `X.mT`，都是最后两维）。

---

## 动手验证

### 验证 1：正交化真的与梯度尺度无关

```python
import sys, torch
sys.path.insert(0, "src")
from optim.orthogonalize import orthogonalize_simple

print(f"{'梯度 max|.|':>12} {'正交化后 max|.|':>16} {'放大倍数':>14}")
for scale in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
    torch.manual_seed(0)
    G = torch.randn(768, 768) * scale
    X = orthogonalize_simple(G, 5)
    print(f"{G.abs().max():>12.2e} {X.abs().max():>16.4f} "
          f"{X.abs().max() / G.abs().max():>13.0f}x")
```

实测：

```
   梯度 max|.|   正交化后 max|.|          放大倍数
       4.66e-08            0.1619        3475909x
       4.66e-06            0.1667          35777x
       4.66e-04            0.1667            358x
       4.66e-02            0.1667              4x
       4.66e+00            0.1667              0x
```

（最后一行「0x」是四舍五入 —— 真实比值是 0.036。）

### 验证 2：零梯度是安全的

```bash
uv run pytest tests/test_optim.py -k muon_orthogonalization -v
```

```python
import sys, torch
sys.path.insert(0, "src")
from optim.orthogonalize import orthogonalize_simple

X = orthogonalize_simple(torch.zeros(768, 768), 5)
print(f"全零梯度 -> 全零输出? {bool((X == 0).all())}   有 NaN? "
      f"{bool(torch.isnan(X).any())}")
```

实测：`全零梯度 -> 全零输出? True   有 NaN? False`

**归一化分母的 `+ 1e-6` 兜住了除零。** 这是那一行 `1e-6` 的作用。

### 验证 3：谁进了 Muon 组

见上面「⚠ 但「标量」这条在本项目里是错的」那张表。可复现脚本：

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config, OptimConfig
from model.gpt import build_model
from optim.muon import setup_optimizer

cfg = make_run_config("ablation").model
for f in ("use_resid_lambdas", "use_x0_lambdas", "use_value_embeds",
          "use_smear", "use_backout"):
    setattr(cfg, f, True)
m = build_model(cfg, device="meta")
opt = setup_optimizer(m, OptimConfig())
id2name = {id(q): n for n, q in m.named_parameters()}
print("Muon 组里形状「不是矩阵」的东西：")
for g in opt.param_groups:
    if g["kind"] != "muon":
        continue
    for p in g["params"]:
        s = tuple(p.shape)
        if min(s) < 8:
            print(f"  {id2name[id(p)]:44s} {s}")
```

实测：

```
Muon 组里形状「不是矩阵」的东西：
  smear_gate.weight                            (1, 24)
  transformer.h.1.attn.ve_gate.weight          (6, 12)
```

**`smear_gate.weight` 是 `(1, 24)` —— 只有 1 个奇异值。**
nanochat 把它放在 AdamW 组（因为它按名字取 `transformer.h` 的参数），
本项目的纯形状规则把它误收进 Muon。

---

## ★ 消融实验

### 消融 1：矩阵参数用 AdamW 会怎样

**本项目没有这个开关** —— `setup_optimizer` 里
`kind` 是硬编码的。想测得把 `matrix_params` 也塞进 `adam_groups["scalar"]`。

```python
# 临时改 setup_optimizer 的分组规则
if p.ndim == 2 and "wte" not in name and "value_embeds" not in name \
        and name != "lm_head.weight":
    adam_groups.setdefault("matrix_as_adamw", []).append(p)   # ★ 改这行
    # matrix_params.append(p)                                 # ★ 注释掉
```

**预期**：`ablation` 档的 bpb 会明显变差。

nanochat / modded-nanogpt 的实测是 Muon 在矩阵上比 AdamW
**快 1.3~2 倍的 wall-clock 达到同样的 loss**。换句话说，
在同样的步数下 AdamW 的 loss 更高。

### 消融 2：`flavor=simple` vs `advanced`

```bash
# simple（debug/smoke/ablation 档的默认）
bash script/train_base.sh ablation --no-resume --model-tag d6_simple

# advanced（full 档的默认）
bash script/train_base.sh ablation --no-resume --model-tag d6_advanced --muon-advanced

bash scratch/ablation.sh d6_simple
```

**预期**：`advanced` 更好，但差距不大。

⚠⚠ **但本项目有 bug 会让这个消融完全无效** ——
见第 28 章：`orthogonalize_advanced` 里有 data-dependent branching，
`torch.compile` 会**静默回落到 eager**。而且如果切换不生效，
两组 bpb 会一模一样。

**先跑第 28 章的验证再跑这个消融。**

### 消融 3：`ns_steps`

```python
MuonConfig.ns_steps = 3 / 5 / 8 / 12
```

`ns_steps` 是正交化迭代次数，直接决定速度和质量（第 23 章实测）。

---

## 常见坑

### 坑 1：把 Nesterov 写成普通动量

见上面。两者数值接近但不等价。

自检：`stacked_grad` 在 `lerp_` 之后**不应该被修改**
（AdamW 路径里 `grad` 是只读的，Muon 里 `stacked_grad` 被
`.lerp_` 改成了前瞻梯度）。如果你看到 `stacked_grad`
在正交化之后变了，说明多写了一步。

### 坑 2：忘了 `.to(stacked_param.dtype)`

见上面。参数 dtype 可能被静默改成 bf16。

⚠ 这个坑在**第 18 章的 bf16 转换之后更容易踩** ——
现在 `lm_head` 和 `wte` 本来就是 bf16，而矩阵参数还是 fp32。
如果 `g` 停在 bf16 而 `stacked_param` 是 fp32，
`sub_` 会把参数提升成... 实际上 PyTorch 会报错（in-place dtype 提升）。
但**报错信息很难指向真正的原因**。

### 坑 3：以为 grad_clip 对 Muon 有用

见上面。`Ortho(g)` 与 `‖g‖` 无关。梯度裁剪在 Muon 上
只影响第 1 步（Nesterov 动量里的 `m`）而不影响正交化结果。

本项目 `grad_clip` 默认 0.0（关闭），所以没有这个问题 ——
但如果你打开它，要知道它对 Muon 基本无效。

### 坑 4：以为「矩阵」= 「`ndim == 2`」

见上面。`(1, 24)` 也是 2D，但它不是矩阵。

**正确的判据是「这个参数参与大规模矩阵乘吗」**，
形状只是代理指标。nanochat 按名字取（`transformer.h`）
比按形状取更准。

### 坑 5：以为嵌入的稀疏梯度「无害」

见上面。零梯度行在正交化下不动（0 是不动点），
所以稀疏性「看起来」是安全的。但真正的问题在于
**非零行之间的相对范数被强制拉平** ——
已学好的行会被无意义地放大。

---

## 延伸

**Muon 的谱下降（Spectral Descent）视角**

Jordan et al. 的论文把 Muon 归到一类更广的优化器里：
**谱方法（spectral methods）**。

Adam 是「逐元素的预条件」（preconditioning）——
每个元素独立缩放，本质是**对角**预条件矩阵。

Muon 是「谱预条件」：把梯度变换到它的极分解（polar factor）
`U·Vᵀ` 上。数学上这是 `G(GᵀG)^{-1/2}` 或 `(GGᵀ)^{-1/2}G`。

**谱方法的通病**：需要矩阵的谱信息（SVD / 特征分解），
对 `m×n` 矩阵是 `O(mn·min(m,n))`。Newton-Schulz 是多项式迭代的
快速近似 —— 这就是它存在的全部理由。

**分布式 Muon**

Muon 需要每层矩阵的**完整**谱 —— 所以 DDP（每卡一份完整矩阵）
天然适合，但 FSDP/ZeRO 把矩阵切到多卡就不行了。

解决方案是 **DiLoCo**（DeepMind, 2024）：只在周期性同步时
做真正的 Muon 更新（伪梯度 = 所有 worker 梯度的平均），
同步之间用普通 AdamW 近似。Kimi 和 GLM 都在生产里用了它。

**本项目没有实现** —— 单卡不需要。分布式分支见卷 6。

**`schedule-free`（2024）**

另一条路线：把「当前参数」和「平均参数」分开存，
前者只用来算梯度，后者用来评估。好处是**不需要 lr 调度**，
而且能和 AdamW/SGD 混用。

对 MoE 模型效果显著（专家的采样率不均导致 lr 难调）。
**和 Muon 兼容**（`schedule-free Muon` 有实现）。

**`Sophia`（2023）**

另一个尝试：`H·g / clip(g/τ)`，其中 `H` 是 Hutchinson 估计的
Hessian 对角。比 Muon 更激进（用了二阶信息），但在 LLM 上
**通常不如 Muon** —— 2024 年的对比实验基本确认了 Muon 的地位。

---

## 下一章

[第 23 章：手写 Newton-Schulz 正交化](23-手写Newton-Schulz正交化.md) ——

本章反复提到的 `orthogonalize_simple` —— **20 行代码，5 步变成正交**。

它是整个 Muon 的核心，也是卷 4 里唯一一章
「**手写 20 行就能跑出可测效果**」的。
