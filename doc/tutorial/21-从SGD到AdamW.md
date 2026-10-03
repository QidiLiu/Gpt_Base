# 21 · 从 SGD 到 AdamW

在讲 Muon 之前，得先把 AdamW 的每个超参讲清楚。

原因很直接：**Muon 只接管矩阵参数。** 本项目 `full` 档里
50.9% 的参数（嵌入类）和 74 个标量还是 AdamW。
不懂 AdamW，就看不懂 Muon 的设计动机 ——
尤其那句「为什么嵌入不能用 Muon」。

---

## 本章目标

- 逐个搞清 `lr` / `beta1` / `beta2` / `eps` / `weight_decay` 在干什么
- 说清 AdamW 的「W」是什么意思，以及它和 L2 正则的区别
- 能解释本项目 4 个 AdamW 分组**为什么各不相同**

---

## 前置回顾

[第 17 章](17-resid-lambdas与x0-lambdas.md) 提到了优化器分组
（`resid_lambdas` 的 lr 该多大），但没说 AdamW 本身怎么工作。

本章补上这块。第 22 章开始讲 Muon。

---

## 概念：从 SGD 出发

最朴素的更新：

```
W ← W - lr · g
```

三个问题让它在 LLM 上训不动：

| 问题 | 症状 |
|---|---|
| **各参数尺度差太多** | 梯度大的参数被推得远，梯度小的几乎不动 |
| **梯度噪声大** | 单步梯度的方向随机，走 Z 字形 |
| **病态方向** | 某些方向曲率极大（陡），另一些极平 |

三个问题对应三个改进：**动量**、**逐参数自适应**、**正交化**。

---

## 概念：动量 —— 用指数移动平均平滑方向

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t
W ← W - lr · m_t
```

`m_t` 是梯度的指数移动平均（EMA）。`β₁ = 0.9` 意味着
大约最近 10 步的平均。

**为什么有效**：如果真实梯度方向是 `d`，每步加一个随机扰动
`ε_t`，那么
- SGD 的净位移：`Σε_t` —— 随机游走，步数 √N
- 动量的净位移：`m_t → d`（因为 EMA 把 `d` 当常数项保留了，
  把 `ε` 的平均掉了）

`β₁` 越大 → 窗口越长 → 越平滑，但**对真实方向的变化反应越慢**。

本项目：`betas_embedding = (0.8, 0.995)`，
`betas_unembedding = (0.8, 0.96)`。

`β₁ = 0.8` 而不是常见的 0.9 —— 窗口更短（~5 步），反应更快。

### Nesterov 变体

Muon 用的是 Nesterov 动量（第 22 章），但 AdamW 路径**不用**。
区别在 muon_step 里：

```python
# Nesterov：先更新动量，再「前瞻」
momentum_buf.lerp_(stacked_grad, 1 - momentum)     # m ← β·m + (1-β)·g
g = stacked_grad.lerp_(momentum_buf, momentum)     # g' = β·m + (1-β)·g
```

`g'` 用的是**已经包含当前梯度**的动量，所以叫「前瞻」。
AdamW 路径没有这一步（`adamw_step` 里直接用 `m32`）。

---

## 概念：β₂ —— 二阶矩，自适应步长

这是 Adam 和 SGD 的**本质区别**。

```
v_t = β₂·v_{t-1} + (1-β₂)·g_t²
```

`v_t` 是**梯度平方**的 EMA。Adam 的更新：

```
W ← W - lr · m_t / (√v_t + ε)
```

**直觉**：如果某个参数的梯度一直是 0.01，那
`√v → 0.01`，`m/√v ≈ 1`，更新量就是 `lr`。
如果梯度一直是 10，`√v → 10`，`m/√v ≈ 1`，更新量还是 `lr`。

**每个参数的实际步长大致等于 `lr`，与它的梯度尺度无关。**

这就是「自适应」的含义。

### β₂ 越大越平滑

`β₂ = 0.999` → 窗口 ~1000 步。`β₂ = 0.995` → ~200 步。

本项目给嵌入用了 `0.995`（比常见值小），给 `lm_head` 用了 `0.96`（更小）。

**为什么嵌入的 β₂ 比输出头大**：嵌入更新频繁（每个 token 都会
碰到不同的行），需要长窗口稳定；`lm_head` 的梯度更平滑
（每个 batch 都看全部词表），短窗口够用。

### ⚠ β₂ = 0.999 在 bf16 下会出事

这是本项目 `adamw_step` docstring 里写的那段：

> bf16 只有 8 位尾数，算 `1 - beta2`（beta2=0.999 时等于 0.001）
> 会直接下溢成 0，动量就再也不衰减了。所以中途必须转 fp32。

bf16 能表示的最小相对间隔约 `2^-8 ≈ 0.0039`。
`1 - 0.999 = 0.001 < 0.0039` → **在 bf16 里就是 0**。

所以 `adamw_step` 的第一行就把一切都转成 fp32：

```python
p32, g32 = p.float(), grad.float()
m32, v32 = exp_avg.float(), exp_avg_sq.float()
```

**只有最后写回是 bf16** —— 那正是我们想省的部分（第 18 章）。

> 这也是为什么 `exp_avg` 存在 bf16 的参数旁边是对的：
> 它们的内容在每次使用前都会被 `.float()`。

---

## 概念：ε —— 防止除零

```python
denom = (v32 / (1 - hp["beta2"] ** hp["step"])).sqrt() + hp["eps"]
```

`eps` 在**分母外面**（加，不是乘进去）。作用是当 `√v → 0` 时
不让除法爆掉。

本项目：`eps = 1e-10`，比 Adam 默认的 `1e-8` 小两个数量级。

**为什么可以这么小**：因为第 18 章的 bf16 转换之后，
`wte` 初期梯度本来就极小（第 19 章的 `smear` 甚至恰好为 0）。
`1e-8` 会在这些位置**主导**整个步长：
`m/√v` 分母被 `1e-8` 撑住，更新量被压到接近 0，
参数就真的不动了。

`1e-10` 把这个地板压得足够低。

---

## 概念：bias correction —— 别在开头用错误的动量

`m_t` 和 `v_t` 初始化为 0。所以开头几步：
- `m_1 = (1-β₁)·g_1` —— 只有真实梯度的 20%
- `v_1 = (1-β₂)·g_1²` —— 只有 0.5%

**直接用会让第一步的更新小得离谱。** 修正：

```
m̂_t = m_t / (1 - β₁^t)        ← 除掉「还没攒够」的部分
v̂_t = v_t / (1 - β₂^t)
```

代码里只修正了 `v`（因为 `m` 最终除以 `√v`，而
`√v` 的偏差修正会顺带抵消一部分）：

```python
denom = (v32 / (1 - hp["beta2"] ** hp["step"])).sqrt() + hp["eps"]
p32.add_(m32 / denom, alpha=-(hp["lr"] / (1 - hp["beta1"] ** hp["step"])))
```

`m` 的修正藏在 `alpha` 里。

**这个写法有个坑**：`hp["beta1"]` 来自 `HyperParams`（第 28 章），
而 `1 - beta1 ** step` 在编译图里是**每步重算**的 ——
`step` 是 0-D tensor 的值，`hp["step"]` 每步都在变。

---

## 概念：weight decay —— 那个「W」是什么意思

AdamW 的名字里那个 W 指 **decoupled**（解耦）。

**L2 正则（Adam+L2）**：把 `λ·W` 加到梯度上

```
g_total = g + λ·W                    ← 衰减混进了梯度
W ← W - lr · Adam(g_total)
```

问题：衰减量被 Adam 的自适应分母**除掉**了。
如果 `√v` 很大（梯度一直很大），`λ·W` 相对就被压小；
如果 `√v` 很小，衰减量被**放大**。**衰减强度和梯度尺度纠缠在一起**，
不可控。

**AdamW（解耦）**：先直接衰减参数，再走 Adam

```python
p32.mul_(1 - hp["lr"] * hp["wd"])     # ← 解耦：直接乘，不经过 Adam
...  # 然后 Adam 更新
```

**每步衰减固定比例 `lr·wd`**，和梯度大小完全无关。

### ⚠ 本项目踩过的坑：`lr·wd` 而不是 `wd`

注意上面写的是 `1 - lr * wd`，不是 `1 - wd`。

这不是笔误，是**每步**衰减。所以总衰减率是 `(1-lr·wd)^T`，
其中 `T` 是总步数。

`full` 档 `lr = 0.02`、`wd = 0.1`（缩放后峰值 0.02134），
所以每步衰减 `0.02 × 0.02134 = 4.27e-4`。

> `wd` 为什么这么小见第 26 章（谨慎 WD）。

---

## 手抄代码

`adamw_step` 全文（这个函数很短，值得逐行抄）：

```python
def adamw_step(p, grad, exp_avg, exp_avg_sq, hp):
    """
    一个 AdamW 更新步，全部融合在一个函数里。

        weight decay (解耦，先做) -> 动量更新 -> 偏差校正 -> 参数更新
    """
    # ── 整段转 fp32（bf16 下 1-beta2 会下溢成 0）──
    p32, g32 = p.float(), grad.float()
    m32, v32 = exp_avg.float(), exp_avg_sq.float()

    # 解耦权重衰减：先衰减（AdamW 的 "W"）
    p32.mul_(1 - hp["lr"] * hp["wd"])

    # 一阶矩、二阶矩
    m32.lerp_(g32, 1 - hp["beta1"])
    v32.lerp_(g32.square(), 1 - hp["beta2"])

    # 偏差校正
    denom = (v32 / (1 - hp["beta2"] ** hp["step"])).sqrt() + hp["eps"]
    p32.add_(m32 / denom, alpha=-(hp["lr"] / (1 - hp["beta1"] ** hp["step"])))

    # 写回（本来就是 fp32 时是 no-op）
    p.copy_(p32); exp_avg.copy_(m32); exp_avg_sq.copy_(v32)
```

**六处值得注意**：

1. **`lerp_` 而不是 `mul_().add_()`**。`lerp_(target, w)` 是
   `self + w·(target - self)`，**单次融合操作**。
   在 `torch.compile` 下明显更快（第 28 章）。
2. **`v32.lerp_(g32.square(), ...)`** —— 传进去的是**临时张量**。
   `.square()` 每次都新建一个，没有内存复用。但在编译图里会被融合。
3. **`eps` 加在 `sqrt` 之后**。写成 `sqrt(v + eps)` 是另一个公式
   （Adam 原论文那样写），数值上略有差别但量级相同。
4. **`alpha=-(lr/(1-β₁^step))` 的负号**。`add_` 只加，所以传负数。
5. **三个状态张量最后一起写回**。分开写会多三次拷贝。
6. **函数体里没有任何 Python 层的 `if`**。这不只是风格问题 ——
   任何 data-dependent branching 都会让 `fullgraph=True` 的编译
   失败（第 28 章那个 23% 的真实 bug）。

---

## 动手验证

### 验证 1：Adam 的「自适应」到底做了什么

```python
import sys, torch
sys.path.insert(0, "src")
from optim.muon import adamw_step, HyperParams

# ⚠⚠⚠ HyperParams 的构造函数**只登记键名，不赋值**。
#    必须先 .set()，否则所有超参是 0，更新算成 NaN 且不报错。
hp = HyperParams(step=0, lr=0, beta1=0, beta2=0, eps=0, wd=0)
hp.set(step=1, lr=0.3, beta1=0.8, beta2=0.995, eps=1e-10, wd=0.0)

print(f"{'梯度尺度':>12} {'|更新量|':>12} {'比值':>8}")
base = None
for scale in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 1000.0):
    p = torch.zeros(1000)
    g = torch.full((1000,), scale)
    adamw_step(p, g, torch.zeros(1000), torch.zeros(1000), hp)
    d = p.abs().max().item()
    if base is None:
        base = d
    print(f"{scale:>12} {d:>12.6f} {d / base:>7.2f}x")
```

实测：

```
        梯度尺度        |更新量|       比值
         0.001     0.300000    1.00x
         0.010     0.300000    1.00x
         0.100     0.300000    1.00x
         1.000     0.300000    1.00x
        10.000     0.300000    1.00x
      1000.000     0.300000    1.00x
```

**梯度差 10⁶ 倍，更新量精确等于 `lr = 0.3`。** 这就是 Adam。

> ⚠⚠ 写这段时我踩了一个真实的 API 坑，值得单独讲：
>
> **`HyperParams(**kwargs)` 只登记键名，所有值恒为 0。**
> 也就是说 `HyperParams(step=1, lr=0.3, beta2=0.995)` 拿到的是
> `lr = 0`、`beta2 = 0`。
>
> 我一开始以为这是个 bug（和直觉相反），想「修」它 ——
> 结果 `tests/test_judging_soundness.py` 里有一条判据
> `test_hyperparams_constructor_does_not_assign_values`
> **专门断言构造函数不赋值**。
>
> 它的 docstring 记着来历：**2026-09 那批坏判据的直接根因**，
> 就是有测试把 kwargs 当赋值传进去，于是所有超参都是 0，
> 偏差校正里 `1 - beta**step` 退化成 1，分母估错，更新算成 NaN。
>
> 修法不是「让它按字面意思赋值」，而是**只保留 `.set()` 一个赋值入口** ——
> 让「我以为 kwargs 生效了」在结构上就不可能发生。
>
> 所以这条是**有意契约，不是 bug**。写任何直接调 `adamw_step` 的
> 验证脚本时，**第一步必须是 `.set()`**。
> 忘了的话：`denom = sqrt(0/1) + eps = eps`，
> `alpha` 里 `1/(1 - 0**1)` 正常，但整体更新是 `0/1e-10` → `inf` → NaN，
> **全程零报错**。

### 验证 2：bias correction 在干什么

⚠ 关键：要看**单步更新量**，不能看累积位置（后者一直在涨，什么也说明不了）。

```python
import sys, torch
sys.path.insert(0, "src")
from optim.muon import adamw_step, HyperParams

g = torch.full((1000,), 0.1)
m_acc, v_acc = torch.zeros(1000), torch.zeros(1000)
print(f"{'已有 step':>10} {'1-β₂^t':>12} {'|单步更新量|':>14}")
step_now = 0
for tgt in (1, 2, 5, 10, 50, 200):
    # 先把动量状态攒到第 tgt 步
    while step_now < tgt:
        m_acc.lerp_(g, 0.2)            # beta1 = 0.8
        v_acc.lerp_(g.square(), 0.005)  # beta2 = 0.995
        step_now += 1
    hp = HyperParams(step=0, lr=0, beta1=0, beta2=0, eps=0, wd=0)
    hp.set(step=step_now, lr=0.3, beta1=0.8, beta2=0.995, eps=1e-10, wd=0.0)
    p = torch.zeros(1000)
    adamw_step(p, g, m_acc.clone(), v_acc.clone(), hp)
    print(f"{step_now:>10} {1 - 0.995 ** step_now:>12.6f} "
          f"{p.abs().max().item():>14.6f}")
```

实测：

```
   已有 step       1-β₂^t        |单步更新量|
----------------------------------------
         1     0.005000       0.382316
         2     0.009975       0.332458
         5     0.024751       0.300932
        10     0.048890       0.293284
        50     0.221687       0.297402
       200     0.633042       0.299566
```

**看第 3 列**：从 `0.382` 收敛到 `0.300`（= `lr`）。

`t=1` 时 `1-β₂ᵗ` 只有 0.005，`√v` 被严重低估，
`m/√v` 被**放大**（不是缩小）—— 第一步多走了 27%。
修正之后各步稳定在 `lr` 附近。

### 验证 3：解耦 WD vs L2 正则

这是本章唯一需要写两套实现的验证。

```python
import sys, torch
sys.path.insert(0, "src")
from optim.muon import adamw_step, HyperParams

def run_adamw(scale, steps=20, lr=0.3, wd=0.1):
    hp = HyperParams(step=0, lr=0, beta1=0, beta2=0, eps=0, wd=0)
    hp.set(step=1, lr=lr, beta1=0.8, beta2=0.995, eps=1e-10, wd=wd)
    p, m, v = torch.ones(1000), torch.zeros(1000), torch.zeros(1000)
    for i in range(steps):
        adamw_step(p, torch.full((1000,), scale), m, v, hp)
        hp._t["step"].fill_(float(i + 1))
    return p.mean().item()

def run_adam_l2(scale, steps=20, lr=0.3, wd=0.1):
    """对照组：Adam + L2 正则（衰减混进梯度）。"""
    hp = HyperParams(step=0, lr=0, beta1=0, beta2=0, eps=0, wd=0)
    hp.set(step=1, lr=lr, beta1=0.8, beta2=0.995, eps=1e-10, wd=0.0)
    p, m, v = torch.ones(1000), torch.zeros(1000), torch.zeros(1000)
    for i in range(steps):
        g = torch.full((1000,), scale) + wd * p     # ★ 衰减混进梯度
        adamw_step(p, g, m, v, hp)
        hp._t["step"].fill_(float(i + 1))
    return p.mean().item()

print(f"{'梯度尺度':>10} {'AdamW mean|W|':>15} {'Adam+L2 mean|W|':>17}")
for scale in (0.01, 0.1, 1.0, 10.0):
    print(f"{scale:>10} {run_adamw(scale):>15.6f} {run_adam_l2(scale):>17.6f}")
```

实测（20 步，`lr=0.3`，`wd=0.1`，梯度恒为正的 `scale`）：

```
      梯度尺度   AdamW mean|W|   Adam+L2 mean|W|
----------------------------------------------
      0.01       -4.025715         -0.109943
       0.1       -4.025717         -0.957952
       1.0       -4.025717         -4.595942
      10.0       -4.025717         -5.010162
```

**两列的对比就是「解耦」的全部含义**：

**AdamW 那一列四个数完全一样**（`-4.025717`，6 位小数全同）——
梯度差 1000 倍，衰减行为**一比特都没变**。

**Adam+L2 那一列从 -0.11 变到 -5.01** —— 45 倍的差异。
梯度越大，衰减越强（因为 `λ·W` 在 `m/√v` 里被放大）。

```bash
uv run pytest tests/test_optim.py -k adamw_decouples -v
```

### 验证 4：四个 AdamW 分组的实际取值

```python
import sys
sys.path.insert(0, "src")
from common.config import make_run_config, OptimConfig
from model.gpt import build_model
from optim.muon import setup_optimizer

cfg = make_run_config("full")
for f in ("use_resid_lambdas", "use_x0_lambdas", "use_value_embeds",
          "use_smear", "use_backout"):
    setattr(cfg.model, f, True)
model = build_model(cfg.model, device="meta")
opt = setup_optimizer(model, OptimConfig())

print(f"{'kind':>7} {'lr':>9} {'betas':>16} {'eps':>7} {'wd':>8} {'张量数':>7}")
for g in opt.param_groups:
    print(f"{g['kind']:>7} {g['lr']:>9.5f} "
          f"{str(g.get('betas', 'momentum 0.95')):>16} "
          f"{g.get('eps', 0):>7.0e} {g.get('weight_decay', 0):>8} "
          f"{len(g['params']):>7}")
```

实测（`full` 档，`dmodel_lr_scale = (768/768)^-0.5 = 1.0000`）：

```
   kind        lr            betas       eps       wd   张量数
--------------------------------------------------------------
  adamw   0.00800    (0.8, 0.96)     1e-10     0.01       1
  adamw   0.30000   (0.8, 0.995)     1e-10    0.001       1
  adamw   0.15000   (0.8, 0.995)     1e-10     0.01      12
  adamw   0.50000    (0.8, 0.95)     1e-10     0.0       4
   muon   0.02000 momentum 0.95      0e+00     0.0       1
   muon   0.02000 momentum 0.95      0e+00     0.0      12
   muon   0.02000 momentum 0.95      0e+00     0.0      96
   muon   0.02000 momentum 0.95      0e+00     0.0      24
   muon   0.02000 momentum 0.95      0e+00     0.0      24
```

**四个 AdamW 组**：

| 组 | lr | betas | wd | 张量 | 角色 |
|---|---|---|---|---|---|
| `lm_head` | 0.008 | (0.8, 0.96) | 0.01 | 1 | 输出头，怕 logits 爆炸 |
| `wte` | 0.3 | (0.8, 0.995) | 0.001 | 1 | 嵌入侧，要快速学表示 |
| `ve` | 0.15 | (0.8, 0.995) | 0.01 | 12 | 嵌入侧的一半 lr |
| `scalar` | 0.5 | (0.8, 0.95) | 0.0 | 4 | 74 个逐层标量 |

**注意 `scalar` 组的 4 个张量** —— 那是 `resid_lambdas` +
`x0_lambdas` + `smear_lambda` + `backout_lambda`。
第 17 章说过：nanochat 给它们 3 套不同的超参，本项目合并成一套。

**注意 Muon 组的 5 个分组** —— 按**形状**分（`sort({p.shape})`），
因为同形才能 `torch.stack`。96 个 `(768,768)` 是所有方阵投影。

---

## ★ 消融实验

本章的四个分组本身就是一组消融。

```bash
# 基线（full 档默认：advanced + 全部分组）
bash script/train_base.sh full --no-resume --model-tag full_adamw

# 把嵌入的 lr 调成和 lm_head 一样
# （改 AdamWConfig.embedding_lr = 0.008）
bash script/train_base.sh full --no-resume --model-tag full_emb_lowlr
```

**预期**：`embedding_lr = 0.3` 是 nanochat 调出来的值，
比 `unembedding_lr = 0.008` **大 37.5 倍**。

为什么？两个位置的角色完全不同：

| 位置 | 作用 | 需要的 lr |
|---|---|---|
| `wte` | 从零开始学「token 的向量表示」 | **大** —— 要快速建立表示 |
| `lm_head` | 把残差流映射到词表 logits | **小** —— 大了 logits 爆炸（第 15 章） |

**这个 37.5 倍的差距是本章最重要的实践结论。**

⚠ 消融要在 `ablation` 档做（44 分钟），不要在 `full` 档（41 小时）。

```bash
bash script/train_base.sh ablation --no-resume --model-tag d6_base
# 改 embedding_lr 后
bash script/train_base.sh ablation --no-resume --model-tag d6_emb_lowlr
bash scratch/ablation.sh d6_base
```

**但要诚实**：`ablation` 档（d6）的嵌入只有 6.3M 参数，
而 `lm_head` 也是 6.3M —— 两者的差异效应在 d6 上可能很小。
**这是第 17 章那条方法论的又一次体现**：小模型上的超参差异
和真实原因的关系没那么直接。

> ★ **2026-10 更正**：「小模型效应落在噪声内」这个说法**本身也是
> 没验证过的假设**。实测 σ_Δ = 0.000190、阈值 0.001，
> 所以 0.001 量级在 d6 上是测得出来的 —— `shared-value-embeds`
> 的 +0.0001 才是唯一真的测不出的。

---

## 常见坑

### 坑 1：`wd` 写成 `1 - wd` 而不是 `1 - lr*wd`

见上面。`wd` 是**每步**的衰减率，`lr` 才是步长。

写成 `1 - wd` 的话，`wd = 0.0213` 时每步衰减 2.13%，
而正确值是 `1 - 0.02 × 0.0213 = 1 - 4.27e-4`（0.043%）。
**差 50 倍** —— 参数会在几百步内被压成 0。

### 坑 2：`eps` 用默认的 1e-8

见上面。bf16 嵌入 + `smear_lambda=0` 的组合会让
某些参数梯度恒为 0，`1e-8` 会把它们的更新完全冻住。

### 坑 3：在 bf16 里算 `1 - beta2`

见上面。这是最隐蔽的一个 —— **它不会报错**，
只是 `1 - beta2` 悄悄变成 0，动量不再衰减，
二阶矩不再更新，Adam 退化成「SGD + 一个常数步长」。

`adamw_step` 第一行的 `.float()` 就是防这个的。

### 坑 4：以为 4 个 AdamW 分组可以合并

第 17 章已经发现：本项目把 `resid_lambdas` / `x0_lambdas` /
`smear` / `backout` 合并成了一个 `scalar` 组
（`lr=0.5`），而 nanochat 是 3 个独立组，
`resid_lambdas` 的 lr 只有 `0.005`。

**AdamW 的四个组（lm_head / wte / ve / scalar）是合理分开的**，
它们的位置角色确实不同。但 scalar 那一组内部
（4 类参数、4 套超参）被合并了 —— 这是本章第 4 组的问题。

### 坑 5：在 AdamW 里加 Nesterov

`adamw_step` **不用** Nesterov 动量。这是 nanochat 的选择。

如果你手抄时照搬 `muon_step` 的
`g = grad.lerp_(momentum_buf, momentum)`，
AdamW 路径就会多一次「前瞻」—— 数值上不炸，
但和 nanochat 的配方不同，效果未知。

**要判断对错只能实测**。本项目没测过。

---

## 延伸

**为什么 `wte` 和 `lm_head` 的 lr 差 37.5 倍不违反 muP**

muP（第 27 章会细讲）说宽度变化时 LR 要按 `(D/768)^-0.5` 缩放。
那是**同一位置之间**的缩放。

而 `wte` 和 `lm_head` 是**不同位置**，它们的 lr 关系由
「各自需要多大的步长」决定，不受 muP 约束。

具体说：
```
wte:        lr = 0.3   ×  (D/768)^-0.5
lm_head:    lr = 0.008 ×  (D/768)^-0.5
两者比值永远是 37.5 —— 任何宽度下都一样。
```

实测四档的 `dmodel_lr_scale`：

| 档位 | `D` | `(D/768)^-0.5` | 有效 `wte` lr |
|---|---|---|---|
| `debug` | 32 | 4.8990 | 1.470 |
| `smoke` | 128 | 2.4495 | 0.735 |
| `ablation` | 384 | 1.4142 | 0.424 |
| `full` | 768 | 1.0000 | 0.300 |

**注意 `debug` 档的 wte lr 是 1.47** —— 比 1 还大。
这在实践中可能过头（参数一步就被推很远），但 `debug` 档
只跑几步做冒烟测试，无所谓。

**Adam 的原始论文（2014）**

Kingma & Ba 的 Adam 有两个版本：
- **Adam**：L2 正则混进梯度
- **AdamW**：解耦权重衰减

后者由 Loshchilov & Hutter（2018，*Decoupled Weight Decay
Regularization*）重新提出。**名字里的 "W" 就是 "decoupled
weight decay" 的缩写**，不是 "weight"。

论文里有一张图证明：解耦之后，**weight decay 和学习率可以
完全解耦调参** —— 混进梯度时，改 lr 会同时改衰减强度。

**Lion（2023）**

`Lion` 试图去掉二阶矩（省一半优化器状态）：
```
u = sign(β₁·m + (1-β₁)·g)          # 只取符号
W ← W - lr · u
```
在 image 上比 Adam 快，但**在 LLM 上通常不如 Adam**。
2024 年的 `schedule-free` 和 `Sophia` 也在这个方向探索。

**本项目为什么还是 AdamW**

因为 Muon 接管了矩阵参数之后，**AdamW 剩下的活很少**：
嵌入类（查表，不需要方向信息）+ 74 个标量。
这些地方 AdamW 的自适应步长是合适的，而且**优化器状态占比
已经不重要了**（第 18 章算过：176M 参数里 151M 是 ve 表，
但它们的状态是 bf16，AdamW 每个参数 2 个状态，共 0.66 GiB）。

---

## 下一章

[第 22 章：为什么矩阵参数适合 Muon](22-为什么矩阵参数适合Muon.md) ——

AdamW 把每个参数的更新量都归一化到 `~lr`。
**这对矩阵参数是灾难**：它抹掉了「哪些方向更值得走」的信息。

第 22 章讲 Muon 怎么用 SVD 视角重新引入这个信息。
