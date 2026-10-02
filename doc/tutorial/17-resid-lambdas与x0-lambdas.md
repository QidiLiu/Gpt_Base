# 17 · resid_lambdas 与 x0_lambdas

卷 3 正式进入 5 个残差流 trick。第一个是最直接的：**在残差上加一个
可学标量。** 24 个参数，几乎不花算力。

---

## 本章目标

- 说清 resid_lambdas 和 x0_lambdas 各自在「总线」上做什么
- 理解为什么 `x0_lambdas` 需要一个独立的 `x0` 快照，而不是复用当前 x
- 亲手验证「恒等初始化会让这两个 trick 在第 0 步完全无效」

---

## 前置回顾

[第 14 章](14-残差流与Pre-LN.md) 建立了「残差流 = 信息总线」这个框架。
[第 16 章](16-meta-device三步建模型.md) 把 `GPT.__init__` 和
`init_weights` 的分工理清了。

本章开始动手改这条总线。**这 5 个 trick 全部是在操纵总线上的信号** ——
如果第 14 章那个比喻没读懂，后面 4 章都会变成一堆魔法。

---

## 概念：resid_lambdas —— 给总线装音量旋钮

第 14 章提到过 ReZero 和 LayerScale：

- **ReZero**（微软）：残差分支乘一个**可学标量**，初始为 **0**
- **LayerScale**（Google）：残差分支乘一个**每通道**的可学缩放

`resid_lambdas` 是 ReZero 的变体，但**初始值不是 0，而是 1.15**：

```python
# __init__：只是占位，真实值在 init_weights 里填
if cfg.use_resid_lambdas:
    self.resid_lambdas = nn.Parameter(torch.ones(cfg.n_layer, device=device))
```

```python
# init_weights：按层线性插值
if cfg.use_resid_lambdas:
    # 深层残差缩放小一点：避免信息在深层过度累积
    for i in range(cfg.n_layer):
        self.resid_lambdas.data[i] = 1.15 - 0.10 * i / max(cfg.n_layer - 1, 1)
```

`full` 档（24 层）的初值：

```
第 0 层 1.150
第 1 层 1.146
  ...每层减 0.10/23 = 0.00435
第 23 层 1.050
```

**为什么要 >1 而不是 =1？** 因为每个 block 会往总线上加东西
（`x = x + attn(...)`）。`1.15` 意味着「总线上已有的信息被放大 15%，
然后再加上这个 block 的新贡献」。这样做的效果是：**旧信息的权重
压过新信息** —— 网络有「倾向于记住」而不是「倾向于覆盖」的先验。

**为什么要随深度递减？** 深层的总线已经很「满」了。如果每层都放大 1.15，
24 层之后 `1.15^24 ≈ 28.6` 倍。递减到 1.05 把纯缩放降到 `9.76` 倍 ——
降了 3 倍。

（注意这里还没算 `x0_lambdas` 的累加。下面会看到两个合起来在
恒等映射下是 **21 倍**。但因为最后有 `rms_norm`，这个倍率本身
不影响 logits —— 重要的是它「让网络自己学到一个合适的倍率」。）

### forward 里怎么用

```python
for i, block in enumerate(self.transformer.h):
    if hasattr(self, "resid_lambdas"):
        x = self.resid_lambdas[i] * x
    if hasattr(self, "x0_lambdas"):
        x = x + self.x0_lambdas[i] * x0
    ...
    x = block(x, cos_sin, self.window_sizes[i], kv_cache, ve)
```

**关键：缩放发生在 block 之前，也就是这一层的「入口」。**
不是出口。这很重要 —— 它意味着第 `i` 层的注意力**和** MLP 看到的
输入都被缩放了，而不只是「这一层的输出被缩放」。

---

## 概念：x0_lambdas —— 把起点按比例加回来

```python
if cfg.use_x0_lambdas:
    self.x0_lambdas = nn.Parameter(torch.zeros(cfg.n_layer, device=device))
```

```python
# init_weights
if cfg.use_x0_lambdas:
    # 浅层更依赖原始嵌入，深层更少
    for i in range(cfg.n_layer):
        self.x0_lambdas.data[i] = 0.20 - 0.15 * i / max(cfg.n_layer - 1, 1)
```

初值从 0.20 递减到 0.05。forward 里：

```python
x = x + self.x0_lambdas[i] * x0
```

`x0` 是**进入第一层之前**的 x 的快照：

```python
# (1) 查表得到初始嵌入
x = self.transformer.wte(idx)
x = rms_norm(x.to(COMPUTE_DTYPE))

# (2) Smear（第 19 章）
if hasattr(self, "smear_lambda"):
    x = self._apply_smear(x, kv_cache)

# (3) 逐层前向
x0 = x            # ← 快照在这里
```

### 为什么需要「加回来」

深层网络的通病：**信息蒸馏**。24 层之后，总线上承载的可能是
「层 3 的输出经过 21 次非线性变换后的结果」，而 token 的直接身份
（这个位置是 "dog" 还是 "run"）被稀释了。

`x0_lambdas` 给网络一条**直连回起点的旁路** —— 每层都可以按需
「重新看一眼原文」。浅层需要多一点（0.20），深层需要少一点（0.05），
因为深层已经能从上下文里推出足够的结论。

> 这个技巧来自 **modded-nanogpt**，原名叫 `x0 resid`。它的思路
> 和 U-Net 的 skip connection 很像：**在每一层都保留一条到输入的
> 短路径**。但 U-Net 跳的是**同一分辨率**，这里跳的是**第 0 层的
> 同一位置**，因为自回归任务里位置本来就是对齐的。

### 为什么不复用当前的 x 当快照

一个常见的手抄错误：

```python
for i, block in enumerate(self.transformer.h):
    x = x + self.x0_lambdas[i] * x        # ❌ 用了当前 x，不是 x0
```

这样写的话，`x0_lambdas[i] * x` 就是在**自己加自己** ——
每层都把总线放大 `1 + x0_lambdas[i]` 倍。`0.20` 意味着每层放大 20%，
24 层后放大 `1.2^24 ≈ 46` 倍。**这不是「加回起点」，是「自我放大」。**

形状上完全合法，loss 也可能下降 —— 但学到的是另一个东西。

---

## 概念：★ 恒等初始化下，这两个 trick 精确地「什么都不做」

这是本章最重要的一个现象，也是本章存在的理由。

`init_weights` 里把每个 block 的两个输出投影都置零：

```python
torch.nn.init.zeros_(block.attn.c_proj.weight)      # 输出投影 = 0 -> 恒等
torch.nn.init.zeros_(block.mlp.c_proj.weight)      # 输出投影 = 0 -> 恒等
```

**所以刚建好的模型里，每个 block 都是严格的恒等映射。**

第 14 章的验证 1 实测过这一点 —— 每层的 RMS 比值精确等于 `1.0000`：

```
层数      0      1      2      3      4      5
init   1.0000 1.0000 1.0000 1.0000 1.0000 1.0000
step15 1.0456 1.0469 1.0486 1.0490 1.0480 1.0471
```

### 那 trick 生效了吗？生效了 —— 但被一个地方精确吸收

`resid_lambdas` 和 `x0_lambdas` 是在**层入口**施加的，所以即使
block 是恒等映射，它们**照样在起作用**。手算一下（`ablation` 档 6 层）：

```
x_{i+1} = resid[i]·x_i + x0[i]·x_0

 i   resid      x0   x_i / x_0
 0  1.1500  0.2000     1.3500
 1  1.1300  0.1700     1.6955
 2  1.1100  0.1400     2.0220
 3  1.0900  0.1100     2.3140
 4  1.0700  0.0800     2.5560
 5  1.0500  0.0500     2.7338
```

总线被放大了 **2.73 倍**。这可不是「什么都没发生」。
（`full` 档 24 层则是 21.07 倍 —— 纯 `resid` 的连乘是 9.76 倍。）

**但注意最后一列**：`x_i / x_0` 在每个元素上都**是同一个常数**
（实测极差只有 1e-6，纯 fp 误差）。也就是说 ——

> **在恒等映射下，x 始终与 x_0 严格平行。**
> 两个 trick 合起来等价于「把整条总线乘一个标量」，
> **方向一点没变**。

原因很简单：`x_0` 加回的是**同一个向量**，而 `resid[i]` 是标量乘。
归纳一下就清楚：若 `x_i ∥ x_0`，则 `resid[i]·x_i + x0[i]·x_0`
还是 `x_0` 的标量倍。

### 那为什么 logits 只差 0.5%？

因为最后那次 `rms_norm` 是**尺度不变**的：

```python
x = rms_norm(x)                    # ← lm_head 之前
logits = self.lm_head(x)
```

`rms_norm(c·x) = rms_norm(x)`，那个 2.73 倍被**完全吸收**。

实测：

```
[恒等映射] 基线 |logits|max = 0.100584
    resid=True  x0=False 逐位相同=False 最大差异=0.000488  相对=  0.5%
    resid=False x0=True  逐位相同=False 最大差异=0.000488  相对=  0.5%
    resid=True  x0=True  逐位相同=False 最大差异=0.000512  相对=  0.5%

[打破恒等映射] 基线 |logits|max = 0.097167
    resid=True  x0=False 逐位相同=False 最大差异=0.032501  相对= 33.4%
    resid=False x0=True  逐位相同=False 最大差异=0.043091  相对= 44.3%
    resid=True  x0=True  逐位相同=False 最大差异=0.055175  相对= 56.8%
```

**0.5% 就是 bf16 舍入量级**（不是逐位相同，因为乘法有舍入）。
一旦 `c_proj` 有非零值，差异立刻跳到 **33% ~ 57%**。

### 所以准确的结论是

**不是「恒等初始化吃掉了 trick」，而是「恒等初始化下 trick 退化成
一个纯缩放，而 `rms_norm` 把纯缩放吃掉了」。**

两个 trick 需要**非线性（block 的真实输出）**才能起作用。
`c_proj=0` 把非线性关掉了，于是 trick 只剩下一个没有后果的标量。

### 这对消融和写判据意味着什么

**写判据时**：如果一个判据「怎么改代码都不红」，先怀疑它是不是
被某个「什么都没发生」的初始状态给废掉了。本项目
`test_core.py` 里的 trick 判据就是这么写的：

```python
# ★ 打破「block 初始是恒等映射」：init_weights 把两个 c_proj 都置零，
#   所以刚建好的模型里每个 block 都是恒等映射，残差流 trick 的影响
#   会被完整地「吃掉」——实测零差异。必须先给 c_proj 一点随机值。
with torch.no_grad():
    for n, prm in model.named_parameters():
        if "c_proj" in n:
            prm.normal_(0.0, 0.05)
```

⚠ 那句注释里的「实测零差异」**措辞不准**（实测是 0.5%，不是 0，
而且原因也不是「被吃掉」）。**判据本身是对的**
（打破恒等映射后差异 33%~57%，判得很紧），
但注释的因果解释错了。

**做消融时**：不要在 `num_iterations` 极小的配置下测这两个 trick ——
你测的是 bf16 舍入，不是 trick 的作用。

> **这是本章最值得带走的一条方法论**：
> **「观测不到效果」有两种完全不同的原因** ——
> (a) 效果不存在；(b) 效果被别处吸收了。
> 区分它们的方式是**找出中间变量直接测**，而不是加大样本量。
> 本例中「总线被放大 2.73 倍」就是那个中间变量。

---

## 概念：这两个标量的优化器待遇

nanochat 给它们**三个独立的分组**，不是一坨：

| 参数 | lr | betas | weight_decay |
|---|---|---|---|
| `resid_lambdas` | `0.005` | `(0.8, 0.95)` | `0.05` |
| `x0_lambdas` | `0.5` | **`(0.96, 0.95)`** | `0.0` |
| `smear_*` / `backout_lambda` | `0.2` | `(0.8, 0.95)` | `0.0` |

三个细节值得说：

1. **`resid_lambdas` 的 lr 是 `x0_lambdas` 的 1/100。**
   因为 `resid_lambdas` 直接乘整条总线 —— 一个学习率稍大的
   标量就能把 24 层的尺度推向爆炸。它还额外吃 `wd=0.05`
   （把 λ 往 1 拉，别让它漂太远）。
2. **`x0_lambdas` 的 `beta1=0.96`**（其他都是 0.8）。更高的
   `beta1` = 更长的动量窗口 = 更平滑的更新。源码注释里直接写了
   `# higher beta1 for x0`。
3. **`smear`/`backout` 单独一组**（lr 0.2），既不是矩阵也不是嵌入。

### ⚠ 本项目把它们合并了

`src/optim/muon.py` 的 `setup_optimizer` 里，所有不属于
矩阵/`wte`/`value_embeds`/`lm_head` 的参数都落进一个 `scalar` 组：

```python
else:
    adam_groups.setdefault("scalar", []).append(p)
```

然后统一给：

```python
if adam_groups.get("scalar"):
    param_groups.append(dict(
        kind="adamw", params=adam_groups["scalar"],
        lr=0.5 * batch_lr_scale, betas=(0.8, 0.95), eps=1e-10,
        weight_decay=0.0))
```

差异：

| 参数 | nanochat | 本项目 | 偏差 |
|---|---|---|---|
| `resid_lambdas` | lr **0.005**, wd 0.05 | lr **0.5**, wd 0 | **lr 大 100×** |
| `x0_lambdas` | lr 0.5, **beta1 0.96** | lr 0.5, beta1 0.8 | 丢了高 beta1 |
| `smear_lambda` / `smear_gate` / `backout_lambda` | lr **0.2** | lr **0.5** | **lr 大 2.5×** |

**`resid_lambdas` 拿到 100 倍的 lr 是最值得警惕的一条** ——
它是唯一直接乘整条总线的参数，尺度失控的风险最高。

> 我把这写进文档，不是要现在改 —— 改它会改变 `full` 档的所有
> 实测数字，属于独立的决策。但**你在读 nanochat 的代码时会看到
> 四个分组，这里只有一坨**，这个差异必须知道。
> 而且它天然是第 27 章（优化器分组）的一个教学案例：
> 「凭什么这几个参数要分开？」

---

## 手抄代码

### 第 1 块：`__init__` 里的占位

```python
# ── 以下都是残差流 trick，默认全部关闭 ──────────────────
# resid_lambdas[i]：第 i 层入口处把残差流整体乘一个可学标量
if cfg.use_resid_lambdas:
    self.resid_lambdas = nn.Parameter(torch.ones(cfg.n_layer, device=device))
# x0_lambdas[i]：第 i 层入口处把「初始嵌入」按比例加回来
if cfg.use_x0_lambdas:
    self.x0_lambdas = nn.Parameter(torch.zeros(cfg.n_layer, device=device))
```

**为什么是 `torch.ones` / `torch.zeros` 而不是随机值？**

这两个值是「假的初值」，只在 meta device 阶段占位。
真实初值在 `init_weights` 里按层插值算出来。

**为什么用 `if cfg.use_*` 包起来，而不是永远建、靠别的方式关掉？**

因为 `use_*` 为假时**参数根本不存在**。这样 `forward` 里可以写：

```python
if hasattr(self, "resid_lambdas"):
```

比 `if cfg.use_resid_lambdas` 更快（省掉一次 config 属性查找），
而且**更安全**：如果参数没建，`self.resid_lambdas[i]` 会立刻
`AttributeError`，而不是静默地用 config 里的旧值。

顺带一提：这个 `hasattr` 模式被重复了 5 次（第 17-19 章），
是本项目里唯一「用代码结构代替配置检查」的地方。

### 第 2 块：`init_weights` 里的真实初值

```python
if cfg.use_resid_lambdas:
    # 深层残差缩放小一点：避免信息在深层过度累积
    for i in range(cfg.n_layer):
        self.resid_lambdas.data[i] = 1.15 - 0.10 * i / max(cfg.n_layer - 1, 1)
if cfg.use_x0_lambdas:
    # 浅层更依赖原始嵌入，深层更少
    for i in range(cfg.n_layer):
        self.x0_lambdas.data[i] = 0.20 - 0.15 * i / max(cfg.n_layer - 1, 1)
```

**三处容易写错**：

1. **`.data[i] = ...` 而不是 `.data = tensor(...)`。**
   前者原地改，保持 `nn.Parameter` 的身份和优化器引用不变；
   后者会**替换整个张量**，优化器手里还握着旧张量的引用，
   于是这个参数**永远不会被更新**（梯度算了，优化器更新的是
   另一块内存）。这个 bug 极难查。
2. **`max(cfg.n_layer - 1, 1)` 那个 `max`。**
   `n_layer=1` 时 `n_layer-1=0`，除零。`max(..., 1)` 把它夹成 1。
   看起来多余，但 `n_layer=1` 是真的能配出来的。
3. **两个初值的「方向」相反**：`resid` 从 1.15 降到 1.05（递减），
   `x0` 从 0.20 降到 0.05（也递减）。**都是浅层大、深层小** ——
   共同的理由是「浅层需要更强的信号，深层已经足够了」。

### 第 3 块：forward 里的两层循环

```python
# (3) 逐层前向
x0 = x
backout_layer = cfg.n_layer // 2
x_backout = None
for i, block in enumerate(self.transformer.h):
    if hasattr(self, "resid_lambdas"):
        x = self.resid_lambdas[i] * x
    if hasattr(self, "x0_lambdas"):
        x = x + self.x0_lambdas[i] * x0
    ve = (self.value_embeds[str(i)](idx).to(x.dtype)
          if hasattr(self, "value_embeds") and str(i) in self.value_embeds else None)
    x = block(x, cos_sin, self.window_sizes[i], kv_cache, ve)
    if i == backout_layer:
        x_backout = x
```

**注意 `x0 = x` 的位置**：在 smear **之后**、循环**之前**。

- 在循环外 —— 否则每层都重新快照，「x0」就变成了「x_{i-1}」
- 在 smear 之后 —— 所以 x0 里**已经包含**了 smear 混进去的
  前一个 token 信息。第 19 章会说明为什么这个顺序是对的

---

## 动手验证

### 验证 1：亲手看见「2.73 倍被 norm 吃掉」

这是本章的核心机制，值得亲手跑一遍。

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.layers import rms_norm

cfg = make_run_config("ablation").model
L = cfg.n_layer
resid = torch.tensor([1.15 - 0.10 * i / (L - 1) for i in range(L)])
x0l = torch.tensor([0.20 - 0.15 * i / (L - 1) for i in range(L)])

torch.manual_seed(0)
x_emb = rms_norm(torch.randn(2, 32, cfg.n_embd))
x0 = x_emb.clone()

print(f"{'层':>2} {'resid':>7} {'x0':>7} {'x_i / x_0 的极差':>18}")
x = x_emb.clone()
for i in range(L):
    x = resid[i] * x + x0l[i] * x0
    ratio = (x / x0).flatten()
    print(f"{i:>2} {resid[i]:>7.4f} {x0l[i]:>7.4f} "
          f"{ratio.max().item() - ratio.min().item():>18.2e}")

# 尺度被 norm 吸收
a, b = rms_norm(x_emb), rms_norm(x)
print(f"\n总缩放 {(x / x_emb).flatten()[0].item():.4f}x")
print(f"rms_norm 之后两者的最大差异: {(a - b).abs().max().item():.2e}  ← 0")
```

实测：

```
层  resid      x0   x_i / x_0 的极差
 0  1.1500  0.2000             0.00e+00
 1  1.1300  0.1700             2.38e-07
 2  1.1100  0.1400             5.96e-07
 3  1.0900  0.1100             7.15e-07
 4  1.0700  0.0800             1.19e-06
 5  1.0500  0.0500             1.91e-06

总缩放 2.7338x
rms_norm 之后两者的最大差异: 9.54e-07  ← 0
```

**两个数字要一起看**：总缩放 2.73 倍（很大），但 norm 后差异
9.5e-07（等于没有）。

### 验证 2：打破恒等映射后，差异跳到 30~57%

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import build_model

def build(resid, x0, break_id=False):
    c = make_run_config("ablation").model
    for f in ("use_value_embeds", "use_smear", "use_backout"):
        setattr(c, f, False)
    c.use_resid_lambdas, c.use_x0_lambdas = resid, x0
    torch.manual_seed(0)
    m = build_model(c, device="cuda")
    if break_id:
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "c_proj" in n:
                    p.normal_(0.0, 0.05)
    return m

x = torch.randint(0, 8192, (2, 64), device="cuda")
for bi, tag in ((False, "恒等映射"), (True, "打破恒等映射")):
    b = build(False, False, bi)
    with torch.no_grad():
        ref = b(x)
    print(f"[{tag}] 基线 |logits|max = {ref.abs().max():.6f}")
    for r, xx in ((True, False), (False, True), (True, True)):
        with torch.no_grad():
            o = build(r, xx, bi)(x)
        d = (o - ref).abs().max().item()
        print(f"    resid={str(r):5s} x0={str(xx):5s} "
              f"最大差异={d:.6f}  相对={d / ref.abs().max().item():5.1%}")
```

实测：

```
[恒等映射] 基线 |logits|max = 0.100584
    resid=True  x0=False 最大差异=0.000488  相对=  0.5%
    resid=False x0=True  最大差异=0.000488  相对=  0.5%
    resid=True  x0=True  最大差异=0.000512  相对=  0.5%
[打破恒等映射] 基线 |logits|max = 0.097167
    resid=True  x0=False 最大差异=0.032501  相对= 33.4%
    resid=False x0=True  最大差异=0.043091  相对= 44.3%
    resid=True  x0=True  最大差异=0.055175  相对= 56.8%
```

**注意一个和直觉相反的结果**：`x0` 单独开（44.3%）比
`resid` 单独开（33.4%）影响更大。

我原以为 `resid`（1.15 倍、乘整条总线）会更强 —— **错了**。
原因是 `x0` 的效果是「加进一个独立方向」，
而 `resid` 只是缩放。方向的变化比尺度的变化更有信息量。

（再次印证了本章开头那句话：**lm_head 之前的 `rms_norm`
把纯尺度变化全吃掉了**。`resid_lambdas` 在恒等映射下
100% 是尺度，所以它的效果必须靠 block 引入方向才能显现。）

### 验证 3：标量的初值确实是线性的

```bash
uv run python -c "
import sys; sys.path.insert(0,'src')
from common.config import make_run_config
cfg = make_run_config('full').model
for f in ('use_resid_lambdas','use_x0_lambdas'): setattr(cfg, f, True)
import torch
from model.gpt import build_model
m = build_model(cfg, device='cuda')
print('resid_lambdas:', ' '.join('%.4f' % v for v in m.resid_lambdas.data[:5]), '... %.4f' % m.resid_lambdas.data[-1])
print('x0_lambdas   :', ' '.join('%.4f' % v for v in m.x0_lambdas.data[:5]), '... %.4f' % m.x0_lambdas.data[-1])
"
```

实测：

```
resid_lambdas: 1.1500 1.1457 1.1413 1.1369 1.1326 ... 1.0500
x0_lambdas   : 0.2000 0.1935 0.1870 0.1804 0.1739 ... 0.0500
```

---

## ★ 消融实验

```bash
# 基线（resid + x0 + ve + smear + backout 全开 = full 的真实配置）
bash script/train_base.sh full --no-resume --model-tag full_all

# 关掉 resid_lambdas
# —— 没有 CLI 开关，需要改 config 的 use_resid_lambdas = False
bash script/train_base.sh full --no-resume --model-tag full_no_resid

# 关掉 x0_lambdas
bash script/train_base.sh full --no-resume --model-tag full_no_x0

bash scratch/ablation.sh full_all
```

**预期（诚实版）**：

- **bpb 的差异可能落在噪声内。** 这两个 trick 的价值更多是
  「训练稳定性 / 收敛速度」，而不是最终 bpb。nanochat 的
  speedrun 里它们的收益体现为**同样步数下 loss 更低**，
  不是「训完了 bpb 差一截」
- 想看到更明显的信号，应该看**中间步数**的 loss 曲线，
  而不是最终 bpb

> ⚠⚠ **`full` 档一次 41 小时。跑这三个消融是 123 小时。**
> 本项目没有那么多算力 —— 这个消融是「设计上该做」的，
> 不是「现在能做完」的。**诚实地说：这几个消融在本项目里
> 尚未执行，不要把上面的「预期」当成实测结果。**

### 更现实的替代：在 `smoke` 档上看**趋势**

```bash
# smoke 档（4 层）跑得快，但层数太少，resid 的逐层差异只有 4 个点
bash script/train_base.sh smoke --no-resume --model-tag s_all
bash script/train_base.sh smoke --no-resume --model-tag s_no_resid
```

⚠ **但 `smoke` 档只有 4 层，`resid_lambdas` 从 1.15 降到 1.10 ——
这个梯度小到没有信号。** 层数越少，trick 越测不出来。

**这是本章最重要的一条方法论**：

> **层数相关的 trick，在小模型上天然测不出来。**
> 不是「跑得不够准」，是**效应本身不存在**。
> 遇到这种消融，正确的做法是承认「本项目测不了」，
> 而不是加大种子数去凑一个噪声结论。

---

## 常见坑

### 坑 1：用当前 x 当 x0

见上面。形状合法、loss 会降，但学的是「自我放大」。

**自查方法**：把 `x0_lambdas` 全部设成 0.2（常数），训几百步，
如果总线的 RMS 爆炸式增长，说明你写成了自我放大。

### 坑 2：`.data = tensor(...)` 而不是 `.data[i] = ...`

见上面。参数**永远不会被更新**，而且没有任何报错 ——
`model.parameters()` 里还有它，`grad` 也非 None，
只是优化器更新的是另一块内存。

**自查**：`opt.param_groups` 里的张量 `id()` 要和
`model.parameters()` 里的对得上。

### 坑 3：以为恒等初始化下能看出 trick 的效果

见上面。实测差异只有 0.5%（bf16 舍入量级），因为两个 trick 在
恒等映射下退化成纯缩放，而 `rms_norm` 把纯缩放吃掉了。

⚠ 顺带一提：`test_core.py` 里那个判据上方的注释写着
「残差流 trick 的影响会被完整地吃掉——实测零差异」。
**「零差异」措辞不准**（实测 0.5%），**「被吃掉」的因果也写错了**
（正确机制是 `rms_norm` 吸收纯缩放）。判据本身是对的，
注释的解释是错的 —— 我写这一章时才发现的，暂未改动。

### 坑 4：把 `hasattr` 写成 `cfg.use_*`

```python
if cfg.use_resid_lambdas:          # 可以，但慢
    x = self.resid_lambdas[i] * x
if hasattr(self, "resid_lambdas"): # 本项目的写法，快且更安全
    x = self.resid_lambdas[i] * x
```

真正的风险是反向：如果哪天有人把 `__init__` 里的 `if cfg.use_*`
改成「总是建参数」，那 `cfg.use_resid_lambdas=False` 就不再能关掉
这个 trick —— 而 `hasattr` 版本会**照常生效**。

**换句话说：`hasattr` 把「开关」的实现细节和 config 解耦了。**
这是好事，但也意味着**你不能靠 `--no-xxx` CLI 开关来关它**
（因为 CLI 改的是 config，而 config 只在 `__init__` 里被读一次）。

### 坑 5：忘了这 24 个参数几乎不算「参数量」

实测（full 档）：

| trick | 加的参数 | 占总参数 | FLOPs/token 增量 |
|---|---|---|---|
| `resid_lambdas` | 24 | 0.000% | 0 |
| `x0_lambdas` | 24 | 0.000% | 0 |

**但它们对优化的影响远大于这个占比。** 因为 `resid_lambdas`
直接乘整条总线 —— 一个 24 元素的向量，控制着 768 维总线在
24 层里的每一次缩放。

**「参数量」和「影响力」是两个不同的量。** 这正是第 27 章
要讲的「优化器分组」的出发点。

---

## 延伸

**ReZero / LayerScale / DeepNet 的对照**

| 方案 | 做法 | 初始值 |
|---|---|---|
| ReZero | 每个残差分支 × 一个可学标量 | **0** |
| **本项目 `resid_lambdas`** | 同上，但**每层一个**且逐层递减 | 1.15 → 1.05 |
| LayerScale | × 一个每通道的可学缩放 | 0.01 |
| DeepNet | 分支输出在**最后**归一化 | — |

注意 ReZero 初始为 0（完全关掉残差分支，让网络从恒等映射开始
「逐层打开」），而本项目初始为 1.15（**一开始就是打开的**）。
两种哲学：

- ReZero：**保守**，让网络自己决定每一层加多少
- 本项目：**自信**，假定「加深就应该保留更多信息」

**DeepNet 的思路值得注意**：它不是缩放，而是**归一化**。
`x = x + f(norm(x))` 里的 `f(norm(x))` 输出是尺度受控的 ——
不管网络学出什么，这一层往上加的量都有固定尺度。
本项目用的是「乘总线的缩放」，控制力弱一些。

**`x0_lambdas` 和 U-Net skip 的异同**

U-Net 的 skip connection 跳的是**编码器同分辨率的特征**，
让解码器能拿到高分辨率的细节。本项目的 `x0` 跳的是**第 0 层的
嵌入**，让每层能拿到 token 身份。

两者形状相同（都是「同一位置、不同深度」的直连），
但语义不同：U-Net 补的是**空间细节**，这里补的是**身份信息**。

**这 5 个 trick 的共同框架**

把它们放一起看，你会发现它们全都在做同一件事 ——
**给残差流加额外的、可控的信息通路**：

| trick | 加的是什么 | 加在哪 |
|---|---|---|
| `resid_lambdas` | 一条**缩放**通路（调音量） | 层入口 |
| `x0_lambdas` | 一条**回到起点**的通路 | 层入口 |
| `value_embeds` | 一条**token 身份**的通路 | 注意力内部（v） |
| `smear` | 一条**前一个 token** 的通路 | 第一层之前 |
| `backout` | 一条**减法**通路（做减法） | 最后 norm 之前 |

**最后一列是理解它们的钥匙**：前四个都在「往上加」，
只有 `backout` 在「往下减」。第 19 章会讲为什么需要减法。

---

## 下一章

[第 18 章：Value Embeddings 与门控](18-Value-Embeddings与门控.md) ——

本章的两个 trick 加起来是 **48 个参数**。
下一章的 trick 加了 **1.5 亿个参数**，占 full 档的 **43.6%**。
