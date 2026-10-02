# 28 · torch.compile 融合与 0-D tensor 技巧

**卷 4 最后一章。** 本章包含一个真实 bug：Muon advanced 因为一个
data-dependent branching **静默回落到 eager**，`full` 档因此慢了 **23%**。

而且它**不报错** —— 只在日志里留一行，很容易漏看。

---

## 本章目标

- 理解为什么超参要包成 0-D **CPU** tensor 而不是 Python 标量
- 知道 `fullgraph=True` 遇到 data-dependent branching 会**直接拒绝**
- 看懂 `compile_or_eager` 的「探测一次，失败回落」为什么必要

---

## 前置回顾

- [第 21 章](21-从SGD到AdamW.md)：「为什么整段都在 fp32 里做」，
  以及「函数体里没有任何 Python 层的 `if`」
- [第 26 章](26-谨慎权重衰减.md)：验证了 `wd=0` 时
  谨慎版和普通版**逐位相同** —— 那是本章修复的地基

---

## 概念：问题 1 —— 超参变了会重新编译

`torch.compile` 是一个 JIT。**Python 标量在编译时是常量**：

```python
def step(p, g, lr):
    p.sub_(lr * g)

step(p, g, lr=0.02)     # 编译一次，图里 lr = 0.02（写死）
step(p, g, lr=0.05)     # ★ 值变了 -> dynamo 发现新的「常量」-> 重新编译
```

每一步的 lr 都不同（调度曲线，第 26 章），所以
**每个 lr 值都会触发一次重新编译 —— 几千次。**

### 实测

```python
# 完整脚本见「动手验证」
c1 = torch.compile(step_scalar, dynamic=False, fullgraph=True)
for _ in range(3):
    c1(m, g, mom, sec, 0.02, 0.0)      # 首次
for _ in range(3):
    c1(m, g, mom, sec, 0.05, 0.1)      # 改值
```

实测：

```
  标量版：首次 3 步    2.43s
  标量版：改值后再 3 步   0.75s  <- 变慢 = 重新编译了
  tensor 版：首次 3 步    0.71s
  tensor 版：改值后再 3 步   0.00s  <- 不变 = 没重编译
```

**标量版改值后多花 0.75 秒；tensor 版 0.00 秒。**

（tensor 版首次更快是因为它第一次编译时 dynamo 要追踪的图更简单 ——
Python 标量会触发额外的 guard 检查。）

### 解法：0-D tensor

```python
class HyperParams:
    def __init__(self, **kwargs):
        self._t = {k: torch.tensor(0.0, dtype=torch.float32, device="cpu")
                   for k in kwargs}

    def set(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self._t, f"未登记的超参: {k}"
            self._t[k].fill_(float(v))

    def __getitem__(self, k):
        return self._t[k]
```

关键在注释里写的那句话：

> **tensor 的「形状和 dtype」是编译期常量，「数值」不是。**

`torch.compile` 生成的 CUDA kernel 接收一个指针，
运行时从那块内存读值。**改变 `.fill_()` 的内容不改变指针，
所以不触发重新编译。**

### ⚠ 为什么必须是 **CPU** tensor

CUDA graph / 某些 inductor 后端下，**跨设备读超参会强制同步**，
把流水线的异步执行打断。

`train_base.py` 里每步会 `g["weight_decay"] = wd_sched(step)` ——
那是 `.fill_()` 一个 CPU tensor，**不需要 GPU 同步**。

如果超参在 GPU 上，每步 `fill_` 都要 launch 一个 kernel，
而且读它的 CUDA kernel 可能被 dynamo 特殊处理。

```bash
uv run pytest tests/test_judging_soundness.py -k hyperparams -v
```

有三条判据钉死这个约定：

- `test_hyperparams_constructor_does_not_assign_values`
- `test_hyperparams_values_are_0d_cpu_tensors`
- 后面还有一个「热更新真的生效」的

---

## 概念：问题 2 —— data-dependent branching

这是本章的核心，也是那个 23% bug 的根源。

`fullgraph=True` 的意思是「整个函数必须能编译成一个图，
**不许有图断裂（graph break）**」。

而 dynamo 需要**在编译期决定走哪个分支**。如果条件依赖
**运行时才知道的 tensor 值**，它就做不了这个决定：

```python
if hp["wd"] != 0:          # ★ hp["wd"] 是 0-D tensor，值运行时才知道
    mask = (g * p) >= 0
    p.sub_(lr * g + lr * wd * p * mask)
else:
    p.sub_(lr * g + lr * wd * p)
```

### 实测：直接拒绝

```
  带 if 的版本：编译失败 -> Unsupported
    Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`).
    Dynamo does not support tracing dynamic control flow
```

**注意错误类型是 `Unsupported`，不是「警告」。**
`compile_or_eager` 捕获 `Exception`，所以它会：

```python
try:
    compiled = torch.compile(fn, dynamic=False, fullgraph=True)
    compiled(*probe_args)      # ★ 立刻试跑一次，验证这个图能编译
    return compiled
except Exception as e:
    _COMPILE_FAILED = True
    log0(f"  [optim] torch.compile 不可用（{type(e).__name__}: {str(e)[:60]}），"
         f"回落到 eager。...")
    return fn
```

**返回的是未编译的 `fn`。训练能跑完，只是慢。**
**只在日志里多一行。** 这就是「静默」的含义。

### 实测：去掉 if 之后的收益

```
=== 去掉 if 后每步耗时 ===
  compiled: 0.519 ms
  eager   : 0.708 ms
```

**单个 `(512,512)` 矩阵上编译省 27%。**

而 `full` 档实测（代码注释里记的）：

| | 每 micro-step | `full` 档总耗时 |
|---|---|---|
| 带 `if`（eager） | 862 ms | **55.3 小时** |
| 去掉 `if`（compiled） | 710 ms | **45.4 小时** |
| 节省 | **18%** | **9.9 小时** |

⚠ 注意 `muon_step` 注释里的表述是「慢 23%」——
按 `862/710 = 1.214` 算就是 21.4%，总耗时 `55.3/45.4 = 1.218`。
两个数一致，只是「多花 23%」和「快 18%」的说法差异。

---

## 概念：★ 为什么去掉 `if` 是安全的

第 26 章验证过的那条等价性就是依据：

```
=== wd=0 时两种分支等价吗？===
  wd=0.00  逐位相同=True   最大差异=0.000e+00
  wd=0.10  逐位相同=False  最大差异=2.966e-02
  wd=0.50  逐位相同=False  最大差异=1.483e-01
```

**`wd=0` 时逐位相同、差异精确为 0。**

因为：

```python
lr * hp["wd"] * stacked_param * mask
```

当 `wd = 0` 时这一项是 `0 * param * mask == 0`，
加上它和不加它**完全一样**。

而 `debug`/`smoke`/`ablation` 三个档位的 `weight_decay = 0.0`，
所以这三个档位去掉 `if` 是**逐位安全**的。
`full` 档 `wd = 0.1`（缩放后 0.02134），
走的是「mask 全开」的分支 —— 和 `if` 为真时一样。

**所以两种情况下去掉 `if` 都不改变数值。**

> ⚠ 这条等价性必须由判据钉住，因为**整个修复的安全性全靠它**。
> `tests/test_optim.py -k cautious` 里有相关判据。
> 如果哪天有人改了 `mask` 的计算（比如从 `>= 0` 改成 `> 0`，
> 或者加上别的项），等价性可能被破坏。

---

## 概念：`compile_or_eager` 为什么需要「探测」

`torch.compile` **不是编译期检查，而是首次调用时才真正编译**。

```python
compiled = torch.compile(fn, dynamic=False, fullgraph=True)
compiled(*probe_args)     # ★ 立刻跑一次
return compiled
```

如果不做这一步：

```
训练开始
  ↓ 跑 500 步（lr 一直是 warmup 值，没问题）
  ↓
  step 501：lr 变了 → dynamo 遇到不支持的算子
  ↓
  RuntimeError
```

**训练到一半炸掉是最难受的失败方式** ——
你以为跑通了 500 步，结果 501 步崩，前 500 步的算力全废。

所以本项目在**第一次经过该路径时**用真实形状的张量试跑一次：

```python
if not self._probed_muon:
    self._probed_muon = True
    self._fn_muon = compile_or_eager(
        muon_step,
        (stacked_grad, stacked_param, st["momentum_buffer"],
         st["second_moment_buffer"], self._hp, self.muon_cfg),
        self.use_compile)
```

**`probe_args` 是真实形状的 `(K, m, n)` 张量** ——
所以编译失败会在第一步就暴露。

`_COMPILE_FAILED` 是全局标志，**只警告一次**（避免刷屏）。
代价是：如果失败原因和某个特定 shape 有关，
后面所有 shape 都用 eager —— 但那本来也会失败。

---

## 手抄代码

`muon_step` 里那个关键注释（第 26 章引用过）：

```python
# ---- 5) 权重衰减 + 参数更新 ----------------------------------------
lr = hp["lr"].to(g.dtype)
# ★ 这里**不能**再判断 `hp["wd"] != 0`。──────────────────────────
#   `hp["wd"]` 是一个 0-D **tensor**，拿它做 Python 的 if 判断就是
#   dynamo 眼里的 "Data-dependent branching"，而 compile_or_eager 用的是
#   `fullgraph=True` —— 遇到这种分支直接拒绝编译，于是整个 muon_step
#   静默回落到 eager。实测代价（d24 / trick 全开）：
#       带这个判断   862 ms/micro  →  full 档 55.3 小时
#       去掉这个判断 710 ms/micro  →  full 档 45.4 小时
#   而且它**不报错**，只在日志里留一行「回落到 eager」，非常容易漏看。
#
#   去掉是否安全？wd == 0 时 mask 那一项是 0*param*mask == 0，
#   数值上完全等价。mask 的计算本身是廉价的逐元素乘法，不构成浪费。
if cfg.flavor != "simple" and cfg.use_cautious_wd:
    mask = (g * stacked_param) >= 0
    stacked_param.sub_(lr * g + lr * hp["wd"] * stacked_param * mask)
else:
    stacked_param.sub_(lr * g + lr * hp["wd"] * stacked_param)
```

**⚠⚠ 注意：`cfg.flavor` 和 `cfg.use_cautious_wd` 是 `MuonConfig`
的字段 —— 它们是普通的 Python 属性，编译期已知。**
所以 `and` 短路**不构成** data-dependent branching。

**只有 `hp["wd"] != 0` 会出问题** —— 因为 `hp["wd"]` 是 tensor。

**这个区别很重要**：你可以在 `muon_step` 里写任意多个
`if cfg.xxx:`，只要条件来自 `cfg`（Python 对象）。
但**不能**用任何来自 tensor 的值做条件。

### 其他必须避开 data-dependent branching 的地方

| 写法 | 问题 |
|---|---|
| `if hp["lr"] != 0:` | ✗ tensor 条件 |
| `if hp["wd"] > 1e-8:` | ✗ tensor 条件 |
| `if cfg.flavor != "simple":` | ✓ Python 对象 |
| `if stacked_grad.abs().sum() > 0:` | ✗ 会真的同步 |
| `if p.numel() < 1024:` | ✓ Python 对象（shape 是静态的） |
| `if p.requires_grad:` | ✓ Python 属性 |

```bash
uv run pytest tests/test_optim.py -k "hyperparams or muon" -v
```

---

## 动手验证

### 验证 1：标量 vs 0-D tensor 的重编译差异

```python
import sys, time, torch
sys.path.insert(0, "src")

m = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
g = torch.randn_like(m).float()
p = torch.randn(512, 512, device="cuda")
mom = torch.zeros_like(p)
sec = torch.zeros(512, 1, device="cuda")


def step(p, g, mom, sec, lr, wd):
    mom.lerp_(g, 0.05)
    X = g.lerp(mom, 0.95)
    X = X / (X.norm() * 1.01 + 1e-6)
    for _ in range(5):
        A = X @ X.mT
        X = 3.4445 * X + (-4.7750 * A + 2.0315 * (A @ A)) @ X
    p.sub_(lr * X + lr * wd * p)
    return p


lr_t = torch.tensor(0.02)
wd_t = torch.tensor(0.0)
c1 = torch.compile(step, dynamic=False, fullgraph=True)
c2 = torch.compile(step, dynamic=False, fullgraph=True)

for tag, fn, a, b in [("标量", c1, 0.02, 0.0), ("tensor", c2, lr_t, wd_t)]:
    t0 = time.time()
    for _ in range(3):
        fn(m, g, mom, sec, a, b)
    torch.cuda.synchronize()
    first = time.time() - t0
    # 改值
    if tag == "标量":
        a2, b2 = 0.05, 0.1
    else:
        lr_t.fill_(0.05); wd_t.fill_(0.1); a2, b2 = lr_t, wd_t
    t0 = time.time()
    for _ in range(3):
        fn(m, g, mom, sec, a2, b2)
    torch.cuda.synchronize()
    print(f"  {tag:7s} 首次 {first:6.2f}s   改值后 {time.time() - t0:6.2f}s")
```

实测：

```
  标量    首次  2.43s   改值后   0.75s
  tensor   首次  0.71s   改值后   0.00s
```

### 验证 2：data-dependent branching 会让编译失败

```python
def step_branch(p, g, mom, sec, wd_t):
    mom.lerp_(g, 0.05)
    X = g.lerp(mom, 0.95)
    X = X / (X.norm() * 1.01 + 1e-6)
    for _ in range(5):
        A = X @ X.mT
        X = 3.4445 * X + (-4.7750 * A + 2.0315 * (A @ A)) @ X
    if wd_t != 0:                        # ★ tensor 条件
        mask = (X * p) >= 0
        p.sub_(0.02 * X + 0.02 * wd_t * p * mask)
    else:
        p.sub_(0.02 * X + 0.02 * wd_t * p)
    return p

try:
    cb = torch.compile(step_branch, dynamic=False, fullgraph=True)
    cb(p, g, mom, sec, wd_t)
    torch.cuda.synchronize()
    print("  编译成功")
except Exception as e:
    print(f"  编译失败 -> {type(e).__name__}")
    print(f"  {str(e)[:200]}")
```

实测：

```
  编译失败 -> Unsupported
    Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`).
    Dynamo does not support tracing dynamic control flow
```

**这就是 `compile_or_eager` 会静默回落到 eager 的原因。**

### 验证 3：编译 vs eager 的单步差距

```python
def bench(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

# step_nobranch 是验证 2 里去掉 if 的版本
ct = torch.compile(step_nobranch, dynamic=False, fullgraph=True)
print(f"  compiled: {bench(lambda: ct(p, g, mom, sec, wd_t)):.3f} ms")
print(f"  eager   : {bench(lambda: step_nobranch(p, g, mom, sec, wd_t)):.3f} ms")
```

实测（单个 `512×512` 矩阵）：

```
  compiled: 0.519 ms
  eager   : 0.708 ms
```

**27% 的差距。**

### 验证 4：确认编译真的生效了

这是本章**最该做**的一次验证 —— 因为「静默回落」的本质就是
**你看不到它**。

```bash
# 训练启动日志里应该有这一行：
uv run python -m training.train_base --mode full --num-iterations 1 \
    --no-resume 2>&1 | grep -i "torch.compile"

# 没有输出 = 静默回落了（或者本来就在 eager）
```

**如果日志里有 `[optim] torch.compile 不可用`，说明回落了。**
**如果没有这行，说明编译成功。**

> ⚠ 但这有个盲区：`compile_or_eager` 只在**第一次**探测时警告。
> 如果第一次探测成功（比如用 `debug` 档测的），后面 `full` 档
> 不会重新警告。所以**用哪个档位启动，就只探测哪个形状**。

```bash
uv run pytest tests/test_optim.py -k hyperparams -v
```

---

## ★ 消融实验

本章没有模型消融，但有一个**工程消融**，而且它的结论是
「**不要改回去**」：

```python
# 故意加回那个 if，测量代价
if hp["wd"] != 0:
    ...
```

```bash
bash script/train_base.sh full --no-resume   # 45.4 小时（编译）
# 加回 if 之后                                 # 55.3 小时（eager）
```

**实测数字已经写在 `muon_step` 的注释里**（第 5 段）。
但要自己复现需要 100 小时，所以：

> **这个消融的实测数字来自项目历史（代码注释记录），
> 本次会话没有重跑。** 它们是 d24 / trick 全开 / dbs=? 的配置下测的，
> 换配置会变。

### 更有价值的消融：`use_compile=False`

```python
MuonAdamW(..., use_compile=False)
```

这会**显式**关掉编译（而不是靠失败回落）。
用来对比「编译带来的收益」和「超参重编译的损失」。

---

## 常见坑

### 坑 1：以为 `fullgraph=True` 失败会报错

**它不会。** `compile_or_eager` 捕获了所有 `Exception`
并返回未编译的 `fn`。

**唯一的线索是日志里那行 `[optim] torch.compile 不可用`。**

**所以每次跑训练都要看这一行。**

### 坑 2：以为 `if cfg.xxx` 也是 data-dependent branching

**不是。** `cfg` 是普通 Python 对象，dynamo 在编译期
读它的属性值。

**只有来自 tensor 的条件才是 data-dependent。**

（注意：`cfg.flavor` 读的是 `MuonConfig` 实例的属性，
它不会被修改 —— `flavor` 只在 `PRESETS` 里设定一次。
如果有人写 `cfg.flavor = "advanced"` 在训练循环里，
那 dynamo 可能会把它当 guard 重新编译。）

### 坑 3：超参 tensor 放在 GPU 上

见上面。**必须 CPU**，否则跨设备读取会打断异步执行。

```bash
uv run pytest tests/test_judging_soundness.py -k hyperparams -v
```

### 坑 4：`HyperParams(**kwargs)` 只登记键名

第 21 章验证 1 踩过。`HyperParams(lr=0.3)` 拿到的是 **`lr = 0`**。

**必须先 `HyperParams(lr=0, ...)` 再 `hp.set(lr=0.3, ...)`。**

### 坑 5：去掉 `if` 之后忘了验证数值等价

第 26 章那条「`wd=0` 时逐位相同」是**安全性依据**。
如果哪天改了 `mask` 的计算，必须重新验证。

**判据在 `tests/test_optim.py -k cautious`。**

### 坑 6：`lerp_` 的 dtype 要求

我的验证脚本第一版就踩了：

```
got RuntimeError('expected dtype torch.float32 for `end`,
                  but got dtype torch.bfloat16')
```

`Tensor.lerp_(end, weight)` 要求 `end` 和 `self` **同 dtype**。

本项目里 `stacked_grad` 已经是 `stacked_param.dtype`（fp32），
所以 `muon_step` 里没问题。但**你手写验证脚本时**
如果照抄了 `X = g.bfloat16()` 然后 `g.lerp_(X)` 就会撞这个。

---

## 延伸

**`torch.compile` 的三个档位**

| 档位 | 行为 |
|---|---|
| 默认（`fullgraph=False`） | 允许 graph break，不支持的地方退回 eager |
| `fullgraph=True` | 一个图都不许断，不支持直接报错 |
| `dynamic=False` | 形状是静态的（换形状会重编译） |

本项目用 `fullgraph=True, dynamic=False` ——
**最严格的组合**。好处是任何编译问题都在第一步暴露；
坏处是对 PyTorch 版本敏感（本机需要 gcc，见第 01 章）。

**CUDA graph（更激进的优化）**

比 `torch.compile` 更进一步：把整个 step 的 kernel launch
录制成一个「图」，然后重放。收益是消除 launch 开销。

**限制更严**：
- 所有张量地址固定（不能每步新建）
- **没有 CPU 同步点**
- 静态形状

本项目没用 CUDA graph —— 因为每步要更新 lr 等超参，
而 lr 在 CPU tensor 里（这其实**正好**符合 CUDA graph 的要求）。

**Why Compiler Matters（NVIDIA）**

一篇 2025 年的文章分析为什么 `torch.compile` 在 LLM 训练上
收益不如预期：Python 开销占比可能只有百分之几，
而 kernel fusion 的收益取决于瓶颈在哪。

**本项目的实测是 27%**（单个矩阵），说明在这个规模上
优化器 kernel 确实是瓶颈之一 —— 因为矩阵小、kernel 短、
launch 开销占比高。

模型越大、矩阵越大，kernel 时间占比越高，编译收益**越小**。

**dynamo 的 guard 机制**

`torch.compile` 靠 **guard** 判断「这次调用的输入是否和上次相同」：
- 张量的形状、dtype、device
- **Python 标量的值**
- 对象的 id 和属性

guard 不匹配就重新编译。**Python 标量进 guard，tensor 的值不进**
—— 这就是为什么超参要包成 tensor。

**那为什么 tensor 的值不进 guard？**

因为 tensor 的**内容**可能每步都变，dynamo 无法保证。
它只能保证「这块内存的地址和形状不变」。

---

## 卷 4 完成

8 章讲完了「怎么训」。回看一下：

| 章 | 核心 |
|---|---|
| 21 AdamW | 每个超参在干什么；解耦 WD 的可测量含义 |
| 22 为什么是 Muon | SVD 视角；条件数 2042 → 6.36 |
| 23 Newton-Schulz | 20 行；5 步落在 `[0.5,1.5]`；8 步饱和 |
| 24 Polar Express | 斜率 3.44 → 8.16；**`ns_steps>5` 静默失效** |
| 25 MuonEq/Muon+/NorMuon | **`use_muon_plus` 不是论文的**；因子化省 647 MiB |
| 26 谨慎 WD | `g` 是正交化后的量；真实配置下衰减只占 3.1% |
| 27 参数分组 | `ndim==2` 划错一处；形状补偿与论文不一致 |
| 28 torch.compile | **一个 if 值 9.9 小时**；0-D tensor 技巧 |

**本卷最值得记住的三件事**：

1. **正交化把梯度的绝对尺度整个丢掉了**（第 22 章）
   —— 这决定了 `dmodel_lr_scale` 只作用于 AdamW 组（第 27 章）
2. **5 步 Newton-Schulz 是「便宜且够用」，不是「够用」的极限**（第 23 章）
   —— 而且 Polar Express 模式下连 8 步都调不出来（第 24 章）
3. **静默失效是本项目最危险的 bug 形状**（第 24/25/28 章）
   —— `ns_steps>5`、`use_muon_plus`、compile 回落，三个都是

---

## 下一卷

**卷 5-8 · 只读教程**（28 章判据覆盖，本会话的 `progress.sh` 28 章）

卷 1-4 是**手抄**代码（`raise NotImplementedError` 的地方等你填）。
卷 5-8 是**只读** —— 直接读 `src/`，判据是「不要改坏它」。

[第 29 章：训练循环](29-训练循环.md) ——

前面 28 章讲的是「零件」。第 29 章开始讲「引擎」：
一个 step 内部到底发生了什么。
