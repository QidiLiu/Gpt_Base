# 07 · 先跑通一个最小 GPT

**卷 2 的开篇。** 前面 6 章把文本变成了 `(inputs, targets)`。
这一章开始，另一条线：把 `(inputs, targets)` 变成 logits。

---

## 本章目标

- 看清一个 GPT 的**完整数据流**：`(B,T) int64` 进去，`(B,T,V)` logits 出来
- 知道 `GPT` 这个类由哪些零件组成，以及它们之间的形状怎么变
- 亲手跑通一个最小模型，亲眼看到第一个 loss 是 `ln(V)`

**本章不写新代码**，但它是后面 8 章的地图 —— 每一章都会放大这张图的一块。

---

## 前置回顾

[第 06 章](06-bits-per-byte.md) 我们算出了一个 loss：预测下一个 token 的
交叉熵，单位是 nats/token。

但那个 loss 是「多少算好」的问题，**不是**「怎么算出来」的问题。
从这一章开始解决后者。

---

## 概念：一次前向的完整形状变化

假设 `debug` 档（d2，32 维，1 头，V=8192），一个 batch：

```
inputs   (8, 128)      int64，值域 [0, 8192)
   │
   ├─ wte 查表 ──────────────► x0  (8, 128, 32)    bf16
   │
   ├─ rms_norm（嵌入后立刻做）─►      (8, 128, 32)
   │
   ├─ ┌─ Block 0 ──────────────┐
   │  ├─ rms_norm  ──────────►  (8, 128, 32)
   │  ├─ Attention ──────────►  (8, 128, 32)      ← 卷2 第 9-12 章
   │  ├─ + 残差 ─────────────►  (8, 128, 32)
   │  ├─ rms_norm  ──────────►  (8, 128, 32)
   │  ├─ MLP (32 → 128 → 32) ►  (8, 128, 32)      ← 第 13 章
   │  └─ + 残差 ─────────────►  (8, 128, 32)
   ├─ └───────────────────────┘
   ├─ ┌─ Block 1 ──────────────┐  （完全相同的结构）
   │  └─ ...                   └─► (8, 128, 32)
   │
   ├─ rms_norm（最后这次很关键）►      (8, 128, 32)      ← 第 14 章
   │
   ├─ lm_head (32 → 8192) ─────►  (8, 128, 8192)  fp32   ← 第 15 章
   │
   └─ 与 targets 算 CE ────────►  标量 loss
```

**注意三个「立刻归一化」的位置**：嵌入之后、每个 Block 的两个子层之前、
以及所有 Block 之后。它们不是可选的 —— 残差流的尺度必须被钉住，
否则深层的 logits 会炸开。这是 Pre-LN 架构的核心（第 14 章）。

---

## 概念：为什么第一个 loss 必须等于 `ln(V)`

一个刚初始化的模型，它的 `lm_head` 权重是

```python
torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
```

`std=0.001` 意味着所有 logits 都挤在 0 附近。经过 softmax 后
**每个 token 的概率几乎相同**，也就是「完全不确定」。而均匀分布的
交叉熵正是：

```
loss = -Σ p_i · ln(p_i) = -ln(1/V) = ln(V)
```

V=8192 时 `ln(8192) = 9.0109`。

**这是你第一个检查点。** 跑起来看到 `loss 9.0109` 说明：
- 形状对（没有意外广播）
- 初始化对（lm_head 没有被 PyTorch 默认初始化覆盖）
- tokenizer 对（V 就是词表大小）

数字不对，先查这三样，别急着调超参。

> 顺带一提：nanochat 和 modded-nanogpt 都用 `std=0.001` 而不是常见的
> `std=0.02`。原因是「几乎均匀」是一个更好的起点 —— 相当于让模型
> 从「什么都不知道」开始，而不是从「某个随机的错误答案」开始。

---

## 概念：`GPT` 类的零件清单

`src/model/gpt.py` 里只有 6 个方法，但它们就是全部：

| 方法 | 卷 | 手抄难度 |
|---|---|---|
| `__init__` | 卷3（第 16-19 章） | 中：要按 `meta device` 的规则写 |
| `init_weights` | 第 16 章 | 中：六步初始化策略 |
| `forward` | 卷2-3 | **高**：组装全部零件 |
| `_apply_smear` | 第 19 章 | 低 |
| `describe` | 卷0 | 只读：打印结构 |
| `build_model`（模块级） | 第 16 章 | **低但关键**：meta device 三步法 |

而零件本身在 `src/model/layers.py`：

| 零件 | 卷 | 章 |
|---|---|---|
| `Linear` | 卷0 | 精度控制（本章只读） |
| `rms_norm` / `layer_norm` / `make_norm` | 卷2 | 08 |
| `precompute_rope` / `apply_rotary_emb` | 卷2 | 09 |
| `CausalSelfAttention` | 卷2 | 10, 11, 12 |
| `MLP` | 卷2 | 13 |
| `Block` | 卷2 | 14 |
| `sdpa_bt` / `attend` | 卷2 | 10, 11 |

**卷2 的 8 章就是把这张表从上往下走一遍。**

---

## 手抄代码

本章没有新代码。但如果你想现在就看到完整结构，可以在 `src/model/gpt.py`
里只读地走一遍 `forward`（第 202-276 行），把上面那张形状图和真实代码
对上。卷3 会逐段拆它。

### 第 1 块：跑一个最小模型（只读 + 跑）

```python
# scratch 目录里没有现成脚本，这一章用 debug 档 20 步就够
import sys
import torch

sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import build_model, sample_from_logits

cfg = make_run_config("debug")           # d2 / 32 维 / V=8192
print(cfg.model.n_layer, cfg.model.n_embd, cfg.model.vocab_size)

model = build_model(cfg.model, device="cuda")
print(model.describe())                   # 结构 + 参数量明细 + FLOPs

B, T = 2, 16
x = torch.randint(0, cfg.model.vocab_size, (B, T), device="cuda")
y = torch.randint(0, cfg.model.vocab_size, (B, T), device="cuda")

logits = model(x)                        # 没有 targets -> 返回 logits
print("logits", tuple(logits.shape))     # (2, 16, 8192)

loss = model(x, y)                       # 有 targets -> 返回标量
print("loss", loss.item())               # 应该 ≈ 9.0109
print("ln(V)", torch.log(torch.tensor(float(cfg.model.vocab_size))).item())
```

---

## 动手验证

### 验证 1：确认第一个 loss 等于 `ln(V)`

```bash
uv run python -m training.train_base --mode debug
```

看第一行日志：

```
step     0/20 (  0.0%) | loss  9.0109 | ...
```

**必须对上 4 位小数。** 如果是 0 或者 8.x，说明 `lm_head` 的初始化被覆盖了
—— 去 `init_weights` 里查 `torch.nn.init.normal_(self.lm_head.weight, ...)`
那行还在不在。

### 验证 2：V 变了，ln(V) 也要跟着变

```bash
uv run python -m training.train_base --mode smoke      # V=8192 -> 9.0109
```

`smoke` 和 `debug` 的词表都是 8192，所以两个都是 9.0109。
想看到差异，用 `ablation` 档（**本意** V=16384，`ln = 9.7041`）：

```bash
uv run python -m training.train_base --mode ablation --num-iterations 1
```

> ⚠ **2026-10 曾一度出现异常：跑出来的是 9.0108（= ln(8192)）而不是 9.7034。**
>
> 原因：`train_tokenizer` 只判断 `tokenizer.pkl` 存不存在，**不检查词表大小**。
> 缓存里那个是某次 `smoke` 建的 8192 词表，所以 `ablation` 传进去的
> `--vocab-size 16384` 被静默忽略了。
>
> **已修（2026-10 同日）**，`train_tokenizer` 现在复用前核对词表大小。
> 重建之后本节重新成立 —— 实测确认 step 0 loss = **9.7043**
> （`ln(16384) = 9.7041`，差 0.0002）。
>
> 根因、修法和故障注入记录见[第 37 章](37-全流程与排错.md)。
>
> **自查一行**：`... 2>&1 | grep 词表`

> 实测（`ablation` 档，CPU 上随机权重，V=16384 时）初始化 loss 是 **9.7034**，
> 理论值 `ln(16384) = 9.7041`，差 0.0007 —— bf16 舍入的量级。
> **不要期待四位小数完全对上**，差在 0.01 以内就说明初始化是对的。
> （`debug` 档同理：实测 9.0114 vs `ln(8192) = 9.0109`。）

### 验证 3：确认形状真的按图走

```bash
uv run python -m scratch/hardware.py
```

它会打印本机的关键事实和 SDPA 布局陷阱的最小复现。
第 11 章会详细讲那个布局陷阱。

---

## ★ 消融实验

本章没有可消融的东西（还没有任何开关）。但有一个**值得做的对照**，
它能帮你建立「架构参数对 loss 的影响有多大」的直觉：

```bash
# 同一个 debug 档，只改宽度（唯一旋钮 depth 不动，用 overrides）
uv run python -m training.train_base --mode debug --depth 4 --no-resume --model-tag dbg_d4
uv run python -m training.train_base --mode debug --depth 2 --no-resume --model-tag dbg_d2
```

两个都是 20 步、初始化 loss 都是 9.0109（因为 V 相同），
但**第 10 步的 loss 会不同** —— 宽模型在相同步数下下降更快。

这个观察的价值：它说明 20 步的 `debug` 档**只能验证「跑通了」，
不能验证「架构差异」**。第 00 章说过「debug 档测不出架构差异」，
这就是原因。

---

## 常见坑

### 坑 1：`loss.item()` 在第一个 step 报 NaN

几乎总是因为**目标里全是 `-1`**（`ignore_index`）。`GPT.forward` 里有一段
兜底会把 NaN 换成 0 并打警告，但如果你的 `targets` 构造逻辑有问题，
那层兜底只会掩盖症状。

用这个定位：

```python
print("targets 有效位置数:", (y >= 0).sum().item(), "/", y.numel())
```

### 坑 2：logits 的形状是 `(B, V)` 而不是 `(B, T, V)`

`forward` 有两条路径：

```python
model(x, y)     # 有 targets -> 返回标量 loss
model(x)        # 没有 targets -> 返回 (B, T, V) logits
```

想要 logits 又想算 loss，只能分两次调用。

### 坑 3：`describe()` 里的「FLOPs / token」比你以为的大

它是 `6 × scaling_params + 注意力项`，其中 `scaling_params` **不含 wte
和 value_embeds**（查表没有矩阵乘的 FLOPs）。所以这个数字比
「参数量 × 6」小。看到差异不用慌，口径不同。

---

## 延伸

nanochat 的 `gpt.py` 大约 600 行，本项目的 `gpt.py` 是 403 行。
差的 200 行主要是 nanochat 支持的那些本项目没有的东西：

- **fp8 量化**（第 12 章 speedrun 靠它快 4%）
- **多卡 ZeRO-2 的参数收集**（本项目在 `muon.py` 里，不在 gpt.py）
- **编译友好的分支**（nanochat 大量用 `torch.compile` 包整个 block）

本项目的取舍是「**每加一个开关，就要在教程里有一章解释它**」。
所以卷2-3 只保留那些「不解释就会用错」的开关。

---

## 下一章

[第 08 章：RMSNorm](08-RMSNorm.md) ——
第一个零件。为什么可以砍掉减均值和偏置。