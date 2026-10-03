# 06 · bits per byte

**卷 1 的收尾。** 这一章给你一把尺子 —— 没有它，你做不了任何消融实验。

---

## 本章目标

- 说清楚为什么 **loss 不能跨模型比较**，而 bpb 可以
- 亲手实现 bpb 的计算，并理解那个「每个 token 多少字节」的表是怎么来的
- 知道 bpb 的参考量级，以及为什么你的模型数字会和论文差很远

---

## 前置回顾

[第 05 章](05-文档打包BOS-aligned-bestfit.md) 完成了数据管线的最后一步。
现在我们有了 `(inputs, targets)` 二元组，可以算 loss 了。

但立刻会遇到一个问题：**loss 到底是多少算「好」？**

---

## 概念：loss 的三个不可比性

假设两个模型的 val loss 分别是 5.2 和 5.0。哪个更好？

**答案：无法判断。** 原因有三个，任何一个都足以让比较失效。

### 不可比性 1：词表大小不同

```
模型 A：词表 8192，一个 token 平均覆盖 3.1 字节
模型 B：词表 32768，一个 token 平均覆盖 3.7 字节
```

loss 的单位是「每个 token 的 nats」。但一个 32768 词表的 token
**携带更多信息**（它区分得更细），所以它每个 token 的 loss 天然更低。

极端例子：字符级 tokenizer（词表 65）的 loss 会在 4-5 nats 左右
（因为每个 token 太简单了），而 BPE tokenizer（词表 32768）的 loss
会在 2-3 左右。**但字符级模型的实际能力差得多**。

### 不可比性 2：数据集不同

模型 A 在 OpenWebText 上评，模型 B 在 ClimbMix 上评。
数据难度不同，loss 不可比。

### 不可比性 3：序列长度不同

长序列的 loss 通常更低（更多上下文）。T=1024 的 loss 和 T=4096 的 loss
不是一个量纲。

---

## 概念：bpb 怎么解决前两个

**bits per byte** = 「预测一个原始字节平均消耗多少比特」。

```
loss 的单位是 nat（自然对数）
1 nat = log2(e) bits ≈ 1.4427 bits

bpb = (每个 token 的 nats) / ln(2) / (每个 token 覆盖的字节数)
```

第二项「每个 token 覆盖的字节数」就是关键。它把度量单位从
「token」换到了「字节」—— 字节是**与分词方式无关**的绝对单位。

更简洁的推导（用累加量，不需要中间量）：
```
bpb = total_nats / ln(2) / total_bytes
```
只需要累加「总 nats」和「总字节数」两个数。

代码位置：`src/evaluation/metrics.py: evaluate_bpb`

```python
@torch.no_grad()
def evaluate_bpb(model, loader, token_bytes, max_batches):
    total_nats, total_bytes = 0.0, 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches: break
        x, y = batch[0], batch[1]
        loss_vec = model(x, y, loss_reduction="none")     # (B, T) 逐 token
        valid = (y >= 0)                                   # -1 是不计 loss 的
        total_nats += loss_vec[valid].sum().item()
        total_bytes += token_bytes[y[valid]].sum().item() # ← 查表
    return total_nats / (math.log(2) * max(total_bytes, 1e-9))
```

**注意 `token_bytes[y[valid]]` 这一行** —— 它查的是 **target** 的字节数，
不是 input 的。这是有意的：模型的工作是「预测这个 token」，
所以应该按「被预测的东西有多长」来计费。

---

## 概念：`token_bytes` 这张表从哪来

对词表里每个 token id，记下它对应多少字节：

```python
def compute_token_bytes(tok, device="cpu"):
    n = tok.get_vocab_size()
    arr = torch.zeros(n, dtype=torch.float32, device=device)
    for tid in range(n):
        arr[tid] = len(tok.decode_bytes(tid))
    return arr
```

代码位置：`src/training/train_tokenizer.py: compute_token_bytes`

**一个必须注意的点**：这张表要在 **tokenizer 定下来之后**算，
用**同一个** tokenizer 对象。算完存成 `token_bytes.pt`。

| token | 字节数 |
|---|---|
| ` the` | 4 |
| `ing` | 3 |
| `2026` | 4 |
| ` ` | 1 |
| `<\|bos\|>` | 8（`<|bos|>` 是 7 个字符） |

**注意 token 的字节数不固定**：高频多字符 token 覆盖的字节多，
单字符 token 只有 1 个字节。所以必须逐个查表，不能用一个平均值。

---

## 概念：bpb 的参考量级

| 文本类型 | bpb |
|---|---|
| 随机字节 | ~8.0（每个字节 8 bit，理论上限） |
| 随机单词（打乱的英文） | ~4-5 |
| 未经训练的小模型 | ~2.5-4 |
| **训练良好的 nanochat d24** | **0.68** |
| 训练良好的 GPT-2 级别 | ~0.9-1.0 |
| 英语自然文本（信息论下界估计） | ~0.8-1.0 |

**本项目实测**（`smoke` 档 d4，896 步，14M token）：
- 随机初始化：2.349
- 训完：1.698

### ✅ 本项目的第一个真实 `val_bpb`（2026-10 实测）

`ablation` 档 d6 完整训完 3,096 步（2.03 亿 token，V=16384，38m31s）：

```
val_bpb: 总 nats=3,482,533 / 总字节=4,671,803 -> 1.0754   （全验证集）
训练循环里的周期性估计（best_val_bpb）      -> 1.0615
```

**外推说的是 1.3-1.5，实测 1.0615 —— 比外推还低。**

⚠ **两个 1.07x 不是同一个数**：`eval_base.sh` 跑全验证集（1.0754），
训练循环里的周期性评估只跑 `max_batches` 个 batch（1.0615）。
**引用时要说明是哪一个**，差 0.014。

### ★★ 更正：本章早期版本把「均匀基线」算错了量纲

⚠ **原来这里写的是「均匀分布基线 `ln(16384)/8 = 1.2130`」—— 那是错的。**

`bpb` 的定义是 **bits per byte**，而 `ln(V)` 是 **nats per token**。
从 nats 换到 bits 必须除以 `ln(2)`：

```
1 nat = log2(e) = 1/ln(2) ≈ 1.4427 bits        ← 少了这一步，量纲就不对了
```

所以：

| | 公式 | 值 | 性质 |
|---|---|---|---|
| **文档旧值** | `ln(16384)/8` | **1.2130** | ❌ **nats/byte，不是 bits/byte** |
| 正确（8 B/token 理想化） | `log2(16384)/8` | **1.7500** | ✅ bits/byte |
| 正确（实测 4.4492 B/token） | `log2(16384)/4.4492` | **3.1466** | ✅ bits/byte |

**差了 1.4427 倍（正好是 `1/ln(2)`）。**

> **讽刺的是，本章后面第 391 行用的就是正确形式**
> （`log2(8192) / 5.535 = 2.3487`）—— 同一章里两种口径并存，
> 我一直没发现。**而我在 ch37 和 README 里又把这个错值抄了一遍。**

### ★ 两个基线必须分清（这是最容易混的地方）

| 基线 | 含义 | 值 |
|---|---|---|
| **随机字节** | 均匀 over 256 个字节值，每字节 8 bit 熵 | **8.0 bpb** |
| **均匀 over 词表** | 模型对 V 个 token 完全无差别 | `log2(V) / bytes_per_token` |

`src/evaluation/metrics.py` 的 docstring 引用的是**第一个**（8 bpb）。
而本章上面比的是**第二个**。

**本项目 V=16384、实测 4.4492 bytes/token：**

| 基线 | 值 | 实测 1.0615 相对它 |
|---|---|---|
| 随机字节 | 8.0000 | 低 **86.7%** |
| 均匀 over 词表（实测 B/tok） | **3.1466** | 低 **66.3%** |
| 均匀 over 词表（8 B 理想化） | 1.7500 | 低 **39.3%** |
| ~~文档旧值~~ | ~~1.2130~~ | ~~「低 12.5%」~~ ❌ |

**所以「模型学到了多少」的正确说法是：比完全无差别的选择好 66%。**
而不是旧值给的「12.5%」—— 那个数把模型的能力夸大了。

### 训练轨迹

| step | `val_bpb` | train loss | `lr_mult` |
|---|---|---|---|
| 600 | 1.2550 | — | 1.000 |
| 1000 | 1.1850 | 3.7701 | 1.000 |
| 2000 | 1.1013 | 3.4871 | 0.567 |
| 2600 | 1.0744 | — | 0.230 |
| 3000 | 1.0630 | 3.3685 | 0.095 |
| 3096 | **1.0615** | 3.3667 | 0.050 |

**step 1000（1.1850）就已经低于「8 B/token 理想化」的均匀基线 1.7500**，
更不用说实测基线 3.1466 —— 实际上它在**第一个验证点之前**就越过了。

**而且结束时仍在下降** —— 最后 96 步还有 `-0.0015`，
所以 3,096 步**没有跑够**。想更低要加步数或换更大的档位。

### 顺带测到的三件事

**① 训练循环的 EMA 与周期性评估给不出精确值**

```
step     0 | loss 9.7044    <- = ln(16384) = 9.7041，初始化正确（bf16 舍入量级）
step    39 | loss 6.5320    <- warmup 刚结束
step  3096 | loss 3.3667
```

**② 内存只有 5,692 MiB，MFU 只有 13.74%**（vs `full` 档的 11,465 MiB / 26.56%）

**d6 的 GEMM 太小，喂不饱这张卡** —— 这就是为什么 `full` 档
才是性能测试的对象，`ablation` 档只适合做消融。

**③ 多选题：MMLU 27.0% / ARC-Easy 24.0% / ARC-Challenge 22.7%**

MMLU 4 选 1 的随机基线是 25% —— 27.0% **只比随机高 2 个百分点**。
ARC 两个都 ≈ 随机。

**这不是 bug**：基座模型没做过 SFT，它不知道「要回答一个字母」，
而 `score_choices` 比的是「选项文本的平均 logprob」。
**23M 参数 / 2 亿 token 的模型没有选择题推理能力是预期的。**
（`eval_sft.sh` 才是那个指标的用武之地。）

**样本输出**是连贯英文但有明显重复退化：

```
'The capital of France is'
    -> ' the largest city in the world. It is the largest city in the world. It is ...'
```

这是小基座模型的典型表现（温度 0.7、无 repetition penalty）。
**能生成语法正确的英文 = 训练管线是通的**，这正是这一步要验证的。

> `full` 档 d24（21.9 亿 token）**仍然是外推**，仓库里没有它的存档。
> 按实测的 `ablation` = 1.06 和缩放关系，它大概会到 **0.9-1.1** ——
> 比原来外推的 1.0-1.2 更乐观一点，但**这仍然是推测**。
>
> 这也是为什么 bpb 的价值在**差值**而不在绝对值。
>
> ★ **2026-10 实测更正**：`ablation` 档的噪声底是 **0.000134**
> （3 次同配置 + 1 次换初始化），不是早期文档里写的 ±0.02 ——
> 那个数从来没被测过。真实效应在 0.001 ~ 0.05，所以消融必须用
> `ablation` 档（3096 步）而不是 `smoke` 档（896 步）：
> **`smoke` 档不是「效应被噪声淹没」，而是会把符号测反**
> （QK-Norm 消融在两档给出相反的符号）。见第 00 章。

**但这不影响本章的价值**：消融实验看的是**差值**，
而差值恰好在 0.01-0.05 量级 —— 比绝对值有意义得多。

---

## 手抄代码

### 第 1 块：算 `token_bytes`

```python
def compute_token_bytes(tok, device="cpu") -> torch.Tensor:
    """
    对每个 token id 记录它对应多少字节。

    用途：bpb（bits per byte）。换算公式：
        bpb = (每 token 的 nats) / ln(2) / (该 token 平均覆盖的字节数)
    所以需要这张表。

    ★ 必须在 tokenizer 最终定下来之后、用同一个对象来算。
    """
    n = tok.get_vocab_size()
    arr = torch.zeros(n, dtype=torch.float32, device=device)
    for tid in range(n):
        try:
            arr[tid] = len(tok.decode_bytes(tid))
        except Exception:
            arr[tid] = 0.0
    return arr
```

### 第 2 块：算 bpb

```python
@torch.no_grad()
def evaluate_bpb(model, loader, token_bytes, max_batches: int) -> float:
    """
    在验证集上算 bpb。

    ── 为什么不直接用 loss？─────────────────────────────
    loss 的单位是「每个 token 的 nats」。但不同模型的词表大小不同，
    一个 token 覆盖的字节数也不同，所以 loss **不可比**。
    8192 词表的模型和 32768 词表的模型，同样的 loss 意味着完全不同的
    建模能力。

    bpb 换算到「原始字节」这个与分词方式无关的单位，就可以公平比较：

        bpb = (每 token 的 nats) / ln(2) / (每 token 的字节数)
            = total_nats / ln(2) / total_bytes

    最后这个形式最干净：只需要累加「总 nats」和「总字节数」两个数。

    ★ 一定要还原 train 模式：train_base.py 会在训练中途调用本函数，
      模型被留在 eval 模式就会一直 eval 下去 —— 现在没有 dropout 看不出
      问题，但哪天加了 dropout 就会静默训坏。
    """
    # ★ 顺序：先记住原状态，再切 eval()。反过来写的话 eval() 已经把
    #   training 置成 False，was_training 恒为 False，下面的还原成了死代码。
    was_training = model.training
    model.eval()
    total_nats, total_bytes = 0.0, 0.0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, y = batch[0], batch[1]
        x, y = x.to(model.get_device()), y.to(model.get_device())
        # 逐 token 的 loss。reduction='none' 才能拿到每个位置的值
        loss_vec = model(x, y, loss_reduction="none")      # (B, T)
        valid = (y >= 0)                                    # -1 是不计 loss 的
        total_nats += loss_vec[valid].sum().item()
        # ★ 查 target 的字节数，不是 input 的：
        #   模型的工作是「预测这个 token」，按被预测物计费
        total_bytes += token_bytes[y[valid]].sum().item()

    if was_training:
        model.train()
    return total_nats / (math.log(2) * max(total_bytes, 1e-9))
```

---

## 动手验证

### 验证 1：亲手算一遍 bpb

新建 `scratch/bpb_demo.py`：

```python
"""
bpb 的完整手工验证。第 06 章验证 1。
"""
import math
import torch
from data.tokenizer import get_tokenizer, get_token_bytes
from data.dataloader import make_dataloader
from common import load_latest, get_runs_dir

tok = get_tokenizer()
tb = get_token_bytes()

# ── 1. 看看 token_bytes 这张表长什么样 ──
print("=== token_bytes 表（前 12 个普通 token）===")
n_normal = tb.numel() - 9
sizes = {}
for tid in range(n_normal):
    sizes.setdefault(int(tb[tid]), 0)
    sizes[int(tb[tid])] += 1
print(f"  普通 token {n_normal} 个，按字节数分布：")
for k in sorted(sizes):
    print(f"    {k} 字节: {sizes[k]:5d} 个 token ({sizes[k]/n_normal*100:5.1f}%)")
mean_bytes = tb[:n_normal].sum().item() / n_normal
print(f"  平均每 token {mean_bytes:.3f} 字节")
print("  ★ 这个均值和 tokenizer 报的 bytes/token 是同一个数，可以互相验证")

# ── 2. 最优 bpb 的理论值：均匀分布 ──
V = tok.get_vocab_size()
print(f"\n=== 理论参考：均匀分布预测所有 token ===")
print(f"  如果模型对 {V} 个 token 完全不确定，它会给每个 {math.log(V):.4f} nats")
print(f"  换算成 bpb = {math.log(V)/math.log(2)/mean_bytes:.4f}")
print(f"  （这就是随机初始化时的 bpb，也等于 loss / ln2 / avg_bytes）")

# ── 3. 用训练好的模型算真实 bpb ──
model, _, meta = load_latest(get_runs_dir(), "d4", "cuda")
model.eval()
dl = make_dataloader(tok, 8, model.config.sequence_len, "val", "cuda")

with torch.no_grad():
    total_nats, total_bytes, total_tokens = 0.0, 0.0, 0
    for i, batch in enumerate(dl):
        if i >= 8:
            break
        x, y = batch[0].to("cuda"), batch[1].to("cuda")
        loss_vec = model(x, y, loss_reduction="none")
        valid = y >= 0
        total_nats += loss_vec[valid].sum().item()
        total_bytes += tb.to("cuda")[y[valid]].sum().item()
        total_tokens += int(valid.sum())
    bpb = total_nats / (math.log(2) * total_bytes)
    avg_loss = total_nats / total_tokens

print(f"\n=== 训练好的 d4 模型（step {meta.get('step')}）===")
print(f"  平均 loss      = {avg_loss:.4f} nats/token")
print(f"  平均每 token   = {total_bytes/total_tokens:.3f} 字节")
print(f"  bpb            = {bpb:.4f}")
print(f"  验证：loss/ln2/bytes = {avg_loss/math.log(2)/(total_bytes/total_tokens):.4f}  [应等于 bpb]")
print(f"  训练日志里的 bpb   = {meta.get('best_val_bpb'):.4f}")
```

跑：

```bash
bash script/train_base.sh smoke     # ★ 必须是 smoke 档，见下
uv run python scratch/bpb_demo.py
```

⚠ **必须先有 `smoke` 档（tag `d4`）的 checkpoint。**

`bpb_demo.py` 第 30 行硬编码了：

```python
model, _, meta = load_latest(get_runs_dir(), default_tag("smoke"), "cuda")
```

**只跑过 `ablation`（或 `full`）的话这里会抛**
`AssertionError: runs/base_checkpoints/d4 里没有存档`。

> 这个脚本没有 `--base-tag` 参数，所以**跑别的档位必须先补跑 smoke**
> （几分钟）。或者改脚本第 30 行接受一个 tag —— 但那属于改代码。

**想直接用任意档位**，用 `eval_base.sh` 更省事（它接受档位参数）。

**三个观察点**：

**① `token_bytes` 的分布很不均匀。**
你会看到大量 1 字节的 token（单个空格、单个标点）和若干 5-10 字节的
长 token。这正是为什么不能用一个「平均字节数」近似 ——
必须逐个查表。

**② 平均值能互相验证。**
tokenizer 训练时报的 `bytes/token`（第 02 章的 3.24）和这里算出的
`mean_bytes` 应该是同一个数。如果不一致，说明 tokenizer 换过了
但 `token_bytes.pt` 没重算。

**③ bpb 远小于均匀分布的参考值。**

```
=== 理论参考：均匀分布（随机初始化）===
  对 8192 个 token 完全不确定 -> 每 token 9.0109 nats
  换算 bpb = log2(8192) / 5.535 = 2.3487

=== 训练好的 d4 模型（step 896）===
  平均 loss            = 4.7264 nats/token
  平均每 token 字节数  = 4.017
  bpb                  = 1.6976
  验证 loss/ln2/bytes  = 1.6976  [应等于 bpb]
  训练日志记录的 bpb   = 1.7136

  学到的知识量 = 2.349 - 1.698 = 0.651 bpb 的下降
```

随机初始化时 bpb = 2.349（= `log2(8192)/平均字节数`），
训好的模型是 1.698。**这个 0.65 的下降就是模型「学到的知识」的量化。**

> **这一段用的是正确公式** —— `log2(V) / 平均字节数`，
> 正是本章前面那个「更正」之后确立的口径。
>
> 之所以看起来和前面不一致，是因为**参数不同**：
>
> | | V | 平均字节数 | 均匀基线 bpb |
> |---|---|---|---|
> | 本段（`smoke` 档 d4）| 8192 | 5.535 | **2.3487** |
> | 前面的实测（`ablation` 档 d6）| 16384 | 4.4492 | **3.1466** |
>
> 词表大一倍 → `log2(V)` 多 1 bit；平均字节数不同 → 分母不同。
> **两者都对，但** `d6` 的 3.1466 反而更高 —— 因为它的词表更大，
> 「完全没学到东西」时的无差别猜测空间更大。

**注意平均字节数是 5.535 而不是第 02 章报的 3.24** ——
第 02 章那个数字是在一段 162 字节的英文样句上测的，不具代表性。
**语料上的真实均值要靠 `token_bytes` 表算。** 这正好说明了为什么要逐个查表。

### 验证 2：亲手制造一次「loss 相同但 bpb 不同」

新建 `scratch/bpb_uncomparable.py`：

```python
"""
证明 loss 不可比、bpb 可比。第 06 章验证 2。
"""
import math
import torch
import torch.nn.functional as F
from data.tokenizer import get_tokenizer, get_token_bytes

tok = get_tokenizer()
tb = get_token_bytes()
V = tok.get_vocab_size()

# 假设两个模型都恰好预测出 5.0 nats/token
LOSS = 5.0
print(f"两个模型的 loss 都是 {LOSS} nats/token\n")

# 模型 A：用 8 位字符做词表（每个 token 只覆盖 1 字节）
#          -> 等等，我们只有一个 tokenizer。改用「假设」来演示。
scenarios = [
    ("字符级（词表 256，每 token 1 字节）", 1.0),
    ("本项目 BPE（词表 8192，实测 5.5 字节）", 5.5),
    ("GPT-2 BPE（词表 50257，约 4.1 字节）", 4.1),
    ("超大词表（词表 256000，约 6.5 字节）", 6.5),
]
print(f"{'场景':<42} {'bytes/token':>11} {'bpb':>8}")
print("-" * 65)
for name, bpt in scenarios:
    bpb = LOSS / math.log(2) / bpt
    print(f"{name:<42} {bpt:11.2f} {bpb:8.3f}")

print("\n→ 同样的 loss，bpb 相差 5 倍以上。所以 loss 不能跨词表比较。")
print("→ 而 bpb 衡量的是「预测一个字节要多少 bit」，与词表无关。")
print()
print("=== 反过来：bpb 相同但 loss 不同 ===")
BPP = 1.0
print(f"{'场景':<42} {'bytes/token':>11} {'loss(nats)':>10}")
print("-" * 65)
for name, bpt in scenarios:
    loss = BPP * bpt * math.log(2)
    print(f"{name:<42} {bpt:11.2f} {loss:10.3f}")
print("\n→ 同样的真实能力，loss 相差 5 倍。bpb 才是那个不变量。")
```

跑：`uv run python scratch/bpb_uncomparable.py`

**这个实验的价值**：它用一张表把「为什么需要 bpb」讲得无需质疑。

---

## ★ 消融实验：bpb 是你的主尺子

**从本章开始，所有消融实验都用 bpb 报告结果。** 流程固定为：

```bash
# 1) 基线
bash script/train_base.sh ablation --model-tag d6_base --no-resume

# 2) 消融（一次只改一个变量）
bash script/train_base.sh ablation --model-tag d6_norope     --no-rope         --no-resume
bash script/train_base.sh ablation --model-tag d6_noqk       --no-qk-norm      --no-resume
bash script/train_base.sh ablation --model-tag d6_gelu       --activation gelu --no-resume
bash script/train_base.sh ablation --model-tag d6_tied       --tie-embeddings  --no-resume
bash script/train_base.sh ablation --model-tag d6_nosoftcap  --no-softcap      --no-resume
bash script/train_base.sh ablation --model-tag d6_muonadv    --muon-advanced   --no-resume
bash script/train_base.sh ablation --model-tag d6_alltricks  --all-tricks      --no-resume
```

> ⚠ **`--no-resume` 不是可选项。** `train_base.py` 默认会自动从
> `runs/base_checkpoints/<tag>/` 里最新的存档续训。同一个 tag 重跑时，
> auto-resume 会把上次的存档捡起来、**0 步就跑完**，
> 两组 bpb 会一模一样 —— 看起来「这个 trick 毫无影响」，
> 实际上是根本没训练。`uv run python -m training.train_base --help` 里有这条说明。

读结果的命令：

```bash
# 每个 checkpoint 旁边有 meta_XXXXXX.json，里面有 best_val_bpb
for t in d6_base d6_norope d6_noqk d6_gelu d6_tied d6_nosoftcap d6_muonadv d6_alltricks; do
  f=$(ls -t runs/base_checkpoints/$t/meta_*.json 2>/dev/null | head -1)
  [ -n "$f" ] && python3 -c "
import json,sys
d = json.load(open('$f'))
print(f'{\"$t\":<18} bpb={d[\"best_val_bpb\"]:.4f}  step={d[\"step\"]}')"
done
```

仓库里**已经有一份现成的** `scratch/ablation.sh`，直接用，不要覆盖它：

```bash
bash scratch/ablation.sh            # 汇总全部（默认基线 d6_base）
bash scratch/ablation.sh d4base     # 指定基线 tag
```

它会扫 `runs/base_checkpoints/*/`，读每个目录里最新的 `meta_*.json`，
打印 `val_bpb` 和相对基线的 `Δ`。它接受一个可选的基线 tag 参数 ——
写死 tag 的版本会在你用 `d4` 做消融时给出错误的 Δ 列。

如果你想自己写一份，注意两个易错点：
- 表头是 3 列（`实验 / val_bpb / Δ vs base`），下面的 `printf` 也必须是 3 列
- Δ 的计算要用 `float()`：`python3 -c "print(f'{${bpb} - ${base}:+.4f}')"`
  这种写法在 bash 里会被当成未定义变量而静默失败（上面那行就是错的示范）

**三条纪律**（否则消融实验就是自欺欺人）：

1. **必须用 `ablation` 档。** `smoke` 档（896 步）会把结论**测反符号**。
2. **一次只改一个变量。**
3. **`|Δ| < 0.001` 时重复跑一次。** 确认不是初始化随机性导致的波动。

> ★ 纪律第 3 条原先写的是 0.02 —— **那是个从未验证过的经验值**
> （项目最早那个 commit 里凭经验写下的）。
> 2026-10 实测：σ_Δ = 0.000190，换初始化额外偏移 0.000437。
> **0.001 ≈ 5.3σ**，比实测最大偏移高 2.3 倍。
>
> **把 0.02 当真值的代价：13 项消融里 11 项被误判成「测不出差异」**，
> 而它们其实是 7σ ~ 279σ 的真实效应。
> 这是「不把具体取值当规律」那条纪律的实例 ——
> **连判断显著性的阈值本身，都曾经是没测过的取值。**

**完整的消融总表模板见 `doc/tutorial/README.md`。**

---

## 常见坑

**坑 1：忘了 `reduction="none"`。**
`F.cross_entropy(..., reduction="mean")` 返回一个标量，
你没法按位置筛选哪些 token 有效。bpb 必须用 `none` 拿到 `(B, T)`
再按 `y >= 0` 掩码累加。

**坑 2：查了 input 的字节数而不是 target 的。**
模型的工作是「预测这个 token」，所以按 target 计费。
查 input 会得到一个**近似相同但系统上略高**的 bpb（因为每行的
最后一个 token 没有 target，它的 input 字节数被多算了）。

**坑 3：tokenizer 换了但 `token_bytes.pt` 没重算。**
症状：bpb 数字看着合理但明显偏大/偏小。
防御：每次重建 tokenizer 后确认 `token_bytes.pt` 一起更新
（`train_tokenizer.py` 里是紧挨着的两行）。

**坑 4：拿自己的 bpb 和论文比。**
你的 d6 / 2 亿 token 和 nanochat 的 d24 / 4e19 FLOPs 差 1250 倍。
**唯一可比的是差值。** `eval_base.sh` 的输出里专门有一段讲这个。

**坑 5：把 `math.log(2)` 写成 `2` 或者忘掉。**
nats → bits 的换算因子是 `1/ln(2) ≈ 1.4427`。
漏掉它会让 bpb 差 44%（1.7136 变成 1.187），而且看起来还挺合理。

---

## 延伸

**nanochat 完整版怎么做的**

`nanochat/loss_eval.py` 的 `evaluate_bpb` 和我们逐行对应，
包括「查 target 字节数」这个细节。

**它多做的**：`get_token_bytes` 存的是 bf16（省一点显存，
但精度够用，因为字节数都是小整数）。我们存 fp32，更保险。

**它用的主指标**：排行榜用的是 **CORE score** 而不是 bpb。
原因：bpb 衡量「语言建模」，CORE 衡量「解决下游任务的能力」。
两者相关但不等价 —— 一个可以靠背诵达到低 bpb 却不会做题。

CORE 的原理（`nanochat/core_eval.py`）值得预告：
**不让模型生成，而是用 loglikelihood 给每个选项打分。**
这个技巧在卷 7 会详细讲，因为它体现了「评测」和「生成」两种范式的差别。

**关于 bpb 的理论下界**：
自然英语的熵大约 1.0-1.2 bits/字符，理论上完美预测需要 ~1.0 bpb。
但要达到这个水平需要近乎无限的模型。实践中：
- 好模型（GPT-4 级）：~0.6-0.7
- 人类水平（用同样方式度量）：~0.8-1.0

**本教程的 bpb 目标**：
`ablation` 档 d6 **实测 1.0615**（`best_val_bpb`）/ 1.0754（全验证集 `eval_base`），
离 GPT-2（~0.95）已经很近了。`full` 档 d24 外推约 **0.9-1.1**（仍未实测）。

比随机初始化（2.8）好得多，而且**足以分辨 0.01-0.05 的消融差异**。

---

## 卷 1 完成

你现在的能力：
- ✅ 解释 BPE 在做什么，能手写一个迷你版
- ✅ 解释 loss mask 的来历，知道 NaN 是怎么来的
- ✅ 实现并可视化 BOS-aligned best-fit，理解它为什么丢 token 也划算
- ✅ 用 bpb 而不是 loss 来衡量模型

**下一卷**开始搭模型：
[第 07 章：先跑通一个最小 GPT](07-先跑通一个最小GPT.md)
