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
    was_training = model.training     # ★ 先读状态，再 eval()
    model.eval()
    total_nats, total_bytes = 0.0, 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches: break
        x, y = batch[0], batch[1]
        loss_vec = model(x, y, loss_reduction="none")     # (B, T) 逐 token
        valid = (y >= 0)                                   # -1 是不计 loss 的
        total_nats += loss_vec[valid].sum().item()
        total_bytes += token_bytes[y[valid]].sum().item() # ← 查表
    if was_training: model.train()      # ★ 还原 train 模式
    return total_nats / (math.log(2) * max(total_bytes, 1e-9))
```

**两个容易漏的点**：

1. `token_bytes[y[valid]]` 查的是 **target** 的字节数，不是 input 的。
   这是有意的：模型的工作是「预测这个 token」，
   所以应该按「被预测的东西有多长」来计费。
2. `was_training` 必须在 `model.eval()` **之前**读。顺序反了的话
   `eval()` 已经把 `training` 置成 False，`was_training` 恒为 False，
   下面的还原就成了死代码。`train_base.py` 会在训练中途调用本函数，
   模型被留在 eval 模式会一直 eval 下去 —— 现在没 dropout 看不出问题，
   加了 dropout 就会静默训坏。

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

`ablation` 档 d6（2 亿 token，比 smoke 多 14 倍数据）大概会到 **1.3-1.5**。
离 GPT-2（~0.95）还差很远 —— 这符合第 01 章说的「硬件差了 1250 倍」。

**但这不影响本章的价值**：消融实验看的是**差值**，
而差值恰好在 0.01-0.05 量级 —— 比绝对值有意义得多。

---

## 手抄代码

### 第 1 块：算 `token_bytes`


<details>
<summary><b>👀 展开参考答案（先自己想 20 分钟）</b></summary>

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

</details>

### 第 2 块：算 bpb


<details>
<summary><b>👀 展开参考答案（先自己想 20 分钟）</b></summary>

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
        # dataloader 产出 (x, y, state)；这里只关心前两个
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

</details>

---

## 动手验证

### 验证 1：亲手算一遍 bpb

新建 `scratch/bpb_demo.py`：

> 🔒 **需要先完成**：卷 1 第 02-06 章（tokenizer + dataloader）+ 至少跑过一次 `train_base`
>
> 现在跑到这步会得到 `NotImplementedError: 待实现：xxx` ——
> 那是「你还没写这一块」，不是环境坏了。想跳过：`git checkout solution`。

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
bash script/train_base.sh smoke     # 先确保有 checkpoint
uv run python scratch/bpb_demo.py
```

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

随机初始化时 bpb = 2.349（= log2(8192)/平均字节数），
训好的模型是 1.698。**这个 0.65 的下降就是模型「学到的知识」的量化。**

**注意平均字节数是 5.535 而不是第 02 章报的 3.24** ——
第 02 章那个数字是在一段 162 字节的英文样句上测的，不具代表性。
**语料上的真实均值要靠 `token_bytes` 表算。** 这正好说明了为什么要逐个查表。

### 验证 2：亲手制造一次「loss 相同但 bpb 不同」

新建 `scratch/bpb_uncomparable.py`：

> 🔒 **需要先完成**：无（纯 math，可以现在就做）
>
> 现在跑到这步会得到 `NotImplementedError: 待实现：xxx` ——
> 那是「你还没写这一块」，不是环境坏了。想跳过：`git checkout solution`。

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
bash script/train_base.sh ablation --model-tag d6_norope     --no-rope       --no-resume
bash script/train_base.sh ablation --model-tag d6_noqk       --no-qk-norm    --no-resume
bash script/train_base.sh ablation --model-tag d6_gelu       --activation gelu --no-resume
bash script/train_base.sh ablation --model-tag d6_tied       --tie-embeddings  --no-resume
bash script/train_base.sh ablation --model-tag d6_nosoftcap  --no-softcap    --no-resume
bash script/train_base.sh ablation --model-tag d6_muonadv    --muon-advanced --no-resume
bash script/train_base.sh ablation --model-tag d6_alltricks  --all-tricks    --no-resume
```

> ⚠ **`--no-resume` 不是可选项。** `train_base.py` 默认会自动从
> `runs/base_checkpoints/<tag>/` 里最新的存档续训。同一个 tag 重跑时，
> auto-resume 会把上次的存档捡起来、**0 步就跑完**，
> 两组 bpb 会一模一样 —— 看起来「这个 trick 毫无影响」，
> 实际上是根本没训练。`script/train_base.py --help` 里有这条说明。

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

1. **必须用 `ablation` 档。** `smoke` 档的 bpb 噪声（±0.02）大于效应本身。
2. **一次只改一个变量。**
3. **`|Δ| < 0.02` 时重复跑一次。** 确认不是初始化随机性导致的波动。

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
`ablation` 档 d6 应该能到 **1.3-1.5**。离 GPT-2（~0.95）还远，
但比随机初始化（2.8）好得多，而且**足以分辨 0.01-0.05 的消融差异**。

---

## 卷 1 完成

你现在的能力：
- ✅ 解释 BPE 在做什么，能手写一个迷你版
- ✅ 解释 loss mask 的来历，知道 NaN 是怎么来的
- ✅ 实现并可视化 BOS-aligned best-fit，理解它为什么丢 token 也划算
- ✅ 用 bpb 而不是 loss 来衡量模型

**下一卷**（卷 2 模型）开始搭模型，从 🚧 第 07 章「先跑通一个最小 GPT」开始
—— 教程正文还没写，但**代码骨架和判据测试已经就位**：

```bash
# 看看卷 2 你要实现什么
grep -n "raise NotImplementedError" src/model/layers.py src/model/gpt.py

# 敲完之后
uv run pytest tests/test_core.py -k "uniform or causal" -v
```

卷 2-4 的骨架一共 34 处待实现（`layers.py` 15 + `gpt.py` 7 +
`orthogonalize.py` 5 + `muon.py` 7），配合 `src/` 里的 docstring
（每处都写清了「要做哪几步」和「怎么验证」）足够跟敲。
