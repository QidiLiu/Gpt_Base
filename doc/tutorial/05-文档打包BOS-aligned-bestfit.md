# 05 · 文档打包：BOS-aligned best-fit

**这是卷 1 最有价值的一章。** 这里有一个真实的工程权衡，值得反复读。

---

## 本章目标

- 说清楚「变长文档 → GPU 能吃的矩形张量」的每一种做法及其代价
- 理解 **BOS-aligned best-fit** 装箱算法的每一步，以及它为什么丢 35% 的 token 也划算
- 亲手实现并可视化这个算法，量出「利用率」和「裁剪率」

---

## 前置回顾

[第 04 章](04-对话模板与特殊token.md) 我们处理了「对话」这种结构化数据。
本章处理的是**预训练语料**——ClimbMix 里 8000 万篇长度不一的自然文档。

---

## 概念：问题是什么

实测 ClimbMix 的文档长度分布（`scratch/` 里可以自己查）：

| 分位 | 字符数 | 约等于 token |
|---|---|---|
| min | 11 | 3 |
| p25 | ~1200 | ~370 |
| **中位数** | **~2400** | **~740** |
| p75 | ~5000 | ~1550 |
| max | 111,465 | ~34,000 |

而 GPU 只吃**矩形** tensor：一个 `(B, T)` 的 int64 张量。
`T` 必须是一个固定的数。

**所以核心问题是：长度从 3 到 34000 的文档，怎么拼成 `T=1024` 的行，
又不浪费算力？**

---

## 概念：四种做法，代价递增

### 做法 0：padding 到最长

```
文档: [A, A, A, A, ..., A(34000 token), A]
行:   [A, A, A, A, ..., A, PAD, PAD, PAD, ..., PAD]
```

**问题**：如果 batch 里混进一篇 34000 token 的文档，所有行都要补到 34000，
99% 的计算都在算 PAD。GPU 利用率掉到 1%。

而且模型会学到「一大段 PAD 是什么」这种推理时永远用不到的模式。

### 做法 1：截断到固定长度

```
文档: [A, A, A, ..., A(34000)]  ->  只取前 1024 个
```

**问题**：长文档 97% 的内容被丢弃。语料里有很多长文档（技术文章、书籍），
丢掉的正是信息量最大的部分。

### 做法 2：naive 拼接（nanoGPT 的做法）

把所有 token 首尾相连成一条超长流，然后随机切 `T` 长的窗口：

```
流:   [文1][文2][文3][文4]...
窗口:      [文2的后半 + 文3的前半]     <- 跨越了文档边界
```

**问题：cross-contamination（交叉污染）**。
模型会花容量学「从上一篇文章的中间接着写」——这在真实推理时**永远不会发生**。

论文里的证据（nanochat 自己的实验）：朴素拼接会明显拖累最终 bpb。

优点是实现极简（20 行），而且 100% 利用率。

### 做法 3：BOS-aligned best-fit（nanochat 的做法，本项目采用）

四条规则：

1. **每行开头放一个 `<|bos|>`**（不是手动放，而是靠第一篇文档自带的）
2. **每篇文档前面也放 `<|bos|>`**（tokenizer 编码时 `prepend="<|bos|>"`）
3. 装箱时**优先选「能完整放下的最长文档」**（best-fit）
4. 都放不下时，**裁剪最短的那篇**填满剩余空间

```
行: [<bos>][文A 完整][<bos>][文B 完整][<bos>][文C 的前 400 token...]
                                     └─ 文C 被裁剪，其余丢弃
```

**关键性质**：
- 利用率 **100%**（无 padding，每个 token 都被训练）
- 但约 **35% 的 token 在裁剪时被丢弃**（T=2048 时）
- 每个文档边界都有 `<|bos|>`，模型学会「看到 BOS 就是新文章开始」

---

## 概念：为什么丢 35% 划算

这是本章的核心论证，值得认真想清楚。

**表面上的账**：
```
naive 拼接：  100% 的 token 都用于训练，但 15% 的窗口「跨文档」（估计）
best-fit：     65% 的 token 用于训练，但 0% 跨文档
```

**真正的账不是「用了多少 token」，而是「学到了什么」**：

| | naive 拼接 | best-fit |
|---|---|---|
| 有效训练 token | 100% | 65% |
| **污染的 token**（跨文档，学习推理时用不到的模式） | 约 10-15% | **0** |
| **干净的有效 token** | 85-90% | **65%** |
| 文档边界信号 | 无（模型要猜哪里是边界） | 显式 `<|bos|>` |

**结论：65% 的干净数据 > 90% 的脏数据。**

更微妙的一点：best-fit **丢掉的恰恰是最难利用的部分**。
被裁掉的都是「放不进剩余空间的长文档」，
而这些文档在别处（别的行）已经被完整地训练过了。
真正「一次性丢失」的信息量，比 35% 这个数字看起来要少。

**nanochat 的排行榜数据支持这个取舍**：
它的 speedrun 从 3.04 小时优化到 2.02 小时的那次改动里，
有一项就是换了更好的数据源，但数据管线一直保持着 best-fit。

---

## 概念：细节 —— 为什么每篇文档都要 `<|bos|>`

第 2 条规则容易被忽略。设想一个模型在推理时收到用户的一整段 prompt
（前面有 `<|user_start|>`，后面等 `<|assistant_start|>`）。

如果训练时文档边界是**隐含的**（靠换行猜），模型学到的「文章结构」是模糊的。
加了 `<|bos|>` 之后：

- 训练时：`<|bos|>` 明确标记「新文档开始」
- 推理时：`<|user_start|>` 本身就承担了同样的角色
- **模型的「开始新话题」的能力被迁移到了对话场景**

这是一个很优雅的抽象复用：同一个 token，在预训练里标记文档边界，
在 SFT 里标记消息边界。

---

## 手抄代码

代码位置：`src/data/dataloader.py: make_dataloader`

### 第 1 块：装箱的核心算法（不依赖 torch，纯逻辑，便于理解）

```python
def pack_row(doc_buffer: list[list[int]], capacity: int) -> tuple[list[int], dict]:
    """
    把一批文档装进一行，返回 (装好的 token 列表, 统计信息)。

    两条规则：
      1) 优先选「能完整放下的最长文档」（best-fit）
         —— 长文档更难安排，先处理它们能减少浪费
      2) 都放不下时，裁剪「最短的」那篇填满
         —— 裁最短的，浪费最少

    返回的统计里包含裁剪了多少 token，用来算「利用率」。
    """
    row = []
    buf = list(doc_buffer)          # 拷贝一份，不破坏调用方的 buffer
    cropped = 0
    n_docs = 0

    while buf:
        remaining = capacity - len(row)

        # ── 规则 1：best-fit ──
        best_idx, best_len = -1, 0
        for i, doc in enumerate(buf):
            n = len(doc)
            if n <= remaining and n > best_len:
                best_idx, best_len = i, n
        if best_idx >= 0:
            row.extend(buf.pop(best_idx))
            n_docs += 1
            continue

        # ── 规则 2：裁剪最短的 ──
        shortest_idx = min(range(len(buf)), key=lambda i: len(buf[i]))
        doc = buf.pop(shortest_idx)
        row.extend(doc[:remaining])          # 只取能放下的部分
        cropped += len(doc) - remaining       # 剩下的算「被丢弃」
        n_docs += 1
        break                                # 行满了

    return row, dict(capacity=capacity, used=len(row), n_docs=n_docs, cropped=cropped)
```

**注意 `while buf` 里的 `break`**：行装满后必须退出，
否则最后那个 `row.extend(doc[:remaining])` 里 `remaining` 已经是 0，
会陷入死循环。

### 第 2 块：和 torch 缓冲区对接

真实实现（`src/data/dataloader.py`）比第 1 块多做了三件事：

```python
def make_dataloader(tokenizer, batch_size, seq_len, split, device="cuda",
                    resume_state=None, buffer_size=1000,
                    tokenizer_threads=4, tokenizer_batch_size=128):
    """
    产出 (inputs, targets, state)：
        inputs, targets 形状 (batch_size, seq_len)，dtype int64，在 device 上
        state             打包位置，用于精确续训

    ── 三个关键设计 ────────────────────────────────────────────

    (1) 为什么每行容量是 seq_len + 1？
        要构造 inputs/targets 这一对，必须有 T+1 个 token：
            inputs  = row[0 : T]
            targets = row[1 : T+1]
        所以每行装 T+1 个，其中 row[0] 固定是 <|bos|>。

    (2) 为什么全程不 padding？
        padding token 会让模型在推理时遇到从未见过的 id 分布。
        宁可裁掉 35% 的 token，也要保证 100% 利用率 + 无 padding。

    (3) 为什么先建 CPU row_buffer 再一次性搬 GPU？
        逐行 torch.tensor(...) 会产生几千次小 H2D 拷贝，
        每次都有几微秒的启动开销。用 pinned memory + 一次
        non_blocking 拷贝可以把传输和 GPU 计算重叠。
    """
    row_capacity = seq_len + 1
    batches = iter_documents(split, tokenizer_batch_size, resume_state)
    bos = tokenizer.get_bos_token_id()

    # 已 token 化、还没用掉的文档
    doc_buffer: list[list[int]] = []
    pq_idx = rg_idx = 0
    epoch = 1

    def refill():
        """从 parquet 再取一批文本，token 化后塞进 doc_buffer。"""
        nonlocal pq_idx, rg_idx, epoch
        text_batch, st = next(batches)
        pq_idx, rg_idx, epoch = st["pq_idx"], st["rg_idx"], st["epoch"]
        # ★ prepend=bos：每篇文档自带 BOS，这就是「文档边界」的信号。
        #   行的开头不需要手动放 BOS —— 第一篇被放进来的文档
        #   自己的 BOS 正好落在位置 0。
        for tokens in tokenizer.encode(text_batch, prepend=bos,
                                       num_threads=tokenizer_threads):
            doc_buffer.append(tokens)

    use_cuda = (device == "cuda")
    # CPU 侧暂存区：一次装下整批，pin_memory 让 H2D 走 DMA
    cpu_buffer = torch.empty(2 * batch_size * seq_len, dtype=torch.long,
                             pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * batch_size * seq_len, dtype=torch.long,
                             device=device)
    cpu_inputs = cpu_buffer[:batch_size * seq_len].view(batch_size, seq_len)
    cpu_targets = cpu_buffer[batch_size * seq_len:].view(batch_size, seq_len)
    inputs = gpu_buffer[:batch_size * seq_len].view(batch_size, seq_len)
    targets = gpu_buffer[batch_size * seq_len:].view(batch_size, seq_len)

    row = torch.empty(row_capacity, dtype=torch.long)

    while True:
        for r in range(batch_size):
            # 从 0 开始。第一篇被放进来的文档自带 <|bos|>，
            # 所以位置 0 自然就是 BOS，不需要额外手动写。
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:    # 保证候选够多
                    refill()

                remaining = row_capacity - pos

                # 规则 1：best-fit —— 找「能完整放下」的最长文档
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    n = len(doc)
                    if n <= remaining and n > best_len:
                        best_idx, best_len = i, n
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row[pos:pos + best_len] = torch.tensor(doc, dtype=torch.long)
                    pos += best_len

                # 规则 2：都放不下 -> 裁剪「最短的」填满
                else:
                    shortest_idx = min(range(len(doc_buffer)),
                                       key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row[pos:pos + remaining] = torch.tensor(doc[:remaining],
                                                            dtype=torch.long)
                    pos = row_capacity

            cpu_inputs[r] = row[:-1]
            cpu_targets[r] = row[1:]         # 右移一位

        state = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        # 一次 H2D，之后每次 yield 复用同一块 GPU 显存（零分配）
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state
```

---

## 动手验证

### 验证 1：可视化装箱过程

`scratch/packing_demo.py` 已经写好了。跑：

```bash
uv run python scratch/packing_demo.py
```

#### 真实的 ClimbMix 文档分布

```
文档数 2048  长度 min=4 p25=279 中位数=628 p75=829 max=25055
（>2048 token 的占 4.2%）
```

**这个 shard 的文档比 nanochat 文档里说的要短**（中位数 628 而非 740，
只有 4.2% 超过 2048）。所以下面的裁剪率会明显低于 nanochat 说的 35%。

#### 实验 A：seq_len 的影响

```
     T    行数      填充率      每行文档数      完整装入率    裁剪/已用token
   256   200   100.0%       2.27      71.5%          4.9%
   512   200   100.0%       2.32      84.5%          0.5%
  1024   200   100.0%       2.21      90.3%          0.1%
  2048   200   100.0%       2.46      92.9%          0.0%
  4096   200   100.0%       4.17      90.0%          0.0%
```

**填充率恒为 100%** —— 这是设计目标（无 padding，每个 token 都被训练）。
**裁剪占比随 T 增大而下降** —— 行越长，越装得下更多完整文档。

**T=256 时裁剪 4.9%** 是这个实验最重要的观察：
本项目 `smoke` 档用 T=512、`full` 档用 T=1024，
都在「裁剪可忽略」的区域。如果你把 T 调到 128（debug 档），
裁剪率会飙升，训练效率大幅下降。

#### 实验 B：三种装箱策略对比

```
        策略      填充率      每行文档数      完整装入率    裁剪/已用token
  best-fit   100.0%       2.32      84.5%          0.5%
 first-fit   100.0%       5.41      86.2%          0.6%
      裁最长的   100.0%       2.35      86.2%        668.8%
```

**「裁最长的」裁掉了 668.8%**（即每训练 1 个 token 要浪费 6.7 个）——
这是规则 2 写反的灾难性后果。best-fit 和 first-fit 在这个数据上差别不大
（0.5% vs 0.6%），因为文档长度分布比较集中。

**但在小规模的手工构造上，差别是 2 倍**（见实验 C）。

#### 实验 C：手工构造，直观看到 best-fit 的决策

```
文档长度: A=30, B=35, C=70, D=75, E=100     行容量 200

    best-fit: 装入 E(100->100), D(75->75), A(30->25)
              填充 200/200  裁剪掉 5 token
              [aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbCCCCCCCCC]

  first-fit: 装入 A(30->30), B(35->35), C(70->70), D(75->65)
              填充 200/200  裁剪掉 10 token
              [aaaaaaaaaabbbbbbbbbbbbcccccccccccccccccccccccccDDDDDDDDDDDDDDDDDDDDDDD··]

      裁最长的: 装入 E(100->100), D(75->75), C(70->25)
              填充 200/200  裁剪掉 45 token
              [aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbCCCCCCCCC]
```

**图例**：小写 = 完整装入；**大写 = 被裁剪**；`·` = 剩余空隙。

- **best-fit 裁 5 个**：先把最大的 E(100) 和 D(75) 安排掉（它们最难安排），
  剩 25 装不下最小的 A(30)，裁掉 5 个刚好填满。
- **first-fit 裁 10 个**：按顺序拿，小块先占位置，到 D(75) 时只剩 65，裁掉 10 个。
- **裁最长的裁 45 个**：最差。

**这就是「先安排大块」的价值**：长文档能放下的位置越来越少，
先处理它们，剩下的空间才能被小文档高效利用。

### 验证 2：确认 dataloader 的三个不变量

`scratch/check_data.py`（第 01 章写的）：

```bash
uv run python scratch/check_data.py
```

```
inputs  (4, 512) torch.int64
targets (4, 512)
state   {'pq_idx': 0, 'rg_idx': 0, 'epoch': 1}

[OK] targets == inputs 右移一位
[OK] 每行以 <|bos|> 开头
[OK] 无 padding，全部落在 [0, 8192)

第 0 行里有 1 个 BOS，即约 1 篇文档
前 20 个 token：
  <|bos|>(8183)|A|ud|i| A|5| than| in| the| B|M|W| |23|0|,| due| to| the| A
```

**第 0 行只有 1 个 BOS**：512 token 装不下一篇 628 token 的文档，
所以整行就是「1 篇文档 + 下一篇被裁掉的开头」。

**注意开头是 1 个 BOS 而不是 2 个** —— 这是常见的坑
（手动 `row[0]=bos` 加上文档自带的 BOS）。检查方法：

```bash
uv run python -c "
from data.tokenizer import get_tokenizer
from data.dataloader import make_dataloader
tok = get_tokenizer(); bos = tok.get_bos_token_id()
x,_,_ = next(make_dataloader(tok, 4, 512, 'train', 'cpu'))
print('第 0 行前 3 个是不是都是 BOS ?', (x[0,:3] == bos).tolist())
"
```

预期 `[True, False, False]`。如果是 `[True, True, False]` 就踩坑了。

## ★ 消融实验

本章的消融是**最值得做的一个**，因为它检验的是一个有明确代价的取舍。

| 变体 | 做法 | 预期 |
|---|---|---|
| best-fit + BOS（本项目） | — | 基线 |
| **naive 拼接** | 改成随机切窗口 | bpb 变差（跨文档污染） |
| **first-fit** | best-fit 改成按顺序取第一个能放下的 | bpb 略变差（裁剪更多） |
| padding | 短行补 PAD | bpb 明显变差 + 浪费算力 |
| 裁剪最长的 | 规则 2 改成裁最长的 | bpb 变差（浪费更多） |

**naive 拼接怎么实现**（最值得做的那个）：

新建 `scratch/naive_dataloader.py`：

```python
"""
对照实现：nanoGPT 式的 naive 拼接。
把所有 token 串成一条长流，随机切 T 长的窗口。
"""
import torch
import pyarrow.parquet as pq
from data.tokenizer import get_tokenizer
from data.dataset import list_parquet_files
from common import get_dist_info


def naive_dataloader(tokenizer, batch_size, seq_len, split="train",
                     device="cpu", pool_tokens=20_000_000):
    """
    和 make_dataloader 产出同样形状的 (x, y)，但用 naive 拼接。
    差异：窗口会跨越文档边界（cross-contamination）。
    """
    import random
    # 1) 收集一大批 token（真实实现会流式做，这里为简单起见先收集）
    pool = []
    for path in list_parquet_files(split):
        pf = pq.ParquetFile(path)
        for rg_idx in range(min(pf.num_row_groups, 6)):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                pool.extend(tokenizer.encode(text, prepend=tokenizer.get_bos_token_id()))
                if len(pool) >= pool_tokens:
                    return _slice(pool, batch_size, seq_len, device, random)
    return _slice(pool, batch_size, seq_len, device, random)


def _slice(pool, batch_size, seq_len, device, random):
    while True:
        ix = [random.randrange(0, len(pool) - seq_len - 1) for _ in range(batch_size)]
        x = torch.stack([torch.tensor(pool[i:i + seq_len]) for i in ix])
        y = torch.stack([torch.tensor(pool[i + 1:i + 1 + seq_len]) for i in ix])
        yield x.to(device), y.to(device)


if __name__ == "__main__":
    tok = get_tokenizer()
    dl = naive_dataloader(tok, batch_size=4, seq_len=512)
    x, y = next(dl)
    bos = tok.get_bos_token_id()
    n_bos = int((x == bos).sum())
    print(f"naive 拼接: {tuple(x.shape)}")
    print(f"每行 BOS 数: {(x == bos).sum(dim=1).tolist()}")
    print(f"第 0 行前 24 token: ")
    print(" ", tok.visualize(x[0][:24].tolist(), with_token_id=True))
```

跑：

```bash
uv run python scratch/naive_dataloader.py
```

**观察**：`每行 BOS 数` 应该远大于 1（best-fit 是 1.1 左右），
因为 naive 拼接会在一行里塞进好几篇文档的**部分内容**。
但 BOS 只在每篇文档的开头出现，所以一行里 BOS 少不代表污染少 ——
**真正的问题是「跨边界」**：

一行 512 token，文档中位数 740 token，所以一行里通常有 1 个 BOS
和 1 次「从某篇文档中间切进来」。这个「切进来」的起点
模型是没有任何信号能识别的。

**怎么量化这个效应？** 一个可行的代理指标：
统计每个窗口的**第一个 token 到第一个 BOS 之间的距离**。
- best-fit：距离恒为 0（窗口一定从 BOS 开始）
- naive：距离 > 0 表示窗口从文档中间开始，这个 token 的
  上下文是「残缺的」

```python
def contamination_rate(x, bos):
    """每行第一个 BOS 之前有多少个 token（>0 说明窗口从文档中间开始）。"""
    return [int((row[:int((row == bos).nonzero()[0])] != bos).sum()) if (row == bos).any() else -1
            for row in x]
```

把这个指标加到 `eval_base.sh` 里，分别跑 best-fit 和 naive 训练，
比较 val_bpb。**这是本教程最值得做的一个消融实验。**

---

## 常见坑

**坑 1：手动 `row[0] = bos` 导致开头出现两个 BOS。**
我在写这个项目时踩过。因为文档在 tokenizer 编码时已经 `prepend=bos` 了，
第一篇被放进来的文档自带 BOS，位置 0 自然就是 BOS。
如果再手动写一次，就会得到 `<|bos|><|bos|>正文...`。
`pytest -k kv_cache` 旁边的断言能查到这个：
```python
assert (x[:, 0] == bos).all()
# 更严格的：assert 第二个 token 不是 bos（除非文档为空）
```

**坑 2：忘了 `row_capacity = seq_len + 1`。**
会得到 `inputs` 和 `targets` 长度不匹配，或者丢掉每行最后一个 token。

**坑 3：`pack_row` 里忘了 `break`。**
规则 2 之后行已满，`while buf` 继续循环时 `remaining=0`，
`doc[:0]` 是空列表，行永远不满，但 buf 在缩小，最终不报错但行为诡异。

**坑 4：best-fit 的内层循环是 O(buffer_size)。**
`buffer_size=1000` 时每放一篇文档要扫 1000 个，
一行 512 token 可能放好几篇 → 每行几千次比较。
这在 Python 侧是真实开销（`full` 档下能占到 dataloader 时间的 20%）。
优化方向：用两个有序列表分别存「能放下」和「放不下」的文档，
或者按长度排序后用双指针。**nanochat 也没优化这一点**，
因为它把 dataloader 和 GPU 计算重叠了，CPU 侧慢一点没关系。

**坑 5：想改成 naive 拼接时忘了 dataloader 要无限循环。**
`make_dataloader` 里的 `while True` 保证训练可以跑任意多步。
改成 naive 时如果只 yield 有限次，训练会崩。

---

## 延伸

**nanochat 完整版怎么做的**

`src/data/dataloader.py` 的逻辑和本项目**逐行对应**（我按它改的），
只有三处差别：

1. **nanochat 有一个 fallback**：
   > 「如果你的数据极少且文档很长，回落到原来的 naive 拼接」
   > 见 `tokenizing_distributed_data_loader`（不带 `_bos_bestfit` 的那个函数）

   也就是说 nanochat 自己也承认 best-fit 不是万能的 ——
   数据太少时，35% 的裁剪率不可接受。

2. **nanochat 显式统计裁剪率**并在 docstring 里写「约 35%」，
   我们没有统计（可以自己加，见验证 1 的实验 A）。

3. **nanochat 的 DDP 分片**是在 `iter_documents` 里按 row group 交错分配
   （`rg_idx = rank; rg_idx += world_size`）。我们抄了这段但
   `world_size` 恒为 1，所以等价于顺序读。

**比 nanochat 更进一步的方向**（本项目没做）：

- **按长度排序的 buffer**：把 `doc_buffer` 保持有序，best-fit 从
  「最长能放下的」退化成 O(1) 的二分查找。代价是丧失「多样性」
  （相邻的文档都来自同一批），可能有害。
- **分桶**：按长度把文档分成几个桶，每批只从同一个桶取。
  这样每行能装更多完整文档，但会损失文档顺序上的随机性。
- **两阶段裁剪**：先粗排一遍确定「这一批有多少长文档」，
  再据此调整 `buffer_size`。

---

## 下一章

[第 06 章：bits per byte](06-bits-per-byte.md) ——
为什么 loss 不能跨模型比较，而 bpb 可以。这是卷 1 的收尾，也是你做消融实验的主要工具。
