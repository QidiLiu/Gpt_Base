# 03 · 手写一个玩具 BPE

## 本章目标

把上一章的「看到合并过程」升级成一个**能训练、能推理、能序列化**的迷你分词器，
并搞清楚生产实现（rustbpe + tiktoken）为什么要拆成两个库、两个阶段。

---

## 前置回顾

[第 02 章](02-为什么需要分词器.md) 我们写了 30 行的 `toy_bpe()`，
看到了合并顺序。本章把它补成一个完整的东西。

---

## 概念：为什么 BPE 要拆成「预切分」和「合并」两阶段

这是本章最核心的认知。生产实现（tiktoken / rustbpe）的结构是：

```
        训练阶段（慢，可以用贪心+全局统计）      推理阶段（必须极快）
        ─────────────────────────────────      ──────────────────────
文本 → ① 预切分（正则）→ ② 反复合并相邻对        文本 → ① 预切分（正则）→ 查 merge 表
              ↑                                              ↑
         只在块内合并                                  不需要「合并」这个过程
         （跨块合并会造出无意义 token）              （因为 merge 表已经定死了）
```

**为什么预切分是必须的？**

如果不预切分，合并是全局的，那么 `the` 后面如果紧跟 ` cat`，
BPE 可能合并出 `the cat`（带空格）这种 token。这在推理时非常糟：
- 同一个词 `cat` 在句首（前有空格）和句中（后有空格）会是**不同的 token**
- 序列长度取决于上下文，缓存/对齐全乱

预切分用正则把文本切成「语义单元」，强制合并只能在单元内进行，
就杜绝了这个问题。

**为什么推理时不需要「合并」这一步？**

因为 BPE 是**贪心且确定**的：给定合并表，编码结果唯一。
所以推理时只要「按 rank 从小到大把能合并的地方都合并掉」，
等价于「从大到小逐个应用 merge 规则，直到没有能应用的」。
tiktoken 用一个哈希表 + 堆在 O(文本长度) 时间内做完。

而训练时必须「统计全语料的相邻对频率」，这是 O(语料) 的，
所以训练慢、推理快 —— 这就是为什么两个库。

---

## 概念：字节兜底（为什么任何文本都能表示）

有一个很朴素的问题：BPE 从「字符」开始合并。如果遇到中文字符怎么办？

答案在 GPT-2/GPT-3 时代就定了：**不用字符，用字节**。

```
基础符号集 = 全部 256 个字节值
```

这样：
- 任何 UTF-8 文本都能被表示（不会 OOV）
- 代价是英文文本的前几轮合并全是「拼回字母」（`b`+`e` → `be` …），
  比较浪费。所以一般跳过前 100 轮左右的合并

rustbpe / tiktoken 内部都处理了这个。本章的玩具版本也会。

---

## 手抄代码

### 第 1 块：迷你 BPE 的核心

新建 `scratch/mini_bpe.py`：


<details>
<summary><b>👀 展开参考答案（先自己想 20 分钟）</b></summary>

```python
"""
一个可训练、可推理、可序列化的迷你 BPE。只为了看清原理，不追求任何性能。
"""

import json
import re
import time
from collections import Counter

# 生产级预切分正则（与 nanochat / tiktoken 的 SPLIT_PATTERN 同构）
#
# ★ 关键：那个「 ?[^\s\w]++[\r\n]*」分支不能省。
#   `++` 是「一次或多次」，它保证标点（? ! , . : ;）被吃掉。
#   我第一版写成 `[^\r\n\w]?+\w+`（`?+` 是「零或一次」），结果 `?` 匹配不上，
#   而 re.finditer 只返回匹配、**不报告跳过** —— 那个问号就静默消失了。
GPT2_PAT = (
    r"""'(?i:[sdmt]|ll|ve|re)"""
    r"""|[^\r\n\w]?+\w+"""          # 可选非字母前缀 + 一串字母
    r"""|\d{1,2}"""                # 1-2 位数字一组
    r"""| ?[^\s\w]++[\r\n]*"""      # 可选空格 + 标点（一次或多次）+ 换行
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""  # 换行、行尾空白、兜底空白
)


class MiniBPE:
    def __init__(self, num_merges: int, min_frequency: int = 2):
        # 合并表：{符号 bytes: rank}。rank 越小优先级越高（= 训练顺序）
        self.merges = {}
        self.num_merges = num_merges
        self.min_frequency = min_frequency

    def pre_tokenize(self, text: str) -> list[bytes]:
        """
        把文本切成不可再分的块。合并只能在块内进行。

        ★ re.finditer 会**静默跳过**不匹配的字符，所以必须验证
          「所有块拼起来 == 原文」。这行断言是本章最值得抄走的东西。
        """
        blocks = [m.group(0).encode("utf-8") for m in re.finditer(GPT2_PAT, text)]
        assert b"".join(blocks) == text.encode("utf-8"), (
            "预切分正则有字符没匹配上！re.finditer 不会报错，只会静默丢掉它们。"
            "通常是缺少标点兜底分支。")
        return blocks

    # ── 阶段 2：训练合并表 ─────────────────────────────────────
    def train(self, text: str):
        """
        ★ 坑：符号必须用 bytes 表示，不能用 chr(byte) 转成 str。
          b"12" 是两个符号 b"1"、b"2" 合并的结果；
          而 chr(49)+chr(50) 拼出来是字符串 "12"，与真正的 "12" 无法区分。
          后果：merge 表里混进「两个数字字节」的假象，编码时张冠李戴。
          生产实现（rustbpe / tiktoken）全程用 bytes 作为符号，
          正是为了避开这个坑。
        """
        words = Counter()
        for block in self.pre_tokenize(text):
            words[self._symbols(block)] += 1     # 初始符号 = 每个单字节

        for _ in range(self.num_merges):
            # ① 统计所有相邻对的加权频率
            pairs = Counter()
            for word, cnt in words.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += cnt
            if not pairs:
                break
            # ② 取最高频
            (a, b), n = pairs.most_common(1)[0]
            if n < self.min_frequency:
                break
            # ③ 记录规则。rank = step，越小优先级越高
            self.merges[a + b] = len(self.merges)
            # ④ 应用合并
            new_words = Counter()
            for word, cnt in words.items():
                new_words[self._merge_word(word, a, b)] += cnt
            words = new_words

    @staticmethod
    def _symbols(block: bytes) -> tuple:
        """把一个块拆成「单字节符号」的 tuple。"""
        return tuple(block[i:i + 1] for i in range(len(block)))

    @staticmethod
    def _merge_word(word, a, b):
        """把 word 里的所有 (a,b) 合并成 a+b，返回新 tuple。"""
        out, i = [], 0
        ab = a + b
        while i < len(word):
            if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
                out.append(ab); i += 2
            else:
                out.append(word[i]); i += 1
        return tuple(out)

    # ── 阶段 3：推理 ───────────────────────────────────────────
    def _merge_all(self, block: bytes) -> list[bytes]:
        """
        对单个块应用全部 merge 规则：按 rank 从小到大反复应用，
        直到没有规则能匹配。这与训练时的贪心过程等价，所以编码唯一。
        """
        parts = list(self._symbols(block))
        while True:
            best_rank, best_i = None, None
            for i in range(len(parts) - 1):
                r = self.merges.get(parts[i] + parts[i + 1])
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        return parts

    def encode(self, text: str) -> list[bytes]:
        out = []
        for block in self.pre_tokenize(text):
            out.extend(self._merge_all(block))
        return out

    def decode(self, tokens: list[bytes]) -> str:
        return b"".join(tokens).decode("utf-8", errors="replace")
```

</details>

完整可运行版本在 `scratch/mini_bpe.py`（已经写好了，直接跑）。

---

## 动手验证

### 验证 1：训一个迷你 BPE 并看压缩率

`scratch/mini_bpe.py` 已经写好了。跑：

> 🔒 **需要先完成**：无 —— 玩具 BPE 是纯 Python 玩具，只用到 `download_tiny_shakespeare`（卷 0）
>
> 现在跑到这步会得到 `NotImplementedError: 待实现：xxx` ——
> 那是「你还没写这一块」，不是环境坏了。想跳过：`git checkout solution`。

```bash
uv run python scratch/mini_bpe.py
```

**实测输出**（纯 Python，2000 条 merge 规则，44MB 语料）：

```
训练 2000 条 merge 规则耗时 50.1s（纯 Python，很慢）

测试文本: 84 字节
编码结果: 28 token, 压缩率 3.00 bytes/token, 耗时 0ms
解码回原文一致 [OK]

  To| be|,| or| not| to| be|,| that| is| the| qu
  est|ion|:| |Whe|ther| '|tis| nob|l|er| in
   the| mind| to| suffer

--- 对比：代码和数字的压缩率明显更低 ---
  57 字节 -> 38 token, 压缩率 1.50
  de|f| comp|ute|(|a|,| b|)|:| return| |2|0|2|6| |*| a| |+| b| | |#| not|e|:| |x|=|1|0|0|0|0|0|0

--- merge 表：rank 就是训练顺序（小的先应用）---
  rank     0 : ' t'
  rank     1 : 'he'
  rank     2 : ' a'
  rank     3 : 'ou'
  rank     4 : ' s'
  rank     5 : ' m'
  rank     6 : 'in'
  rank     7 : ' w'
  rank     8 : 're'
  rank     9 : 'ha'
  rank    10 : ':\n'
  ...

=== 预切分为什么不能省 ===
  'the cat sat' -> 预切分 ['the', ' cat', ' sat']
  同一个词在不同上下文里被切成不同 token：
    'I like the cat'       -> I| like| the| c|at
    'the cat'              -> the| c|at
    'say the cat'          -> s|ay| the| c|at
```

**五个观察点**：

**① 压缩率 3.00 bytes/token，接近生产实现的 3.24。**
但代价是训练花了 **50 秒**（生产实现 6 秒，语料还多 250 倍）。

**② 英文散文 3.00，代码只有 1.50 —— 差了一倍。**
看代码的分词结果：`de|f| comp|ute|(|a|,| b|)` 里
`def` 被切成了 `de`+`f`，`compute` 切成了 `comp`+`ute`。
原因很简单：这些子串在**英文散文中**从没出现过，所以 merge 表里没有。
**BPE 是领域相关的** —— 用英文训的 tokenizer 编码代码会很慢。
（反过来，nanochat 选 ClimbMix 而不是纯英文语料，就是为了多样性。）

**③ 数字被切成一位一组**：`2|0|2|6`。
我们这个玩具版用了 `\d{1,2}` 但规则数太少，2000 条规则不够把
`20`/`26` 这些常用组合合出来（它们在英文语料里太罕见）。
生产实现靠 **1.5 万倍的训练语料**（10 亿字符）+ 32768 条规则解决这个问题。

**④ merge 表的 rank 0-9 全是最高频的**：` t` `he` ` a` `ou` ` s` ` m` `in` ` w` `re` `ha`。
这就是「用高频换低频」的直接证据。rank 1999 那种（`'cher'`）
只出现两三次，合并它几乎不省什么，但占了一个词表位置。

**⑤ 预切分的效果**：`the` 在三种上下文里都完整成 token，
` cat` 也完整成一个 token（**带前导空格**）。

---

### ★ 这一章我踩到的两个真 bug（值得单独讲）

写这个玩具 BPE 时我踩了两个坑，它们都属于「**静默错误**」——
不报错、能跑、但结果是错的。这类 bug 最值得学。

#### 坑 A：用 `chr(byte)` 表示符号，导致「两个字节」和「一个字符」混淆

第一版我把字节转成字符：`parts = [chr(c) for c in block]`。
看起来没问题（`chr(65)` 就是 `'A'`），但：

```
b"12"  的两个符号是   b"1" 和 b"2"
chr(49)+chr(50) 拼出来是  "12"   <- 这是「一个字符串」，
                                   与上面两个符号在类型层面无法区分
```

结果 merge 表里出现了 `"129"`、`"20"`、`"134"` 这种**「两个数字字节」**
的条目，而 `decode` 出来的东西完全不是原文。

**症状**：压缩率只有 1.04 bytes/token（而不是 3.0），
merge 表里的 token 全是数字串。

**修法**：符号全程用 `bytes` 表示（`block[i:i+1]`），
merge 表的键是 `bytes`，拼接用 `+`（`bytes + bytes` 还是 `bytes`）。
rustbpe / tiktoken 内部就是这么做的。

#### 坑 B：预切分正则漏了标点分支，字符被 `re.finditer` 静默丢弃

第一版正则：
```python
r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\w]?+\w+|\d+|\s*[\r\n]|\s+(?!\S)|\s+"""
```

测试文本里有 `question:`，那个 `?` 匹配不上任何分支。
而 **`re.finditer` 只返回匹配结果，不会警告你跳过了什么** ——
所以那个问号就凭空消失了。

**症状**：`decode(encode(text)) != text`。如果我没加那句断言，
这个 bug 会一路溜到模型训练里，表现为「模型偶尔漏掉标点」。

**修法**（两处，缺一不可）：
1. 正则加标点兜底分支 `| ?[^\s\w]++[\r\n]*`（`++` = 一次或多次）
2. **`pre_tokenize` 里加断言**：
   ```python
   assert b"".join(blocks) == text.encode("utf-8"), "预切分正则有字符没匹配上"
   ```
   这行断言是本章最值得抄走的东西。**任何「切分」函数都该有覆盖率检查。**

#### 两个坑的共同教训

| 坑 | 表面症状 | 真实原因 | 防御手段 |
|---|---|---|---|
| A | 压缩率偏低 | 符号表示有歧义 | round-trip 测试（encode→decode 必须等于原文） |
| B | 解码对不上 | 切分有遗漏 | 覆盖率断言（切分结果拼起来必须等于原文） |

**两者的防御手段是同一个：加一个「往返一致性」断言。**
这是本教程反复出现的主题 —— 形状对了不等于数值对（第 01 章的 SDPA 布局坑
也是同一类）。

---

## ★ 消融实验

本章的消融：**预切分正则的松紧程度**。

`tiktoken` 的编码器一旦训练完成，正则是**焊死**的（存在 tokenizer.json 里）。
想改就必须重训。所以这个消融的成本是「重训 tokenizer + 重训模型」。

| 变体 | 做法 | 预期 |
|---|---|---|
| 数字 `{1,2}`（默认） | — | 基线 |
| 数字 `{1,3}` | 改 `SPLIT_PATTERN` | 词表被数字 token 占据，压缩率略升但通用 token 变少，**小词表下更差** |
| 数字 `{1,1}` | 改 `SPLIT_PATTERN` | 数字全是单 token，序列变长，注意力计算量上升 |
| 不预切分字母前缀 | 去掉 `[^\r\n\p{L}\p{N}]?+` | 标点和词分离，序列变长 |

**做第一个**（成本最低，最能说明问题）：

```bash
cp src/data/tokenizer.py scratch/tokenizer.py.bak
# 把 \p{N}{1,2} 改成 \p{N}{1,3}
sed -i 's/\\p{N}{1,2}/\\p{N}{1,3}/' src/data/tokenizer.py

rm -rf ~/.cache/gpt_base/tokenizer
uv run python -m training.train_tokenizer --vocab-size 8192 --shards 1
# 对比 bytes/token：默认约 3.24，{1,3} 会怎样？
```

**预期**：`{1,3}` 的 bytes/token 会**略高**（数字压得更紧），
但你会看到分词预览里出现 `202`、`026` 这类 token，
而通用的英文片段被挤掉了。**在 8192 这种小词表下净效果是负的。**
这就是 nanochat 作者实测后选 `{1,2}` 的原因。

---

## 常见坑

**坑 1：`Counter` 的键必须可哈希。**
用 `list` 当键会报 `TypeError: unhashable type: 'list'`。
必须用 `tuple`。（第 02 章的 `toy_bpe` 一开始就踩了这个。）

**坑 2：预切分和合并的边界搞混。**
最常见的错误写法是「先把全文合并，再预切分」——顺序反了。
正确顺序永远是：**先切块，再块内合并**。

**坑 3：推理时不用 merge 表，直接用训练时的 word 表。**
BPE 训练里的 `words` 只在那次语料上有效。
推理时文本是新的，必须靠 `self.merges` 这个「规则表」重新编码。

**坑 4：中文文本下纯 Python 版本会卡死。**
因为中文会被预切成很多块，每块都进 `_merge_all` 的双重循环。
玩具版只在英文小语料上用，别拿它去编码中文。

---

## 延伸

**nanochat / tiktoken 完整版怎么做的**

| | 玩具版（本项目 scratch） | 生产版（rustbpe + tiktoken） |
|---|---|---|
| 训练语料 | 4 MB 字符串 | 10 亿字符的生成器流 |
| 训练实现 | 纯 Python，O(V²) | Rust，并行 + 分块 |
| 推理 | 纯 Python，O(len²×rules) | Rust + SIMD，接近线性 |
| 预切分 | 简化正则 | GPT-4 级精调的 `SPLIT_PATTERN` |
| 字节兜底 | 有（从字节开始） | 有，且**跳过前 ~100 次合并** |
| 序列化 | JSON | 二进制的 `.tiktoken` / pickle |
| 中间表示 | 直接用 str | **`mergeable_ranks`: `bytes → rank`** |

**那个「中间表示」是整个设计的优雅之处**：
`mergeable_ranks` 是一个 `bytes → int` 的字典，同时表达了
「这个字节串是一个 token」和「它的合并优先级」。
tiktoken 只需要这张表就能编码，不需要训练过程。

**我们项目里 `BPETokenizer.train` 的最后一步就是这个转换**：
```python
mergeable_ranks = {bytes(k): v for k, v in t.get_mergeable_ranks()}
```
把 rustbpe 的输出翻译成 tiktoken 要的格式。

**关于「跳过前 100 次合并」**：
从字节开始训练时，前 100 条 merge 基本是「把 `b` 和 `e` 拼回 `be`」这类
在英文上没意义的操作。生产实现会跳过它们，直接从多字节符号开始。
本项目的 rustbpe 内部已经处理了。

---

## 下一章

[第 04 章：对话模板与特殊 token](04-对话模板与特殊token.md) ——
9 个特殊 token 如何定义了「模型怎么说话」，以及 SFT 的 loss mask 从哪来。
