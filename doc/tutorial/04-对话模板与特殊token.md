# 04 · 对话模板与特殊 token

## 本章目标

- 看懂 `render_conversation` 输出的 `(ids, mask)` 这一对东西
- **彻底理解 SFT 的 loss mask 是怎么来的**，以及一个 bug 会造成多大伤害
- 掌握一个必须知道的陷阱：训练目标和推理格式必须 **token 级一致**

---

## 前置回顾

[第 03 章](03-手写一个玩具BPE.md) 我们得到了一个能用的 tokenizer。
现在用它来处理「对话」这种比文档更复杂的数据。

---

## 概念：9 个特殊 token 各自解决什么问题

代码位置：`src/data/tokenizer.py: SPECIAL_TOKENS`

```python
SPECIAL_TOKENS = [
    "<|bos|>",              # 序列/文档开始
    "<|user_start|>",       # user 消息开始
    "<|user_end|>",
    "<|assistant_start|>",  # assistant 消息开始
    "<|assistant_end|>",
    "<|python_start|>",     # assistant 调用 python 工具
    "<|python_end|>",
    "<|output_start|>",     # 工具输出回传给 assistant
    "<|output_end|>",
]
```

**第一个关键设计：为什么需要 `<|user_start|>` / `<|assistant_start|>` 这类标记？**

因为预训练语料是纯文本，「谁在说话」这个信息是**隐含**的
（从换行、缩进、说话人名字里猜）。但对话模型需要这个信息是**显式**的。

用特殊 token 的好处是：
- **一个 token 就能表达**「角色切换」，不会和内容里的文本混淆
- 推理时模型采样到 `<|assistant_start|>` 就知道「该我说话了」，
  采样到 `<|assistant_end|>` 就知道「我说完了」

**第二个关键设计：`<|bos|>` 放最前，而且预训练时就在用。**

`dataloader.py` 的 best-fit 装箱算法**依赖「每行以 BOS 开头」这个不变量**。
所以 BOS 不是「对话专用」的 token，它是「序列开始」的通用标记。

**第三个关键设计：4 个 python/output token 构成一个「工具调用协议」。**

```
<|assistant_start|> 我算一下 <|python_start|> 12*7 <|python_end|> <|output_start|> 84 <|output_end|> 答案是 84 <|assistant_end|>
                      ↑ 模型生成        ↑ 引擎执行这段            ↑ 引擎把结果塞回去
```

引擎（`src/inference/engine.py`）里跑一个状态机：
1. 模型采样到 `<|python_start|>` → 开始累积后面的 token
2. 遇到 `<|python_end|>` → 把累积的 token 解码成字符串，交给计算器
3. 计算器返回结果 → 引擎**强制注入** `<|output_start|> 结果 <|output_end|>`
4. 模型继续生成

**为什么要强制注入而不是让模型自己「猜」结果？**
因为小模型算数很不可靠。工具调用的价值就在于**把计算外包给可靠的外部程序**。
这也解释了为什么 `python_output` 段的 mask 是 0 ——
那不是模型产生的，训练时不该让模型学去猜结果（本章的 mask 表已列出）。

---

## 概念：loss mask —— SFT 的全部秘密

这是本章的核心，也是整个卷 7 的地基。

`render_conversation` 返回两个列表：

```python
ids  = [所有 token 的 id]
mask = [每个位置是否参与 loss，1=参与，0=不参与]
```

渲染一段对话：
```
<|bos|><|user_start|>今天天气<|user_end|><|assistant_start|>不错<|assistant_end|>
 mask:  0      0         0        0        0         0      1        1        0
                            └─ 全是 user 侧，不学 ─┘      └─ 学这个 ┘  └─ 收尾也学 ┘
```

**为什么 user 侧不参与 loss？**

因为训练目标不同：

| 阶段 | 目标 | 损失函数 |
|---|---|---|
| 预训练 | 「无条件地接着写下一个 token」 | 所有位置都算 |
| SFT | 「**轮到 assistant 时，assistant 该说什么**」 | 只有 assistant 的 token 算 |

如果在 SFT 时也把 user 侧算进 loss，模型会同时学两件事：
「怎么提问」+「怎么回答」。前者是浪费（推理时没人替它提问），
后者还会被稀释。

**还有个更微妙的原因**：推理时 prompt 是**已知的**，
模型只需要生成 assistant 部分。所以「给定上文预测下一个 token」
这个能力在 prompt 部分已经无用武之地。

---

## 概念：陷阱 —— 训练目标必须 token 级一致

这是 nanochat 代码注释里专门强调的一个坑，值得单独讲。

**场景**：多选题的 prompt 渲染。

```python
def render_mc(question, letters, choices):
    q = f"Multiple Choice question: {question}\n"
    q += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    q += "\nRespond only with the letter of the correct answer."
    return q
```

注意格式是 `- Paris=F`，**等号和字母之间没有空格**。为什么？

```
BPE 分词时：  "- Paris=F"  ->  ["- Paris", "=F"]
              "- Paris=F " -> ["- Paris", "=F", " "]

但如果写成 "- Paris = F"：
              "- Paris = F" -> ["- Paris", " =", " F"]   # " F" 是另一个 token！
```

而 assistant 回答时只会输出 `F`（不带空格）。
所以：
- prompt 里的 `F` 前面没有空格 → token 是 `"F"`（假设 id=100）
- 模型的回答是 `F` → token 也是 `F`（id=100）
- ✅ 模型学到「输出 token 100 就对应回答 F」

如果 prompt 写成 `= F`：
- prompt 里的 `F` 前面有空格 → token 是 `" F"`（id=500）
- 模型的回答是 `F` → token `F`（id=100）
- ❌ **模型永远学不会这个映射**，因为它从来没见过「F 是 100 而不是 500」

**这就是「小模型对格式细节极其敏感」的根源。**
大模型有足够容量去「理解」这种不一致，小模型没有 ——
它只能靠统计相关性，而统计在这里是矛盾的。

代码位置：`src/data/tasks.py: render_mc`，注释里写了这条。

---

## 手抄代码

完整实现在 `src/data/tokenizer.py: render_conversation`。核心逻辑：

### 第 1 块：一个能处理工具调用的消息渲染器


<details>
<summary><b>👀 展开参考答案（先自己想 20 分钟）</b></summary>

```python
def render_conversation(self, conversation: dict, max_tokens: int = 2048,
                        truncate: str = "right"):
    """
    把一段对话渲染成 (ids, mask)。

    mask[i] == 1  -> 第 i 个 token **参与 loss**
    mask[i] == 0  -> 第 i 个 token **不参与 loss**（只是上下文）

    ── truncate 参数：一个必须踩过一次才明白的坑 ──────────────
      "right"（从尾部切掉多余的）—— 用于**预训练/评测**，保留开头
      "left" （从头部切掉多余的）—— 用于 **SFT**，保留结尾

      为什么 SFT 必须用 left？
        对话很长时（>max_tokens），从尾部截断会把**最后一条 assistant
        回复整段切掉**，只剩 user 的问题 —— 这时 mask 全是 0，
        交叉熵在 ignore_index 全命中时算出 0/0 = **NaN**。
        实测 512 长度的 SFT 数据里有 23% 的样本落入这个陷阱。
    """
    assert truncate in ("right", "left")
    ids, mask = [], []

    def add(token_ids, mask_val: int):
        """往输出里追加 token，并给每个位置打上 mask 标签。"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))

    messages = conversation["messages"]
    # system 消息合并进后面的 user 消息（不支持独立 system 段）
    if messages[0]["role"] == "system":
        messages = copy.deepcopy(messages)
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]

    S = self.encode_special
    bos, user_s, user_e = self.bos_token_id, S("<|user_start|>"), S("<|user_end|>")
    asst_s, asst_e = S("<|assistant_start|>"), S("<|assistant_end|>")
    py_s, py_e = S("<|python_start|>"), S("<|python_end|>")
    out_s, out_e = S("<|output_start|>"), S("<|output_end|>")

    add(bos, 0)                       # BOS 不参与 loss
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == expected, (
            f"第 {i} 条消息来自 {message['role']}，但按交替规则应该是 {expected}")
        content = message["content"]

        if message["role"] == "user":
            add(user_s, 0)            # 全部 mask=0
            add(self.encode(content), 0)
            add(user_e, 0)
        else:
            add(asst_s, 0)            # assistant_start 本身 mask=0
            if isinstance(content, str):
                add(self.encode(content), 1)          # ← 唯一 mask=1 的地方
            elif isinstance(content, list):
                for part in content:
                    vids = self.encode(part["text"])
                    if part["type"] == "text":
                        add(vids, 1)                   # assistant 说的字
                    elif part["type"] == "python":     # assistant 调工具
                        add(py_s, 1); add(vids, 1); add(py_e, 1)
                    elif part["type"] == "python_output":   # 引擎塞回来的
                        add(out_s, 0); add(vids, 0); add(out_e, 0)   # ← mask=0
            add(asst_e, 1)            # assistant_end 也学：教会模型何时停

    if truncate == "right":
        return ids[:max_tokens], mask[:max_tokens]
    return ids[-max_tokens:], mask[-max_tokens:]     # left: 保留结尾
```

</details>

**注意三处 mask 的取值**：

| 位置 | mask | 理由 |
|---|---|---|
| `<|assistant_start|>` | **0** | 它是「轮到你说了」的信号，不是「说的话」 |
| assistant 的内容 | **1** | 这才是要学的 |
| `<|assistant_end|>` | **1** | 模型必须学会何时停止 |
| `python_output` 段 | **0** | 引擎生成的，模型不该学去猜 |

`<|assistant_end|>` 用 mask=1 是个精妙的设计：
如果它是 0，模型永远学不会「什么时候该停」，
推理时就会一直说下去停不下来。

---

## 动手验证

### 验证 1：看一条简单对话的 mask

新建 `scratch/mask_demo.py`：

> 🔒 **需要先完成**：卷 1 第 04 章（`render_conversation`）
>
> 现在跑到这步会得到 `NotImplementedError: 待实现：xxx` ——
> 那是「你还没写这一块」，不是环境坏了。想跳过：`git checkout solution`。

> 📦 `solution` 分支的 `scratch/mask_demo.py` 已经含验证 1-3，直接跑即可。
> 下面三块是它的逐段拆解，供你从零手敲时对照。

```python
import torch
import torch.nn.functional as F
from data.tokenizer import get_tokenizer

tok = get_tokenizer()
conv = {"messages": [
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]}

ids, mask = tok.render_conversation(conv)
print(f"{'id':>6} {'mask':>5}  token")
print("-" * 40)
for i, (t, m) in enumerate(zip(ids, mask)):
    print(f"{t:6d} {m:5d}  {tok.decode([t])!r}")

print(f"\n参与 loss 的 token: {sum(mask)} / {len(mask)}")
learned_ids = [t for t, m in zip(ids, mask) if m == 1]
print(f"它们拼起来是: {tok.decode(learned_ids)!r}")
# 注意监督信号里**包含** <|assistant_end|>（mask=1 是刻意的）：
# 模型必须学会「什么时候停」，所以收尾 token 也要学。
assert tok.decode(learned_ids).startswith("Paris.")
assert learned_ids[-1] == tok.encode_special("<|assistant_end|>")
print("[OK] 监督信号 = assistant 的回答 + <|assistant_end|>（教模型何时停）")
```

跑：`uv run python scratch/mask_demo.py`

**实测输出**：

```
   id  mask  token
----------------------------------------
  8183     0  '<|bos|>'
  8184     0  '<|user_start|>'
  1112     0  'What'
   306     0  ' is'
   262     0  ' the'
  6415     0  ' capital'
   285     0  ' of'
  7183     0  ' France'
    63     0  '?'
  8185     0  '<|user_end|>'
  8186     0  '<|assistant_start|>'
    80     1  'P'
   287     1  'ar'
   274     1  'is'
    46     1  '.'
  8187     1  '<|assistant_end|>'

参与 loss 的 token: 5 / 16
它们拼起来是: 'Paris.<|assistant_end|>'
[OK] 监督信号 = assistant 的回答 + <|assistant_end|>（教模型何时停）
```

**逐行核对**：`<|bos|>` `<|user_start|>` 整个 user 段 `<|user_end|>`
`<|assistant_start|>` 共 11 个 token 全是 0；`P ar is .` 和 `<|assistant_end|>`
共 5 个是 1。

注意 `'Paris'` 被切成了 `P|ar|is` 三个 token —— 因为它是我们临时写的
短文本，训练语料里没出现过这个组合，tokenizer 只能退回子词。
这正是第 02 章说的「BPE 的合并取决于训练语料」。

### 验证 2：看带工具调用的对话

追加到 `scratch/mask_demo.py`：

> 🔒 **需要先完成**：卷 1 第 04 章
>
> 现在跑到这步会得到 `NotImplementedError: 待实现：xxx` ——
> 那是「你还没写这一块」，不是环境坏了。想跳过：`git checkout solution`。

```python
# ── 带工具调用的对话 ──
print("\n" + "=" * 60)
print("带工具调用的对话（GSM8K 的格式）")
print("=" * 60)
conv2 = {"messages": [
    {"role": "user", "content": "Mimi has 2 x 12 = "},
    {"role": "assistant", "content": [
        {"type": "text",           "text": "2*12"},
        {"type": "python_output",  "text": "24"},
        {"type": "text",           "text": " sea shells."},
    ]},
]}
ids2, mask2 = tok.render_conversation(conv2)
print(tok.visualize(ids2, mask2))
print(f"\n绿=参与loss(1)  红=不参与(0)")
learned = tok.decode([t for t, m in zip(ids2, mask2) if m == 1])
print(f"模型要学的: {learned!r}")
assert "24" not in learned, "python_output 的结果不该被学（它是引擎算的）"
assert "2*12" in learned and "sea shells" in learned
print("[OK] 工具调用表达式和自然语言要学，工具返回值不学")
```

**观察重点**：`24`（工具返回值）是**红色**，`2*12` 和 ` sea shells.` 是**绿色**。

这个设计意味着：
- 模型学会「遇到计算 → 输出 `<|python_start|>表达式<|python_end|>`」
- 但模型**不学**「算完之后结果是多少」——那是引擎的活

这样训练出来的模型很「诚实」：它不会假装自己会算数，而是真的去调工具。

### 验证 3：亲眼看到「截断把监督信号全切掉」

追加到 `scratch/mask_demo.py`：

> 🔒 **需要先完成**：卷 1 第 04 章
>
> 现在跑到这步会得到 `NotImplementedError: 待实现：xxx` ——
> 那是「你还没写这一块」，不是环境坏了。想跳过：`git checkout solution`。

> ⚠️ 本块用到 `torch` 和 `F`，但**这两行 import 在验证 1 建的文件顶部**
> （`import torch` / `import torch.nn.functional as F`）。
> 如果你是新建空文件直接贴本块，会得到 `NameError: name 'torch' is not defined`。

```python
# ── 验证 3：亲眼看到「截断把监督信号全切掉」──
print("\n" + "=" * 60)
print("截断策略：为什么 SFT 必须 truncate='left'")
print("=" * 60)
long_conv = {"messages": [
    {"role": "user", "content": "问题" * 200},        # 超长 user
    {"role": "assistant", "content": "这是一个很长很长的回答。" * 40},
]}

for mode in ("right", "left"):
    ids3, mask3 = tok.render_conversation(long_conv, max_tokens=128, truncate=mode)
    n_sup = sum(mask3)
    # 模拟 GPT.forward：右移一位，mask=0 的位置置 -1
    V = 10000
    tgt = torch.tensor([i - 1 for i in ids3[1:]], dtype=torch.long)
    keep = torch.tensor([m for m in mask3[:-1]], dtype=torch.bool)
    tgt = torch.where(keep, tgt, torch.full_like(tgt, -1))
    loss = F.cross_entropy(torch.randn(len(tgt), V), tgt, ignore_index=-1)
    print(f"  truncate={mode:5s}: {len(ids3)} token, 参与loss {n_sup:4d} 个"
          f"  ->  实际 loss = {loss.item()}")
```

跑：`uv run python scratch/mask_demo.py`

**实测输出**：

```
截断策略：为什么 SFT 必须 truncate='left'
============================================================
  truncate=right: 128 token, 参与loss    0 个  ->  实际 loss = nan
  truncate=left : 128 token, 参与loss  128 个  ->  实际 loss = 9.62
```

**`truncate="right"` 时监督信号是 0 个**，因为 128 个 token 全部被
user 的超长问题占满，assistant 的回答被整段切掉了。
此时 `F.cross_entropy(..., ignore_index=-1, reduction="mean")` 算出 **0/0 = NaN**。

**这个 NaN 一旦进入反向传播，会污染全部参数，而且不可恢复。**

实测：在 512 长度的真实 SFT 数据上，`truncate="right"` 会让
**23% 的样本**（3000 条里的 699 条）落入这个陷阱。修法有三重：

1. `render_conversation(truncate="left")` —— 保留结尾（正确的修法）
2. dataloader 里 `if not any(mask[1:]): continue` —— 跳过（防御）
3. `GPT.forward` 里检测到非有限 loss 就降级为 0 并告警（兜底）

三重都在 `src/` 里实现了。**这一章的三个坑叠在一起，是我写这个项目时
遇到的最有教育意义的一串 bug。**

---

## ★ 消融实验

本章的消融是「`python_output` 段的 mask」：

| 变体 | 做法 | 预期 |
|---|---|---|
| mask=0（默认） | — | 基线。模型学会调工具，但不自夸会算数 |
| mask=1 | 改 `render_conversation` 里 `python_output` 段 | 模型同时学会「输出表达式」和「假装知道结果」。推理时它可能跳过调用直接编答案 |

**这个消融很有价值，因为它检验一个具体的主张**：
「不该让模型学工具的返回值」。

**怎么做**（会破坏 GSM8K 训练数据的一致性，所以只能短跑）：

```bash
cp src/data/tokenizer.py scratch/tokenizer.py.bak
# 把 python_output 三个 add(...) 的 mask_val 从 0 改成 1
sed -i 's/add(out_s, 0); add(vids, 0); add(out_e, 0)/add(out_s, 1); add(vids, 1); add(out_e, 1)/' src/data/tokenizer.py

bash script/train_sft.sh smoke --num-iterations 300
uv run python -m training.chat --mode smoke -p "What is 12 * 7?" -m 64

# ⚠ 一定要恢复，否则 render_conversation 会被永久改掉
cp scratch/tokenizer.py.bak src/data/tokenizer.py
```

**观察**：模型是「先编一个答案再调工具」还是「先调工具再回答」？
前者说明它学了「编造」；后者说明格式没学会。

**注意**：这个实验改变了训练数据的分布，所以 GSM8K pass@1 的绝对值不可比。
要看的是**行为模式**的差异。

---

## 常见坑

**坑 1：`truncate="right"` 导致的 NaN。**
本章验证 3 已经完整演示。**用 `truncate="left"`。**

**坑 2：mask 忘了同步。**
`add()` 函数存在的唯一理由就是「同时 extend ids 和 mask」。
如果哪里直接 `ids.extend(...)` 而没更新 mask，长度就对不上，
后面 `zip(ids, mask)` 会静默截断到较短的那个。

**坑 3：角色顺序假设。**
代码里 `expected = "user" if i % 2 == 0 else "assistant"`。
这假设对话严格交替。真实的 SmolTalk 数据里会有连续两个 user 消息
（多轮追问被合并），`data/tasks.py: SmolTalk.get_example` 里有合并逻辑处理这个。

**坑 4：把 `<|assistant_end|>` 的 mask 设成 0。**
后果是模型学不会停止，推理时一直说下去。
可以用 `bash script/chat.sh smoke -m 200` 观察：回复永远到不了自然结尾。

**坑 5：prompt 里的字母前面多了空格。**
要写 `"- Paris=F"`（等号和字母**之间不能有空格**），
写成 `"- Paris= F"` 就错了。见本章「陷阱」一节。
用 `uv run python -c "
from data.tokenizer import get_tokenizer
t = get_tokenizer()
for s in ['=F', '= F']:
    print(repr(s), t.encode(s))
"` 就能看到差异。

---

## 延伸

**nanochat 完整版怎么做的**

基本一致，两点细节：
- nanochat 的 `render_conversation` 只有 `max_tokens`，截断就是从右
  （`ids[:max_tokens]`）。但它没有这个问题，因为 nanochat 的 SFT 数据
  （SmolTalk）对话普遍较短，超过 2048 的很少
- nanochat 有一个 `visualize_tokenization()` 辅助函数用于调试，
  我们叫 `visualize()`，实现一样

**本项目多做的**：
- `truncate` 参数 + 三重 NaN 防护（因为我们用更短的 seq_len=512，
  23% 的样本会超长，这个坑必然踩到）
- `evaluate_bpb` 依赖 `token_bytes[y]` 的求和，所以我们额外提供了
  `decode_bytes(token_id)` 和 `token_bytes.pt`（第 06 章）

**一个值得知道的工业实践**：
很多模型会把 `python_output` 段也设成 mask=1，理由是「让模型学会
校对自己的计算结果」。但这需要一个足够强的模型。
对 1600 万参数的小模型，nanochat 的选择（mask=0）更稳 ——
因为它唯一能可靠学到的是「**格式**」，不是「正确性」。

---

## 下一章

[第 05 章：文档打包 —— BOS-aligned best-fit](05-文档打包BOS-aligned-bestfit.md) ——
变长文档怎么不浪费地拼成 GPU 能吃的矩形。这是卷 1 最有价值的一章。
