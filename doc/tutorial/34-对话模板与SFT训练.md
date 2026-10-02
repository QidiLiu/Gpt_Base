# 34 · 对话模板与 SFT 训练

第 04 章讲过 loss mask 的**概念**。
本章讲它的**实现**，以及一个本项目记录在案的真实陷阱：
`truncate` 的方向选错会让整批样本的 loss 变成 NaN。

---

## 本章目标

- 逐行读懂 `render_conversation`，特别是 mask 的赋值规则
- 理解 `truncate="left"` 为什么是 SFT 的**必需**而非可选项
- 知道那个「f-string 拼特殊 token 会失败」的坑（`encode_ordinary`）

---

## 前置回顾

- [第 04 章](04-对话模板与特殊token.md)：对话模板和 loss mask 的设计
- [第 33 章](33-生成循环与终止控制.md)：`token_masks` 是生成结果，也是训练信号
- [第 02 章](02-为什么需要分词器.md)：tiktoken 的 `encode_ordinary` vs `encode_special`

> **本章是只读教程。** 代码在 `src/data/tokenizer.py` 的
> `render_conversation`、`src/training/train_sft.py`、
> `src/inference/engine.py` 的 `render_chat_prompt`。

---

## 概念：mask 的赋值规则

```python
def render_conversation(self, conversation: dict, max_tokens: int = 2048,
                        truncate: str = "right"):
    """
    把一段对话渲染成 (ids, mask)。

    mask[i] == 1  -> 第 i 个 token **参与 loss**
    mask[i] == 0  -> 第 i 个 token **不参与 loss**（只是上下文）

    这个 mask 是 SFT 的全部秘密：预训练时模型学「接着往下写」，
    但 SFT 只想让它学「assistant 该说什么」。所以 user 消息、BOS、
    各种结构性 token 全部 mask=0，只有 assistant 真正吐出的内容 mask=1。

    注意 python_output 段是 mask=0：那是引擎执行工具后塞回去的，
    不是模型自己产生的，不该训练模型去猜。
    """
```

**规则很简单**：

| token | mask |
|---|---|
| `<|bos|>` | 0 |
| `<|user_start|>` / `<|user_end|>` | 0 |
| user 的内容 | 0 |
| `<|assistant_start|>` | 0 |
| **assistant 的内容** | **1** |
| `<|assistant_end|>` | 0 |
| `<|python_start|>` / `<|output_start|>` | 0 |
| **python 的代码** | **1**（模型自己生成的） |
| **工具输出** | **0**（环境产生的） |

⚠ 最后三行是容易搞混的：**python 代码是模型写的（mask=1），
工具输出是引擎执行的（mask=0）**。

### 辅助函数 `add`

```python
def add(token_ids, mask_val: int):
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    ids.extend(token_ids)
    mask.extend([mask_val] * len(token_ids))
```

**一个 `mask_val` 广播到整段。** 这是整个 mask 构造的核心 ——
每次调用「这一整段的 mask 都是 X」。

⚠ `isinstance(token_ids, int)` 的检查是必要的 ——
`encode_special` 返回 list，但 `bos_token_id` 是 int。

### system 消息合并

```python
messages = conversation["messages"]
# system 消息合并进后面的 user 消息（不支持独立 system 段）
if messages[0]["role"] == "system":
    messages = copy.deepcopy(messages)
    assert messages[1]["role"] == "user", "system 之后必须是 user"
    messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
    messages = messages[1:]
```

**为什么用 `copy.deepcopy`**：`messages` 来自调用方，
直接改会污染原始数据（**同一份数据可能被渲染多次**
—— 评测和训练各一次）。

⚠ **这是一个真实的 bug 来源**：如果只做浅拷贝或者直接改 list，
评测阶段的渲染会看到已经被改过的内容。

`engine.py` 的 `render_chat_prompt` 做了同样的事，
而且两条路径必须**保持一致**（判据：
`test_render_chat_prompt_merges_system_into_user`）。

### 角色交替断言

```python
expected = "user" if i % 2 == 0 else "assistant"
assert message["role"] == expected, ...
```

**强制 user/assistant 严格交替。**

这意味着**本项目的对话模板不支持**：
- 连续两条 assistant
- assistant 连续说 tool call 然后 tool result 再 assistant（不是交替）

⚠ 那怎么支持多轮工具调用？答：**一条 assistant 消息的 content 里
同时包含 `<|python_start|>...<|output_start|>...` 段**。
所以在「消息」层面仍然是 user/assistant 交替的。

---

## 概念：★★ truncate 方向 —— 一个真实的 NaN 陷阱

这是本章最重要的内容，docstring 里记着：

> `truncate` 参数：一个必须踩过一次才明白的坑
>
> **"right"（从尾部切掉多余的）—— 用于预训练/评测，保留开头**
> **"left"（从头部切掉多余的）—— 用于 SFT，保留结尾**
>
> 为什么 SFT 必须用 left？
>
> 对话很长时（>max_tokens），从尾部截断会把**最后一条 assistant
> 回复整段切掉**，只剩 user 的问题 —— 这时 mask 全是 0，
> 交叉熵在 ignore_index 全命中时算出 0/0 = **NaN**。
> 实测 512 长度的 SFT 数据里有 23% 的样本落入这个陷阱。

### 实测复现

```python
import sys, torch
sys.path.insert(0, "src")
from data.tokenizer import get_tokenizer

tok = get_tokenizer()

def probe(long_user, long_asst, turns=2, MAX=512):
    msgs = []
    for t in range(turns):
        msgs.append({"role": "user", "content": f"Q{t}. " + "u" * long_user})
        msgs.append({"role": "assistant",
                     "content": "A. " + "a" * (long_asst if t == turns - 1 else 20)})
    return {trunc: (lambda r: (len(r[0]), sum(r[1])))(
                tok.render_conversation({"messages": msgs},
                                        max_tokens=MAX, truncate=trunc))
            for trunc in ("right", "left")}

print(f"{'user 长':>8} {'asst 长':>8} {'right len':>11} {'right mask1':>12} "
      f"{'left len':>10} {'left mask1':>11}")
for lu in (200, 600, 1200):
    r = probe(lu, 30)
    print(f"{lu:>8} {30:>8} {r['right'][0]:>11} {r['right'][1]:>12} "
          f"{r['left'][0]:>10} {r['left'][1]:>11}")
```

实测（`max_tokens=512`）：

```
  user 长   asst 长   right len   right mask1   left len   left mask1
       200        30         469            56        469           56
       600        30         512             0        512           33
      1200        30         512             0        512           33
```

**`user 长=600` 那一行：`truncate="right"` 的 mask1 变成 0。**

原因很直白：prompt 本身就超过 512 了，从尾部切 512 个 token
**根本切不到 assistant 那一段** —— 剩下的全是 user 的问题和
BOS，全部 `mask=0`。

而 `truncate="left"` 保留了尾部 512 个 token，
**其中有 33 个是 assistant 的内容** → mask1 = 33。

### NaN 的确切来源

```python
import torch, torch.nn.functional as F
logits = torch.randn(8, 100)
t = torch.full((8,), -1, dtype=torch.long)
ce = F.cross_entropy(logits, t, ignore_index=-1, reduction="mean")
print(ce.item(), torch.isnan(ce).item())
```

实测：

```
nan True
```

**`reduction="mean"` + 全部 target 被 ignore = NaN。**

而 `reduction="sum"` 给 0.0、`reduction="none"` 给全 0 向量。

⚠ **我第一次测这个时写错了**：用了 `targets = torch.zeros(4)`
（值是 **0** 不是 **-1**），所以 `ignore_index=-1` 一个都没忽略，
算出来是正常的 1.87。**差点据此去改一条正确的 docstring。**

这就是本项目「单步测量不可信」的另一个变种 ——
**测试本身的 bug 会导致你质疑正确的代码**。

### `gpt.py` 里的 NaN 兜底

正因为会 NaN，`GPT.forward` 有这段：

```python
elif loss_reduction == "mean" and not torch.isfinite(loss):
    # 整批 target 都被 ignore（都是 -1）时，F.cross_entropy 算的是 0/0 = NaN。
    # 这通常意味着上游的 mask 逻辑出了 bug（例如截断把监督信号全切掉了）。
    # 这里降级成 0 而不是让 NaN 污染全部权重 —— 后者不可恢复。
    log0("  [警告] 本批没有有效的监督 token（targets 全为 -1），loss 记为 0。"
         "通常是截断策略把 assistant 部分切掉了，检查 truncate 参数。")
    loss = logits.sum() * 0.0
```

**`logits.sum() * 0.0` 不是「= 0」这么简单** ——
它的梯度是 `0`（因为 `d/dx (x·0) = 0`），
而**如果写成 `torch.tensor(0.0)` 就没有梯度**，
`backward()` 会因为「没有需要梯度的输出」而报错或者什么都不做。

**这个技巧叫「乘 0 保梯度」。** 本项目用它来
「让参数完全不动，但保持计算图完整」。

⚠ **NaN 一旦进梯度就会污染全部权重且不可恢复**
（AdamW 的动量里也有 NaN，会一直传播）。所以这个兜底是必须的。

### 关于那个 23%

我的合成数据（user 长 50-1200 字符、2-5 轮、`max_tokens=512`）
测出 **192/300 = 64%** 的样本 mask 全为 0。

**这和 docstring 的 23% 量级相同但不相同** —— 因为我的长度分布是
我编的，真实 SFT 数据（smoltalk）的对话长度分布不同。

**诚实的说法**：
- **机制已验证**（`user 长 ≥ 600` 时 `truncate="right"` 必然 mask 全 0）
- **23% 这个具体数字依赖数据分布**，我没有真实数据可复现

---

## 概念：★ `encode_ordinary` 会忽略特殊 token

`engine.py` 的 `render_chat_prompt` docstring 记着这个坑：

> **★ 为什么不能 f-string 拼特殊 token**
>
> `BPETokenizer.encode()` 内部走的是 tiktoken 的 **`encode_ordinary`**，
> 而 `encode_ordinary` 的定义就是「**忽略** special tokens」——
> 它会把 `"<|user_start|>"` 当成普通文本再切一遍。
>
> 实测（8192 词表的真实 tokenizer，文本 "What is 12 * 7?"）：
>
> ```
> 错：tokenizer.encode(f"<|user_start|>{q}<|user_end|><|assistant_start|>")
>     -> 34 个 token，特殊 id 8184 **根本没出现**，
>        被切成 [60, 124, 305, 263, 95, 321, ...]
> 对：render_chat_prompt(tok, q)
>     -> 12 个 token，8184 / 8185 / 8186 各出现 1 次
> ```
>
> 两边 decode 回文本**完全一样**，所以这个 bug 在日志里看不出来 ——
> 但训练时（`render_conversation` 用 `encode_special`）模型只见过 1 个 id，
> 评测时喂 12 个，格式对不上，准确率因此失去意义。

**这是一个「日志里完全看不出来」的 bug**，因为
`decode(encode(text)) == text` 恒成立（可逆分词的定义）。

**唯一的检测方法是检查 token id 序列本身。**

### 正确写法

```python
def render_chat_prompt(tokenizer, user_text, system=None):
    S = tokenizer.encode_special
    ids = [tokenizer.get_bos_token_id()]
    if system:
        # system 合并进 user（与 render_conversation 的约定一致）
        user_text = f"{system}\n\n{user_text}"
    ids += [S("<|user_start|>")]
    ids += tokenizer.encode(user_text)
    ids += [S("<|user_end|>"), S("<|assistant_start|>")]
    return ids
```

**关键**：`user_text` 用 `encode`（普通文本），
特殊 token 用 `encode_special`（逐个取 id），**最后拼 id 列表**。

⚠ **注意 `user_text` 里如果 `system` 拼接了带特殊 token 的内容
就会再次踩坑。** `system` 提示必须是纯文本。

### 两条路径必须一致

| | 训练 | 评测 |
|---|---|---|
| 函数 | `tokenizer.render_conversation` | `engine.render_chat_prompt` |
| 特殊 token 怎么取 | `encode_special` | `encode_special` |
| system 怎么处理 | 合并进 user | 合并进 user |

⚠ **两处必须完全一致**，否则模型训练时见到的格式和推理时不一样。
判据：`test_render_chat_prompt_uses_real_special_tokens`
和 `test_render_chat_prompt_merges_system_into_user`。

---

## 读这段代码时注意什么：`load_sft_model` 的回落

```python
def load_sft_model(mode, tag, device):
    """优先加载 SFT 后的模型；没跑过 SFT 就回落到基座。"""
    base_tag = tag or default_tag(mode)
    sft_dir = os.path.join(runs, "sft_checkpoints", base_tag)
    tokenizer = get_tokenizer()

    if os.path.isdir(sft_dir) and find_latest(sft_dir) is not None:
        step = find_latest(sft_dir)
        meta = load_checkpoint(sft_dir, step, "cpu", "meta")
        log0(f"载入 SFT 模型 {base_tag} (step {step})")
    else:
        log0(f"未找到 SFT 模型，回落到基座 {base_tag}"
             f"（先跑 bash script/train_sft.sh {mode}）")
        from common.checkpoint import load_latest
        model, tokenizer, meta = load_latest(runs, base_tag, device)
        return model, tokenizer, base_tag

    fields = set(ModelConfig.__dataclass_fields__)
    cfg = ModelConfig(**{k: v for k, v in meta["model_config"].items() if k in fields})
    model = build_model(cfg, device=device)
    model.load_state_dict(load_checkpoint(sft_dir, step, device, "model"))
    return model, tokenizer, base_tag
```

**这是一个很实用的设计**：没跑过 SFT 就用基座，
而不是报错。

⚠ 但有个隐患：**基座模型没接受过对话模板训练**，
所以对话质量会很差（它只是「接着往下写」）。
**日志里的「回落到基座」是唯一的提示。**

### `ModelConfig(**{...})` 的过滤

```python
fields = set(ModelConfig.__dataclass_fields__)
cfg = ModelConfig(**{k: v for k, v in meta["model_config"].items() if k in fields})
```

**从 checkpoint 的 meta 里恢复配置，但过滤掉不认识的字段。**

⚠ 这是**前向兼容**的做法 —— 老 checkpoint 的 meta 里
如果有后来被删掉的字段，不会因为 `ModelConfig` 变了而加载失败。

**但它也是「静默忽略」的来源** —— 如果 meta 里有个拼错的字段名
（比如 `n_layer` 写成 `n_layers`），它会被静默丢掉，
然后 `ModelConfig` 用默认值 —— **模型形状错了，
`load_state_dict` 会报 size mismatch**（还算好）。

---

## 动手验证

### 验证 1：truncate 方向的实测

见上面那张表。**关键是 `user 长=600` 那一行：`right` 的 mask1 = 0。**

### 验证 2：NaN 的确切来源

```python
import torch, torch.nn.functional as F
for red in ("mean", "sum", "none"):
    logits = torch.randn(8, 100)
    t = torch.full((8,), -1, dtype=torch.long)
    r = F.cross_entropy(logits, t, ignore_index=-1, reduction=red)
    print(f"reduction={red:5s} -> {r if r.numel()>1 else r.item()}  "
          f"is_nan={torch.isnan(r).any().item()}")
```

实测：

```
reduction=mean  -> nan  is_nan=True
reduction=sum   -> 0.0  is_nan=False
reduction=none  -> torch.Size([8])  is_nan=False
```

### 验证 3：`encode_ordinary` 忽略特殊 token

```bash
uv run python -c "
import sys; sys.path.insert(0,'src')
from data.tokenizer import get_tokenizer
tok = get_tokenizer()
q = 'What is 12 * 7?'
S = tok.encode_special

wrong = tok.encode(f'<|user_start|>{q}<|user_end|><|assistant_start|>')
print('f-string 拼接 :', len(wrong), '个 token')
print('  特殊 id 出现?', S('<|user_start|>')[0] in wrong)

right = [tok.get_bos_token_id(), S('<|user_start|>')] + tok.encode(q) \
        + [S('<|user_end|>'), S('<|assistant_start|>')]
print('正确拼接      :', len(right), '个 token')
print('  特殊 id 出现?', S('<|user_start|>')[0] in right)
print()
print('两边 decode 回文本一样?', tok.decode(wrong) == tok.decode(right))
print('-> 所以这个 bug 在日志里完全看不出来。')
"
```

**期望**：f-string 版 token 数明显更多、特殊 id 不出现，
但 decode 出来一样。

### 验证 4：两条渲染路径的一致性

```bash
uv run pytest tests/test_engine.py -k "render or chat_prompt" -v
uv run pytest tests/test_data.py -k "mask or truncate or alternation" -v
```

---

## 常见坑

### 坑 1：`truncate="right"` 用在 SFT 上

**这是本章的主题。** 会让整批样本 mask 全 0 → loss NaN
→ 被兜底降级成 0 → **那一步等于白跑**。

⚠ 而兜底会打印警告 —— **如果你没看日志，就完全不知道
有多少步是白跑的**。

### 坑 2：f-string 拼特殊 token

见上面。**日志里完全看不出来。**

### 坑 3：直接改 `messages` 不用 `deepcopy`

会污染调用方的数据（评测和训练可能各渲染一次）。

### 坑 4：以为 NaN 兜底是「= 0 就够了」

**必须用 `logits.sum() * 0.0`** ——
它保留了计算图，梯度是 0（`d/dx(x·0) = 0`）。
写 `torch.tensor(0.0)` 就断了梯度。

### 坑 5：测 NaN 时把 target 写成 0 而不是 -1

**我自己踩了。** `ignore_index=-1` 遇到 0 一个都忽略不到，
结果正常，于是差点去改一条正确的 docstring。

### 坑 6：没跑过 SFT 却以为对话能用

`load_sft_model` 会回落到基座，日志有一行提示。
**基座模型没接受过对话模板训练**，输出会很差。

---

## 延伸

**为什么 tool output 不进 loss**

再强调一次（这是 RLHF 的标准做法）：

1. **不可预测** —— 函数调用结果由环境决定
2. **信息泄漏** —— 模型能背出工具输出就不需要真的调用
3. **长度不可控** —— 会扰乱 batch

**但 python 代码要进 loss** —— 模型需要学会「什么时候调工具、
调什么参数」。

所以 mask 的分界是「**模型产生的 vs 环境产生的**」，
不是「不可预测的 vs 可预测的」。

**smoltalk 数据集**

本项目 SFT 用 `HuggingFaceTB/smotalk`（在
`~/.cache/gpt_base/task_data/` 里能看到）。

它是一个「多来源混合」的指令数据集合。`render_conversation`
只需要 `{"messages": [{"role":..., "content":...}]}` 这个结构 ——
所以任何 chat 格式的数据集都能用。

**多轮对话的长度分布决定 NaN 率**

本章实测的 64% vs docstring 的 23%，差别完全来自
「user 消息有多长 / 有多少轮」。

**这意味着 NaN 率是一个数据属性，不是代码属性** ——
换数据集就要重新评估。而**代码层没有任何防护**
（只有 `GPT.forward` 那个事后兜底）。

**更彻底的做法**：在 `render_conversation` 里就检测
「截断后 mask 全 0」并**直接丢弃这条样本**，
或者至少**记一个计数器**。

⚠ 本项目**没有**这个检测。**只有日志里逐 batch 的警告。**

---

## 下一章

[第 35 章：评测与指标](35-评测与指标.md) ——

前面 30 章讲怎么训。本章讲怎么**判断训得怎么样**。

三个指标，三种口径：
- **bpb**（第 06 章讲过原理）：唯一跨模型可比的
- **pass@k**：采样多样性 vs 正确率
- **多选题**：一个容易实现错的「字符串还原」问题

其中 pass@k 有一个**极易写错**的点：
`c` 是总采样数，不是前缀长度。
