# 18 · Value Embeddings 与门控

上一章的两个 trick 加起来是 **48 个参数**。
本章这个 trick 加了 **1.51 亿个参数**，占 `full` 档的 **43.6%**。

先把这个反差摆出来 —— 它是本章要回答的核心问题。

> ⚠ **本章的数字按预设的 `V=16384` 算，而 2026-10 实测当前仓库
> 跑的是 `V=8192`**（`train_tokenizer` 的缓存不按词表大小分桶，
> 详见[第 37 章](37-全流程与排错.md)）。
>
> 所以本章所有「ve 表参数量」要减半：
> `16384 × 768 = 12,582,912` → **`8192 × 768 = 6,291,456`**，
> 12 张表从 1.51 亿变成 **7,549 万**。
> **方法论（为什么要独立 ve 表、低秩分解怎么省）不受影响。**

---

## 本章目标

- 说清 value embedding 补的是什么信息，为什么 v 是正确的注入点
- 理解「隔层 + 末层必有」这个布置，以及每层一张独立表的代价
- 亲手量出那 1.51 亿参数的来龙去脉

---

## 前置回顾

- [第 10 章](10-注意力三步曲.md)：注意力的五步，特别是 `v` 的角色
- [第 12 章](12-QK-Norm与GQA.md)：GQA 下 `n_kv_head` 可以小于 `n_head`
- [第 17 章](17-resid-lambdas与x0-lambdas.md)：残差流 trick 的共同框架

---

## 概念：注意力里丢掉了什么

先回到注意力本身。第 10 章写过：

```
q = W_q·x        查「我要找什么」
k = W_k·x        查「我能被什么找到」
v = W_v·x        查「我提供什么内容」
```

三个投影都是**同一个 x 的线性变换**。x 是残差流的内容 ——
经过若干层累积的「上下文状态」。

所以**注意力机制里有一条隐含的假设**：`v` 里的一切信息，
都来自残差流。**token 的身份信息不在这个假设的显式范围内。**

而残差流的第 0 层是 `rms_norm(wte[idx])`。理论上 token 身份在
第 0 层就在里面，往后每层都能读到。**为什么还需要单独补一路？**

### nanochat 作者的说法

源码里的注释直指浅层：

> 「注意力理论上能自己学出来，但在浅层很难。」

这是本章的动机。浅层（第 0、1 层）的残差流还几乎就是嵌入本身，
注意力权重还很粗糙。**要让它学会「先看前一个词」这种最基础的
关联，需要的步数比想象中多。**

把 token 身份直接混进 `v`，等于**跳过学习过程直接提供答案**。

### 为什么注入点是 v 而不是 q/k

| 注入位置 | 会发生什么 | 问题 |
|---|---|---|
| **q** | 「我是什么 token」影响「我要找什么」 | 语义错乱：找的能力被身份污染 |
| **k** | 「我是什么 token」影响「谁能找到我」 | 改变了可见性 —— 变成一种硬性的注意力偏置，且会影响 QK-Norm 的上界 |
| **v** | 「我是什么 token」成为**输出内容**的一部分 | ✓ 干净：检索逻辑不变，只是「被找到时提供的东西」多了一项 |

**v 是唯一正确的注入点**，因为注意力分数 `q·k` 完全不受影响。
第 12 章算过 QK-Norm 后 logits 的上界是 `1.2²×√64 = 11.52` ——
如果 value embedding 进了 k，这个上界就要重算。

**这是一个通用的设计原则**：给模型「补信息」时，优先补
**内容（value）** 而不是 **检索（q/k）**，因为内容变了不影响
「找到谁」。

---

## 概念：value embedding 的实现

### 一张普通的嵌入表，但每层一张

```python
# Value Embeddings：给隔层 + 末层的 attention 额外提供 token 身份信息
if cfg.use_value_embeds:
    kv_dim = cfg.n_kv_head * cfg.head_dim
    self.value_embeds = nn.ModuleDict({
        str(i): nn.Embedding(padded, kv_dim, device=device)
        for i in range(cfg.n_layer) if has_value_embed(i, cfg.n_layer)
    })
```

注意 `kv_dim` 而不是 `n_embd`。本项目默认 `n_kv_head == n_head`
（第 12 章），所以 `kv_dim == n_embd == 768` —— 但**开了 GQA 就会
不等**，那时 `ve` 的维度会跟着变小。

### forward：门控混合

```python
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

# Value Embedding（ResFormer）：把「token 本身的身份」按门控混进 value
if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    # 门控由输入的前 12 个通道决定，范围 (0, 3) —— 内容相关的门控
    gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    v = v + gate.unsqueeze(-1) * ve
```

三个设计选择，每一个都有理由：

**① `gate = 3·sigmoid(...)`，范围 (0, 3)**

不是 (0, 1) 是 (0, 3) —— **允许门完全打开（接近 3）**。
如果范围是 (0, 1)，value embedding 最多只能「加一点点」，
被 `c_v` 的输出压住。让它能到 3 倍意味着**在某些 token 上
identity 可以主导内容**。

**② 门由 `x[..., :12]` 决定 —— 只看前 12 个通道**

```python
self.ve_gate_channels = 12
self.ve_gate = Linear(self.ve_gate_channels, cfg.n_kv_head, bias=False, ...)
```

这是个**极轻量的条件门控**：768 维输入只看前 12 维，
输出 `n_kv_head` 个标量（每头一个）。

为什么只看前 12 维？**省参数**。若用全 768 维，门控矩阵是
`768 × 12 = 9216` 参数 × 12 层 = 11 万。用前 12 维则是
`12 × 12 = 144` × 12 层 = **1728** 参数。

为什么这个省法可接受？因为门只需要判断「这个位置要不要强调
自己的身份」—— 这是个**低维判断**，前 12 通道足够。
（这是启发式，没有理论保证。）

**③ `ve_gate` 的输出维度是 `n_kv_head`，不是 `kv_dim`**

一个标量管一个 kv 头的**全部维度**。即：
```
v[., h, :] = v[., h, :] + gate[., h] * ve[., h, :]
```
而不是逐通道门控。参数量 `12 × n_kv_head` 而非 `12 × kv_dim`。

### 初始化

```python
if cfg.use_value_embeds:
    for ve in self.value_embeds.values():
        torch.nn.init.uniform_(ve.weight, -s, s)
    for block in self.transformer.h:
        if block.attn.ve_gate is not None:
            torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
```

`ve` 的初值和 `c_v` 一样（`-s, s`，`s = √3·n_embd^-0.5`）——
所以 identity 一开始和 `c_v` 的输出**量级相当**，不是小扰动。

`ve_gate` 初始化在 `(0, 0.02)` 的**小正区间**，于是初始
`gate ≈ 3·sigmoid(≈0) = 3×0.5 = 1.5`。注释说
「gates start slightly above neutral」。

---

## 概念：「隔层 + 末层必有」的布置

```python
def has_value_embed(layer_idx: int, n_layer: int) -> bool:
    """
    Value Embedding 只加在「隔一层」和「最后一层」上。

    为什么不每层都加？
      残差流的容量有限，每加一路信号就多一份需要维护的信息。
      隔层加 + 末层必有，是 ResFormer 论文的折中：足够提供身份信息，
      又不至于淹没原始的上下文信号。
    """
    return layer_idx % 2 == (n_layer - 1) % 2
```

这个式子保证**最后一层一定有**。实测：

| `n_layer` | 有 ve 的层 | 占比 |
|---|---|---|
| 6 | `[1, 3, 5]` | 50% |
| 24 | `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]` | 50% |

**注意 `n_layer` 是偶数时是奇数层；奇数时是偶数层**
（`n_layer-1` 决定奇偶）。`n_layer=6` → `(n_layer-1)%2 = 1` →
索引为奇数的层。末层 5 是奇数 ✓。

为什么「末层必有」最要紧？因为**最靠近 `lm_head` 的那一层**
决定了输出。如果它没有 identity 通路，深层蒸馏掉的 token 身份
就永远补不回来了。

**这个取舍省了一半的参数**（50% vs 100%），
是 ResFormer 论文的折中。

---

## 概念：★ 那 1.51 亿参数是怎么来的

这是本章最重要的一段。让我先给你看实测：

```
trick                          加的参数       占总参数     FLOPs/token 增量
--------------------------------------------------------------------
use_value_embeds        150,996,672       77.420%                  0
```

**分解**（`full` 档：`padded_vocab = 16384`，`kv_dim = 768`）：

```
每张 ve 表 = 16384 × 768 = 12,582,912 参数
ve 表张数 = 12
合计       = 150,994,944 参数
```

**加上 `wte`（12.58M）和 `lm_head`（12.58M）之后**：

| 部件 | 参数量 | 占 `full` |
|---|---|---|
| `wte`（token 嵌入） | 12,582,912 | 3.6% |
| **`value_embeds`（12 张）** | **150,994,944** | **43.6%** |
| `lm_head` | 12,582,912 | 3.6% |
| transformer 矩阵（24 层全部） | 169,871,040 | 49.1% |
| 逐层标量 | 74 | 0.0% |
| **合计** | **346,031,882** | 100% |

**三个数字要一起看**：

1. **`value_embeds` 比整个 transformer 矩阵还接近**（0.89×）
2. **一张 ve 表 = 一张完整的 `wte` 表**（都是 `16384 × 768`）
3. **12 层各有一张**，所以嵌入类参数（wte + ve + lm_head）
   一共占了 **50.9%**

### 这是 faithful 复刻，不是本项目的 bug

我核对了 nanochat 原文（`nanochat/gpt.py`）：

```python
self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim)
                                   for i in range(config.n_layer) if has_ve(i, config.n_layer)})
```

**nanochat 就是每层一张独立表。** 本项目一字不差地复刻了它。

**但复刻不等于合理**，所以本项目加了一个开关：

| 方案 | CLI | ve 参数 | 总参数 |
|---|---|---|---|
| **每层独立（nanochat）** | 默认 | 150,994,944 | 346,031,882 |
| **全层共享一张** | `--shared-value-embeds` | 12,582,912 | **207,619,850** |
| 差 | | **138,412,032** | **-40.0%** |

（`full` 档实测，两个数都是真建模型数出来的。）

12 张表意味着：**每个 token 要学 12 个不同的向量**（每层一个），
而它们全部在描述「这个 token 是谁」这件事。

**共享显然更省**，但会有一个真实的损失：不同层需要的
「身份投影方向」不同。比如第 1 层可能需要粗粒度的词性信息
（`ve` 应该主要编码「这是个动词」），第 23 层可能需要细粒度的
语法角色。共享一张表的话，这两种需求会互相拉扯。

**我倾向共享**（省 40% 参数），但这是**没有实测的判断** ——
也可能 nanochat 实测过共享会掉点，代价就是现在这样。

### 开关的实现：一个刻意的类型分叉

共享时 `self.value_embeds` 是**裸的 `nn.Embedding`**，
独立时是 `nn.ModuleDict`。所以有 4 处要分开：

```python
# __init__
if cfg.share_value_embeds:
    self.value_embeds = nn.Embedding(padded, kv_dim, device=device)
else:
    self.value_embeds = nn.ModuleDict({...})

# init_weights（初始化）
if cfg.share_value_embeds:
    torch.nn.init.uniform_(self.value_embeds.weight, -s, s)
else:
    for ve in self.value_embeds.values():
        torch.nn.init.uniform_(ve.weight, -s, s)

# init_weights 末尾（降 bf16）
if isinstance(self.value_embeds, nn.Embedding):
    self.value_embeds.to(dtype=COMPUTE_DTYPE)
else:
    for ve in self.value_embeds.values():
        ve.to(dtype=COMPUTE_DTYPE)

# forward（取值）
if isinstance(self.value_embeds, nn.Embedding):
    ve = (self.value_embeds(idx).to(x.dtype)
          if has_value_embed(i, cfg.n_layer) else None)
else:
    ve = (self.value_embeds[str(i)](idx).to(x.dtype)
          if str(i) in self.value_embeds else None)
```

**用 `isinstance` 而不是 `hasattr(self.value_embeds, "values")`** ——
后者能跑（`ModuleDict` 有 `values()`，`nn.Embedding` 没有），
但意图不清晰，读者得先知道这个巧合。

> ⚠ **注意共享模式下 forward 里多了一个 `has_value_embed` 判断。**
> 共享表对所有层都存在，但仍然**只喂给「该有 ve」的层** ——
> `ve_gate` 只建在那 50% 的 block 上（见下面的手抄代码）。
> 如果漏了这个判断，所有层都会收到 `ve`，但只有一半的层会消费它，
> 另一半层的 `ve` 参数收不到梯度 —— 而
> `setup_optimizer` 的完整性断言**不会**抓到（参数都被分组了）。


### 顺带：优化器分组也不一样

nanochat 给 ve **单独一组**，`embedding_lr` 的一半：

```python
dict(kind='adamw', params=value_embeds_params,
     lr=embedding_lr * dmodel_lr_scale * 0.5,
     betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01)
```

本项目也这么做了（`muon.py` 里 `ve` 组，`lr=ac.embedding_lr * ... * 0.5`），
`weight_decay=0.01` 也一致。这一处**是对齐的**。

---

## 手抄代码

### 第 1 块：`has_value_embed` 的布置规则

```python
def has_value_embed(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2
```

**这一行是整个 trick 里最容易写错的地方。**

常见的手抄版本：

```python
def has_value_embed(layer_idx, n_layer):
    return layer_idx % 2 == 1          # ❌ 只保证奇数层
```

这样写的话，**`n_layer` 为偶数时末层（奇数索引）恰好被覆盖**，
看起来没问题。但 `n_layer` 为奇数时（索引 0,2,4），末层 4 是偶数，
**末层就没有 ve 了** —— 而末层是最重要的那一层。

正确写法里的 `(n_layer - 1) % 2` 就是为了处理这个。

### 第 2 块：`__init__` 里建表

```python
if cfg.use_value_embeds:
    kv_dim = cfg.n_kv_head * cfg.head_dim
    self.value_embeds = nn.ModuleDict({
        str(i): nn.Embedding(padded, kv_dim, device=device)
        for i in range(cfg.n_layer) if has_value_embed(i, cfg.n_layer)
    })
```

**三处容易写错**：

1. **键是 `str(i)` 不是 `i`。** 因为 `nn.ModuleDict` 的键必须是
   字符串。而 forward 里就要写 `self.value_embeds[str(i)]` ——
   这个 `str()` 转换是「到处都要记得转换」的典型来源。
   忘了就会 `KeyError: 1`。
2. **输出维度是 `kv_dim` 不是 `n_embd`。** 第 12 章的 GQA 一样：
   默认配置下两者相等，测不出来；开了 GQA 就炸。
3. **维度是 `padded` 不是 `cfg.vocab_size`。** 否则
   `nn.Embedding` 的行数和 `wte` 不一致 —— 而 forward 里
   `ve = self.value_embeds[str(i)](idx)` 传的 `idx` 里有
   padding 区的 id 吗？本项目 `vocab_size == padded` 所以测不出来，
   但**对齐了更安全**。

### 第 3 块：attention 里的门控混合

```python
if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    v = v + gate.unsqueeze(-1) * ve
```

**三处容易写错**：

1. **`unsqueeze(-1)` 漏了。** `gate` 是 `(B,T,n_kv_head)`，
   `ve` 是 `(B,T,n_kv_head,head_dim)`。不做广播的话
   `gate * ve` 会因为最后一维不匹配而报错（或者在
   `n_kv_head == head_dim` 时**碰巧**对上 —— `full` 档
   两个都是 12 和 64，不等；`debug` 档可能相等）。
2. **`ve.view` 要在 gate 之前还是之后都行**，但 `ve` 是 forward
   参数（每次调用都新建的），`view` 是**无返回值的原地操作**，
   所以必须在使用 `ve` 前 `view` 好。
3. **`x[..., :12]` 取的是归一化后的 `x`。** `self.attn` 收到的是
   `self.norm1(x)`（第 14 章的 Pre-LN），所以门控的输入已经是
   尺度受控的向量。这是 `ve_gate` 能只用 12 通道的原因之一。

### 第 4 块：`ve_gate` 的创建条件

```python
self.ve_gate = (Linear(self.ve_gate_channels, cfg.n_kv_head, bias=False, device=device)
                if (cfg.use_value_embeds and has_value_embed(layer_idx, cfg.n_layer))
                else None)
```

**`and has_value_embed(...)` 这个条件不能漏。**

漏了的话每个 block 都有一个 `ve_gate`，但只有一半的 block
会收到 `ve != None`。多余的 `ve_gate` 是**纯废参数** ——
有梯度（初始为 0 时梯度也是 0）但永远不更新。

更糟的是：`optim.muon.setup_optimizer` 里有一个断言

```python
accounted = len(matrix_params) + sum(len(v) for v in adam_groups.values())
assert accounted == sum(1 for _ in model.parameters()), "参数分组不完整：..."
```

`ve_gate` 是 2 维，会被归到 `matrix_params`。如果它们真的没用，
**Muon 会去正交化一批永远为零的梯度** —— 不报错，只是浪费。

---

## 动手验证

### 验证 1：那 1.51 亿参数的分解

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import build_model
from model.layers import has_value_embed

cfg = make_run_config("full").model
m = build_model(cfg, device="meta")

V, kv = m.padded_vocab_size, cfg.n_kv_head * cfg.head_dim
n_ve = len(m.value_embeds)
print(f"padded_vocab={V}  kv_dim={kv}  ve 表张数={n_ve}")
print(f"每张表 = {V * kv:,}")
print(f"ve 合计 = {n_ve * V * kv:,}")
print()
bd = m.param_breakdown()
tot = bd["合计"]
for k, v in bd.items():
    bar = "#" * int(40 * v / bd["transformer矩阵"])
    print(f"  {k:22s} {v:>12,}  {v / tot:5.1%}  {bar}")
print()
print(f"ve / transformer矩阵 = {bd['value_embeds'] / bd['transformer矩阵']:.2f}x")
print(f"嵌入类 (wte+ve+lm_head) 占 {(bd['wte (token嵌入,查表)'] + bd['value_embeds'] + bd['lm_head']) / tot:.1%}")
```

实测（`full` 档）：

```
padded_vocab=16384  kv_dim=768  ve 表张数=12
每张表 = 12,582,912
ve 合计 = 150,994,944

  wte (token嵌入,查表)         12,582,912    3.6%  ##
  value_embeds            150,994,944   43.6%  ###################################
  lm_head                  12,582,912    3.6%  ##
  transformer矩阵         169,871,040   49.1%  ########################################
  逐层标量                         74    0.0%
  合计                    346,031,882  100.0%

ve / transformer矩阵 = 0.89x
嵌入类占 50.9%
```

**看一眼那两根最长的** —— 一个是 transformer 本体，一个是 value embeds。
它们几乎一样长。

### 验证 2：门控范围确实是 (0, 3)

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import build_model

cfg = make_run_config("full").model
m = build_model(cfg, device="cuda")
blk = m.transformer.h[1]          # 第 1 层有 ve
print(f"ve_gate 是 None? {blk.attn.ve_gate is None}")
print(f"ve_gate 形状: {tuple(blk.attn.ve_gate.weight.shape)}  (12 通道 -> 12 头)")
print(f"参数量: {blk.attn.ve_gate.weight.numel()}")
print(f"ve_gate 权重范围: [{blk.attn.ve_gate.weight.min():.4f}, "
      f"{blk.attn.ve_gate.weight.max():.4f}]")

x = torch.randn(2, 8, cfg.n_embd, device="cuda")
with torch.no_grad():
    g = 3 * torch.sigmoid(blk.attn.ve_gate(x[..., :12]))
print(f"\n随机输入下 gate 范围: [{g.min():.4f}, {g.max():.4f}]")
print(f"  全 1 的输入（x 前 12 通道=0）时: ", end="")
with torch.no_grad():
    g0 = 3 * torch.sigmoid(blk.attn.ve_gate(torch.zeros(2, 8, 12, device="cuda")))
print(f"[{g0.min():.4f}, {g0.max():.4f}]  <- 初始化的 'slightly above neutral'")
```

实测：

```
ve_gate 是 None? False
ve_gate 形状: (12, 12)  (12 通道 -> 12 头)
参数量: 144
ve_gate 权重范围: [0.0002, 0.0197]

随机输入下 gate 范围: [1.4364, 1.5604]
零输入 gate = 1.5000  <- 初始化的 'slightly above neutral'
```

**看最后两行**：

- **零输入时 `gate` 精确等于 1.5。** 因为 `sigmoid(0) = 0.5`，
  `3 × 0.5 = 1.5`。这是初始化刻意达到的 ——
  「slightly above neutral」（中性值 1 的 1.5 倍）。
- **随机输入下范围只有 `[1.44, 1.56]`**，因为 `ve_gate` 的权重
  被初始化在 `(0, 0.02)` 这么小的区间，所以 gate 几乎动不了。
  **门控的「能力」要靠训练慢慢长出来。**

### 验证 3：`has_value_embed` 的边界情况

```bash
uv run pytest tests/test_core.py -k value_embeds -v
```

自己也可以扫一遍各种深度：

```python
import sys
sys.path.insert(0, "src")
from model.layers import has_value_embed

for L in range(1, 9):
    idx = [i for i in range(L) if has_value_embed(i, L)]
    print(f"n_layer={L}: 有 ve 的层 = {idx}  末层 {L-1} 在里面? {L-1 in idx}")
```

实测：

```
n_layer=1: 有 ve 的层 = [0]  末层 0 在里面? True
n_layer=2: 有 ve 的层 = [1]  末层 1 在里面? True
n_layer=3: 有 ve 的层 = [0, 2]  末层 2 在里面? True
n_layer=4: 有 ve 的层 = [1, 3]  末层 3 在里面? True
n_layer=5: 有 ve 的层 = [0, 2, 4]  末层 4 在里面? True
n_layer=6: 有 ve 的层 = [1, 3, 5]  末层 5 在里面? True
```

**奇数深度时是偶数层，偶数深度时是奇数层** —— `(n_layer-1) % 2`
就是干这个的。如果写成 `layer_idx % 2 == 1`，`n_layer=1` 和
`n_layer=3` 的末层就没 ve 了。

### 验证 4：ve 真的进了 v，且没进 k/q

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import build_model

cfg = make_run_config("ablation").model
cfg.use_value_embeds = True
m = build_model(cfg, device="cuda")
blk = m.transformer.h[1]

x = torch.randn(2, 8, cfg.n_embd, device="cuda")
ve = torch.randn(2, 8, cfg.n_kv_head, cfg.head_dim, device="cuda")

with torch.no_grad():
    y_ve = blk.attn(x, m.cos[:, :8], m.sin[:, :8], (64, 0), None, ve)
    y_no = blk.attn(x, m.cos[:, :8], m.sin[:, :8], (64, 0), None, None)
print(f"有 ve: {y_ve.shape}   无 ve: {y_no.shape}")
print(f"输出差异: {(y_ve - y_no).abs().max().item():.6f}")
print("（c_proj 初始为 0，所以差异应该是 0 —— 需要先打破恒等映射）")
```

⚠ **这个验证在初始状态下测不出东西** —— `c_proj` 全零
（第 17 章的主题）。要测先给 `c_proj` 灌随机值。

---

## ★ 消融实验

### 消融 1：ve 有没有用（`full` 档的默认组合里它是开着的）

```bash
# 基线（ve 开）
bash script/train_base.sh full --no-resume --model-tag full_ve

# 关掉（有现成开关：--no-value-embeds）
bash script/train_base.sh full --no-resume --model-tag full_nove --no-value-embeds

bash scratch/ablation.sh full_ve
```

**预期**：`full_nove` 的 bpb 会**略差**。但注意 —— 关掉 ve
同时省了 1.51 亿参数（43.6%）。所以这个消融的真正结论应该是
一条**帕累托前沿**，不是单看 bpb：

| | bpb | 参数量 |
|---|---|---|
| ve 开 | ? | 346M |
| ve 关 | ? | 195M |

**如果 bpb 差 0.5%、参数省 43.6%，那 ve 值不值？**
这个权衡取决于你的目标（本项目目标是「在 16 GB 上训得动」），
不取决于「哪个 bpb 低」。

> ⚠⚠ 又是 41 小时 × 2 = 82 小时。**本项目没跑过。**

### 消融 2：★ 共享 vs 独立（本章最值得做的那个）

**有现成开关**：

```bash
# 每层一张（默认，nanochat 的做法）
bash script/train_base.sh full --no-resume --model-tag full_ve_sep

# 共享一张
bash script/train_base.sh full --no-resume --model-tag full_ve_shr --shared-value-embeds

bash scratch/ablation.sh full_ve_sep
```

**预期**：共享版 bpb **持平或略好**（少了 1.38 亿参数，泛化更好），
参数量从 346M 降到 208M（**-40%**）。

**这个消融的价值远超 bpb 本身** —— 如果成立，
`full` 档就能在同预算下训更大的 transformer 本体。

> ⚠⚠ `full` 档一次 41 小时，两个配置 82 小时。
> **本项目没跑过这个消融**，上面是预期不是实测。
> 但它是卷 3 里唯一一个「可能改变 full 档整体配置」的消融，
> 值得单独排时间做。

### 消融 3：ve_gate 的通道数

```python
self.ve_gate_channels = 12    # 改成 64 / 256 看 bpb
```

参数量 `12 × n_kv_head = 144` 实在太小，改到 256 也才
`256 × 12 × 12 = 36864`。**用门控的视角看，ve 的「聪明」程度
几乎全在嵌入表上，门控只是个开关。**

---

## 常见坑

### 坑 1：`str(i)` 忘了转换

见上面。`KeyError: 1`。

### 坑 2：`gate` 漏了 `unsqueeze(-1)`

见上面。**最阴的是它在某些配置下碰巧能跑** ——
如果 `n_kv_head == head_dim`，`gate * ve` 的形状就碰巧对上了。

### 坑 3：`has_value_embed` 写成 `layer_idx % 2 == 1`

见上面。**奇数深度时末层会丢。** `full` 档 24 层是偶数，
所以这个 bug 在本项目的主要档位上测不出来 —— 只有 `debug`
（2 层）和 `smoke`（4 层）也是偶数，全部测不出来。

> 这是「为什么本项目的档位都是偶数层反而让某些 bug 隐形」
> 的一个例子。判据里 `test_value_embeds_live_on_alternate_layers`
> 用的是极小模型，深度由测试自己指定 —— 这才是能抓到它的方式。

### 坑 4：以为 ve 的输出维度是 `n_embd`

见上面。默认配置下相等，GQA 一开就炸。

### 坑 5：忘了 43.6% 的参数预算

这是本章最大的坑 —— **不是代码 bug，是设计陷阱**。

你如果只看 `param_breakdown()` 的最后一行「合计 346,031,882」，
会以为这是一个 346M 的 transformer。实际上：

- transformer 本体只有 170M
- **另外 176M 全是嵌入类的表**（wte + 12 张 ve + lm_head）

**这直接影响 scaling 预测。** 第 02 章讲过「嵌入占比 10%~25%
是比较健康的范围」—— 本项目 `full` 档的嵌入类是 **50.9%**，
远远超出那个范围。

> ⚠ 也就是说：**`full` 档实际上不是 nanochat 那个 124M-ish 的
> 模型放大 2.8 倍**，而是一个 170M transformer + 176M 嵌入表的
> 组合。用 `num_matmul_params()`（195M）而不是
> `num_params()`（346M）做 scaling 计算是对的 ——
> 嵌入表是查表，没有矩阵乘 FLOPs。
> **但显存和通信成本是按 346M 算的。**

---

## ★ 一个显存优化：嵌入类降到 bf16

本章的三类参数（`wte` + 每张 `ve` + `lm_head`）占了 `full` 档的
**50.9%**。它们全是 fp32，而 nanochat 在 `init_weights` 末尾
把它们转成 `COMPUTE_DTYPE`（bf16）。**本项目现在也这么做。**

### 为什么这三类可以降精度

```python
# init_weights 末尾
if COMPUTE_DTYPE != torch.float16:
    self.transformer.wte.to(dtype=COMPUTE_DTYPE)
    self.lm_head.to(dtype=COMPUTE_DTYPE)
    if hasattr(self, "value_embeds"):
        if isinstance(self.value_embeds, nn.Embedding):
            self.value_embeds.to(dtype=COMPUTE_DTYPE)
        else:
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)
```

三个理由：

| 理由 | 说明 |
|---|---|
| **没有矩阵乘 FLOPs** | `num_matmul_params()` 不统计它们（第 16 章）。它们是查表，所以精度不影响「训练速度」这个口径 |
| **前向不累积** | 查表的舍入误差不会像深层矩阵乘那样一层层放大 |
| **优化器内部转 fp32** | `adamw_step` 第 85 行 `p32, g32 = p.float(), grad.float()`，**整段计算都在 fp32**，只有最后写回是 bf16 |

第三点是关键。`adamw_step` 的 docstring 里原本就写着

> 「我们把 wte / value_embeds 直接存成 COMPUTE_DTYPE（bf16）省显存。
> 但 bf16 只有 8 位尾数，算 `1 - beta2`（beta2=0.999 时等于 0.001）
> 会直接下溢成 0，动量就再也不衰减了。所以中途必须转 fp32。」

**这段注释描述的行为 `init_weights` 从来没实现过。**
现在实现了 —— 优化器的代码一直是对的，只是没人把参数喂成 bf16。

### ⚠ 为什么矩阵参数**不能**这么转

**Muon 要做 Newton-Schulz 正交化**（第 23 章）：那是大矩阵上的
连续乘法（5 步 `X ← 1.5X − 0.5X(XᵀX)`），bf16 的 8 位尾数会让
正交性直接崩掉。

所以转换必须**只点名那三类**，不能用 `self.to(dtype=...)` ——
那会把矩阵参数一起降级。

`tests/test_core.py` 里有一个判据专门盯这件事（见下面的验证 1）。

### ⚠ 为什么 fp16 是例外

`GradScaler` 需要用 fp32 梯度 `unscale_` loss。fp16 的梯度
容易上溢/下溢，`unscale_` 会失效。所以 nanochat 和本项目都
保留了 fp32 嵌入。`COMPUTE_DTYPE` 恒为 bf16，这个分支是保险。

### 实测省了多少

**估算（`full` 档）**：嵌入类 176,160,768 个参数，从 fp32 降到 bf16：

```
参数本身        176,160,768 × 2 B = 336 MiB
AdamW 两个状态  176,160,768 × 2 × 2 B = 672 MiB
                              合计 ≈ 0.98 GiB
```

（16 GB 卡的 6%。）

**实测（`ablation` 档，含全部 5 个 trick，dbs=8，跑 6 步）**：

```
                 嵌入类 bf16   参数总数     峰值 GiB
ablation bf16    31,457,280   42,074,366     3.017   ← 省 188 MiB
ablation fp32            0   42,074,366     3.200
占比              74.8%
```

**注意参数量没变**（42,074,366 两边一样）—— 省的是**每个参数
占几个字节**，不是参数个数。

> 为什么 `ablation` 档的 bf16 占比（74.8%）比 `full` 档（50.9%）高？
> 因为 `ablation` 档 transformer 本体小得多（d6 vs d24），
> 而嵌入类的绝对大小一样（`V×D` 随 `n_embd` 线性，`D=384` vs `768`）。
> **d6 上 12 张 ve 表就已经占了大半** —— 这从另一个角度印证了
> 本章开头那个 43.6% 不是 full 档的特例。

### 验证 1：判据必须同时检查「转了」和「没转太多」

```bash
uv run pytest tests/test_core.py -k compute_dtype -v
```

这个判据有四条断言，我用**两路故障注入**验证过它抓得住：

```python
# 注入 1：把整段转换删掉
if False: pass
#   -> 红：wte 是 fp32 不是 bf16

# 注入 2：转换写得太宽
self.to(dtype=COMPUTE_DTYPE)      # ★ 把矩阵参数也降了
#   -> 红：「矩阵参数 transformer.h.0.attn.c_q.weight 被降成了
#          torch.bfloat16 -> Muon 正交化会失效」
```

**注入 2 是这个判据存在的全部理由。** 一个只检查「嵌入类是 bf16」
的判据会完全放行注入 2 —— 而那正是会让 Muon 静默失效的 bug。

### ⚠ 这个改动的代价（诚实地说明）

bf16 的 8 位尾数意味着**参数更新本身是粗粒度的**。
AdamW 的单步更新量 `lr ≈ 0.3 × 1/√768 ≈ 0.011`，而一个典型
嵌入权重的量级是 `~1`。相对更新量约 1% —— **远大于 bf16 的
相对精度（约 0.4%）**，所以单步更新不会完全丢失。

但如果学习率调得更小，或者某个 token 的嵌入长期不被更新，
bf16 的分辨率会成为下限。

**nanochat 明确接受了这个代价**（「optimizer can tolerate
reduced-precision embeddings」）。本项目跟随，但你要知道
**这是一个取舍，不是一个纯粹的免费午餐**。

**本项目的验证范围**：只测了「loss 有限、梯度非 None、
参数没变成 0/NaN」（`smoke` 和 `ablation` 两档各 40 步）。
**没有测过「bf16 嵌入 vs fp32 嵌入的最终 bpb 差异」** ——
那需要完整训练，属卷 5 之后才能做的事。

---

## 延伸

**ResFormer 的原始论文**

《Simplifying Transformer Blocks》（Zhang et al. 2024）的核心论点：
Transformer block 里的非线性和逐元素运算可以大量简化。
其中「value residual」这一条就是本节的内容 ——
把 token 嵌入直接作为一条残差加到 `v` 上。

论文的消融结论：**简化后参数量和 FLOPs 都下降，但 loss 不变甚至更好**。
原因是原来那些「需要用矩阵乘来学出来的 token 级变换」，
用一个查表就能直接表达。

**为什么是「隔层」而不是每层**

论文的经验结论之一。直觉解释：value residual 是一个
「token 身份 → 内容」的直连通路。**每层都加会让它成为主导信号**，
而模型的上下文计算反而被挤掉。隔层加等于
「每两层刷新一次身份信息」。

**和 U-Net / DenseNet skip 的对照**

| 方案 | 跳过什么 | 跳的时机 |
|---|---|---|
| U-Net | 编码器的同分辨率特征 | 解码器每层 |
| DenseNet | 所有前层的特征（concat） | 每层 |
| **value residual** | **token 嵌入** | **隔层，且只加到 v** |

value residual 的独特之处是**它不经过残差流**，直接注入注意力内部。
所以它影响的只是「被找到时提供什么」，不改变「找谁」——
这让它比其他 skip 方式更「温和」。

**如果想省参数又不共享表**

有个折中：**低秩分解**。把 `16384 × 768` 拆成
`16384 × 64` + `64 × 768`，参数从 12.58M 降到 1.05M（-92%），
12 张表共省 1.39 亿。代价是每层多一个小矩阵乘
（`768 × 64` 和 `64 × 768`，很小）。

这个方案 nanochat 没做，本项目也没有。但它是**唯一同时满足
「每层独立」和「参数可控」的路**。如果有时间，这是一个有价值的
扩展方向。

---

## 下一章

[第 19 章：Smear 与 Backout](19-Smear与Backout.md) ——

本章的 trick 加了 1.5 亿参数，占 43.6%。
下一章两个 trick 加起来是 **26 个参数**，占 0.0000%。

而且其中有一个**做减法** —— 前四个 trick 全都在往上加，
只有它在往下减。为什么需要减法？
