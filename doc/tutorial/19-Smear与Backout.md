# 19 · Smear 与 Backout

上一章的 trick 加了 1.5 亿参数，占 43.6%。
本章两个 trick 加起来是 **26 个参数**，占 0.0000075%。

而且其中有一个**做减法** —— 前四个 trick 全都在往上加，
只有它在往下减。

---

## 本章目标

- 说清 smear 补的是什么信息，为什么「前一个 token」是最便宜的增强
- 理解 backout 为什么需要**减法**，以及它和前面四个「往上加」的 trick 在思路上有何不同
- 掌握 smear 的三个分支（训练 / prefill / decode）—— 这是全项目分支最多的一个函数

---

## 前置回顾

- [第 17 章](17-resid-lambdas与x0-lambdas.md)：5 个 trick 的共同框架（都是「给残差流加额外通路」）
- [第 16 章](16-meta-device三步建模型.md)：meta device 三步法（第 17 章那个恒等初始化的坑还记得吗）

---

## 概念：Smear —— 免费送一个 bigram 线索

### 动机

第 18 章讲了 value embedding 补「token 身份」。但身份有个明显的不足：

```
"The" "cat" "sat" "on" "the" "mat"
```

每个 token 的嵌入只知道「我是一个孤立的词」。**「上一个词是什么」
这个信息不在任何一个 token 的嵌入里。**

注意力理论上能学出来（浅层很难，见第 18 章），但那需要步数。
smear 的做法是**直接把它加进去**：

```python
# _apply_smear 的 docstring
动机：只从 token 身份学到的嵌入里没有任何「上一个词是什么」的信息。
注意力理论上能自己学出来，但在浅层很难。把前一个 token 的嵌入
直接加过来，等于免费给了模型一个 bigram 线索。
```

「smear」是「涂抹」的意思 —— 把前一个 token 的信息**抹**到
当前位置上。

### 实现：门控混合

```python
B, T, _ = x.shape
Tq = x.size(1)
if kv_cache is None:
    assert T > 1, "训练时序列长度必须 > 1"
    gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
    return torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
```

数学上就是：

```
x'[0] = x[0]                          （第一个位置没有前驱）
x'[t] = x[t] + λ·sigmoid(gate(x[t]))·x[t-1]     （t ≥ 1）
```

**注意 `x[t-1]` 是「原始的 x」不是「smear 之后的 x'」。**
所以 smear 不会递归传播 —— 第 5 个 token 不会带上第 4 个的 smear 内容。
这是一次性的「一跳」操作。

实现上用 `torch.cat` 而不是 in-place：`x[:, 1:] + gate * x[:, :-1]`
会同时读 `x` 的两段，in-place 会踩到自己的数据。

### 三个数：λ、gate 的输入、初始化

**① `smear_lambda` 是总闸门**

```python
if cfg.use_smear:
    self.smear_gate = Linear(24, 1, bias=False, device=device)
    self.smear_lambda = nn.Parameter(torch.zeros(1, device=device))
```

```python
# init_weights
if cfg.use_smear:
    torch.nn.init.zeros_(self.smear_lambda)      # 初始关闭
    torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
```

**`smear_lambda` 初始为 0** —— 这是 5 个 trick 里**唯一一个**
初始关闭的。原因很实在：它加的是**前一个 token 的整个嵌入**，
量级和 x 一样（不像第 18 章的 ve 有 gate 缩放）。
如果不关掉，残差流在第 0 层就被污染了。

注释说「门控 sigmoid(0)=0.5 但 lambda=0」——
所以 `smear_gate` 的初始值被完全短路掉了，训练开始时
**整个 smear 通路是关闭的**，由 `smear_lambda` 慢慢打开。

**这和 `resid_lambdas`（初始 1.15）、`ve_gate`（初始 1.5）正好相反。**

> **这是五个 trick 里唯一保守的。** 想想为什么：
> resid 乘的是「已有信息」（放大无害），
> ve 走的是门控混合（有 sigmoid 限幅），
> 而 smear 加的是「另一个 token 的完整嵌入」——
> **在浅层，token 嵌入是最「原始」的信号，污染它的代价最大。**
> 所以这里必须从 0 开始让网络自己决定要不要。

**② `smear_gate` 只看前 24 通道**

```python
gate = ... torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
```

`smear_gate` 是 `Linear(24, 1)` —— **24 个参数**。
输出是标量（不是每头一个）—— 和第 18 章的 `ve_gate` 不同。

理由：smear 加的是**整个嵌入**（所有头共享），
所以门也应该是全局的一个标量。

⚠ **硬编码的 24 有个隐患**：`x[..., :24]` 要求 `n_embd >= 24`。
本项目最小的 `debug` 档 `n_embd = 32`，安全。
但如果你把 `n_embd` 配到 16，smear 会静默地只读 16 个通道 ——
**不报错，只是门控的输入维度变了。**

**③ 初始化 `(0, 0.02)`**

和 `ve_gate` 一样的小正区间。但因为 `λ=0` 短路了它，
所以这个初始化此刻**毫无作用** —— 只有等 `λ` 离开 0 之后才生效。

---

## 概念：Smear 的三个分支（本章最需要小心的部分）

`_apply_smear` 是全项目**分支最多**的函数，因为它要同时服务
训练和 KV cache 推理两种形态：

```python
def _apply_smear(self, x, kv_cache):
    B, T, _ = x.shape
    Tq = x.size(1)
    if kv_cache is None:
        # ── 训练 / 朴素 generate ──
        assert T > 1, "训练时序列长度必须 > 1"
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        return torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

    # ── 推理：单个 token 时要靠 cache 记住上一步的嵌入 ──
    prev = kv_cache.prev_embedding
    kv_cache.prev_embedding = x[:, -1:, :]
    if Tq > 1:  # prefill，和训练一致
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        return torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    if prev is not None:  # decode
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
        return x + gate * prev
    return x
```

### 三个形态

| 形态 | `kv_cache` | `T` | 走哪条 |
|---|---|---|---|
| **训练** | `None` | `T > 1` | 切片 `x[:, :-1]` |
| **prefill**（推理第一批） | 有 | `T > 1` | **和训练完全一致** |
| **decode**（逐 token 生成） | 有 | `T == 1` | 从 `kv_cache.prev_embedding` 拿 |

### 为什么 decode 需要 cache

训练时 `x[:, :-1]` 能直接看到前一个位置。但 decode 阶段
**每次只前向一个 token** —— 手上的 `x` 只有当前这个位置，
看不到上一个（上一个已经被 KV cache 消化掉了）。

所以要在 `kv_cache` 上挂一个字段记住上一步的嵌入：

```python
prev = kv_cache.prev_embedding
kv_cache.prev_embedding = x[:, -1:, :]      # ← 先存当前，供下次用
```

⚠ **注意这两行的顺序：先读旧的，再存新的。**
写反了的话 `prev` 会等于当前 `x`，smear 就变成 `x + λ·x`（自我放大）。

### 那个 `if prev is not None` 是干什么的

prefill 的第一个 token 之前**没有任何前驱**。所以第一次 decode
时 `prev_embedding` 还是 `None`，此时直接 `return x`（不做 smear）。

`kv_cache` 在构造时必须把 `prev_embedding` 初始化为 `None`。
这个字段**不在第 16 章那份 `register_buffer` 列表里**，
因为它是 KV cache 对象自己的属性，不是模型的 buffer。

### 一致性很重要

如果训练和推理的 smear 行为不一致，训练好的模型在推理时会
看到不同的输入分布 —— **bpb 正常但生成质量差**。

本项目的正确做法：**prefill 分支逐字复制训练分支的代码**
（连表达式都一样）。这不是 DRY 的失败，这是**刻意的重复** ——
两条路径必须永远同步。

---

## 概念：Backout —— 唯一做减法的 trick

前四个 trick 全都在往上加信息。backout 是往下**减**：

```python
# Backout：最后 norm 之前，减去中层残差以抹掉低层特征
if cfg.use_backout:
    self.backout_lambda = nn.Parameter(0.2 * torch.ones(1, device=device))
```

```python
# init_weights
if cfg.use_backout:
    torch.nn.init.constant_(self.backout_lambda, 0.2)
```

forward：

```python
# (3) 逐层前向
x0 = x
backout_layer = cfg.n_layer // 2
x_backout = None
for i, block in enumerate(self.transformer.h):
    ...
    x = block(x, cos_sin, self.window_sizes[i], kv_cache, ve)
    if i == backout_layer:
        x_backout = x            # ← 快照中层

# (4) Backout：减掉中层残差
if x_backout is not None and hasattr(self, "backout_lambda"):
    x = x - self.backout_lambda.to(x.dtype) * x_backout

x = rms_norm(x)
```

数学上：

```
x_final = x_L - λ · x_{L/2}
```

`full` 档：`x_final = x_24 - 0.2·x_12`。

### 为什么要减

源码注释：
「减去中层残差以抹掉低层特征」。

**直觉**：深层残差流是所有层的累加。第 12 层的输出
**仍然包含「第 12 层之前的所有信息」** —— 也就是说，
到第 24 层时，「低层特征」已经被重复累加了 12 次
（第 12 层加了一次，之后 12 层里又各自往上加了它的一部分）。

一个常见的**特征放大问题**：某些「低层」特征（浅层就能算出来、
不需要深度的特征）会在残差流里**过度累积**，喧宾夺主。

减掉中层快照 = **要求最后一层去重新学一遍**，
不要依赖「已经堆在那儿的低层特征」。

### 和前三章思路的对比

| trick | 方向 | 加/减什么 |
|---|---|---|
| `resid_lambdas` | × 1.15 | 放大已有 |
| `x0_lambdas` | + | 加回起点 |
| `value_embeds` | + | 加 token 身份 |
| `smear` | + | 加前一个 token |
| **`backout`** | **−** | **减掉中层** |

**四个加、一个减。** 为什么需要减法？

因为**加法没有「上限」**。前四个 trick 都在往一条单向的总线上
加东西；没有反向的力，总线会越来越「满」，
而「满」意味着新信息被旧信息淹没。

backout 是**唯一一个提供反向力的 trick**。它让残差流变成
一个**有增益也有衰减的动态系统**，而不是纯累加器。

> 这个视角很重要：**残差流不是「越多越好」，而是要有淘汰机制。**
> 「淘汰」就是 backout 干的事。

### 初始值 0.2 不是 0

如果 λ=0，backout 就完全关闭。和 smear 一样「安全」的做法是 0。

但 nanochat 选了 0.2 —— **一上来就减**。

这看起来违反直觉（毕竟刚才说 smear 初始 0 更安全），
但两者有区别：

- **smear 加的是「污染性」信号**（别的 token 的嵌入）
- **backout 减的是「重复」信号**（已经过时的中层输出）

减掉过时信息和加入污染信息，风险方向相反。
所以 backout 可以一开始就打开。

---

## 手抄代码

### 第 1 块：`__init__`

```python
# Smear：把前一个 token 的嵌入按门控混进当前 token
if cfg.use_smear:
    self.smear_gate = Linear(24, 1, bias=False, device=device)
    self.smear_lambda = nn.Parameter(torch.zeros(1, device=device))
# Backout：最后 norm 之前，减去中层残差以抹掉低层特征
if cfg.use_backout:
    self.backout_lambda = nn.Parameter(0.2 * torch.ones(1, device=device))
```

⚠ **`smear_gate` 硬编码了 24**，和 `x_embd` 无关。
这是 nanochat 的原样。本项目没把它提成配置项 ——
见「常见坑 3」。

### 第 2 块：`init_weights`

```python
if cfg.use_smear:
    torch.nn.init.zeros_(self.smear_lambda)      # 初始关闭：门控 sigmoid(0)=0.5 但 lambda=0
    torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
if cfg.use_backout:
    torch.nn.init.constant_(self.backout_lambda, 0.2)
```

**`torch.nn.init.constant_` 用在这里是对的** ——
`backout_lambda` 本来就是 `0.2 * ones(1)`，用 `constant_` 只是
把那个初值重新写一遍（防止将来改了构造处的默认值而忘了同步）。

`torch.zeros(1)` + `zeros_` 也一样。两处写同一个数字是有意的
**双重保险**（构造处是「占位」，`init_weights` 是「权威」）。

### 第 3 块：forward 里的插入位置

```python
# (1) 查表得到初始嵌入
x = self.transformer.wte(idx)
x = rms_norm(x.to(COMPUTE_DTYPE))

# (2) Smear：把前一个 token 的嵌入混进来
if hasattr(self, "smear_lambda"):
    x = self._apply_smear(x, kv_cache)

# (3) 逐层前向
x0 = x                      # ← x0 在 smear 之后
```

**⚠⚠ 关键顺序：`x0 = x` 在 smear 之后。**

第 17 章说过这一点，这里说清楚**为什么**：

如果 `x0` 在 smear **之前**取，那 `x0_lambdas` 加回来的就是
「纯嵌入」。而 smear 的作用是往 x 里加「前一个 token」——
两者**部分冗余**（都在补 token 身份/上下文）。

让 `x0` 含 smear，等于说「加回到起点的通路，通路上带着
前一个 token 的信息」。这让 `x0_lambdas` 的直连更完整 ——
浅层想快速拿到「本 token + 前一个 token」时，有一条近路。

---

## 动手验证

### 验证 1：参数预算（74 vs 1.5 亿）

```bash
uv run python -c "
import sys; sys.path.insert(0,'src')
from common.config import make_run_config
from model.gpt import build_model
cfg = make_run_config('full').model
m = build_model(cfg, device='meta')
tot = m.param_breakdown()['合计']
items = [('resid_lambdas', 24), ('x0_lambdas', 24),
         ('value_embeds', m.param_breakdown()['value_embeds']),
         ('smear_lambda', 1), ('backout_lambda', 1),
         ('smear_gate', m.smear_gate.weight.numel())]
for k, v in items:
    print('  %-16s %12s  %6.2f%%' % (k, f'{v:,}', 100*v/tot))
"
```

实测：

```
  resid_lambdas          24   0.000007%
  x0_lambdas             24   0.000007%
  value_embeds  150,994,944  43.636833%
  smear_lambda            1   0.0000003%
  backout_lambda          1   0.0000003%
  smear_gate             24   0.000007%
```

**本章两个 trick 一共 26 个参数**（`smear_lambda` 1 + `backout_lambda` 1
+ `smear_gate` 24），占 `full` 档的 0.0000075%。

> 顺带看 `value_embeds` 的 43.64% 里有多少是门控？
> `ve_gate` 是 `12×12=144` 参数 × 12 层 = **1728**。
> 也就是说那 1.5 亿里 **99.9989% 是查表，只有 0.0011% 是门控。**
> **ve 的「聪明」几乎全在表里，门控只是个能学出条件的开关。**

### 验证 2：smear 的三个分支行为一致

这是本章最该亲手测的东西 —— 因为「prefill 和训练一致」
是**正确性要求**，不是优化。

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import build_model

cfg = make_run_config("ablation").model
cfg.use_smear = True
m = build_model(cfg, device="cuda")

# 打开门控，否则 λ=0 什么都测不出来
with torch.no_grad():
    m.smear_lambda.fill_(0.3)
    m.smear_gate.weight.normal_(0.0, 0.05)

# 造一个假的 kv_cache（只需要 prev_embedding 字段）
class FakeCache:
    def __init__(self):
        self.prev_embedding = None

x = torch.randn(1, 8, cfg.n_embd, device="cuda")

# 训练形态：一次性喂 8 个 token
train_out = m._apply_smear(x, None)

# prefill 形态：同样 8 个，但带 kv_cache
c = FakeCache()
prefill_out = m._apply_smear(x, c)

print(f"训练 vs prefill 逐位相同: {torch.equal(train_out, prefill_out)}")
print(f"  max|diff| = {(train_out - prefill_out).abs().max().item():.2e}")

# decode 形态：逐个 token 喂，结果拼起来
c2 = FakeCache()
steps = []
for t in range(8):
    steps.append(m._apply_smear(x[:, t:t+1], c2).squeeze(1))
decode_out = torch.stack(steps, dim=1)

print(f"\n训练 vs decode  max|diff| = "
      f"{(train_out - decode_out).abs().max().item():.2e}")
print(f"  decode 的第一个 token 与 train 的第一个 token 相同? "
      f"{torch.equal(train_out[:, 0], decode_out[:, 0])}")
```

**实测**：

```
训练 vs prefill 逐位相同: True
  max|diff| = 0.00e+00

训练 vs decode  max|diff| = 1.19e-07
  第 0 个位置相同? True
```

**三个数字的含义**：

- **训练 vs prefill 逐位相同（diff = 0）** —— 符合预期，
  那段代码是复制的。**diff 精确为 0** 是「真的复制了」的证据；
  如果是「碰巧数值接近」，diff 会有 1e-7 量级。
- **训练 vs decode 差 1.19e-07** —— fp32 的舍入量级，
  因为 decode 路径上多了几次 kernel 边界。**语义完全一致**。
- **第 0 个位置完全相同** —— 因为 `x[0]` 在两个路径下都
  「没有前驱所以不做 smear」。

> ⚠ 这是本章最有价值的判据类型：**跨形态一致性**。
> 本项目的 `tests/test_engine.py` 里有大量这类判据
> （第 30 章会讲 KV cache）。

### 验证 3：`prev_embedding` 的读写顺序

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import build_model

cfg = make_run_config("ablation").model
cfg.use_smear = True
m = build_model(cfg, device="cuda")
with torch.no_grad():
    m.smear_lambda.fill_(0.3)
    m.smear_gate.weight.normal_(0.0, 0.05)

class FakeCache:
    def __init__(self):
        self.prev_embedding = None

x = torch.randn(1, 8, cfg.n_embd, device="cuda")

# 第一步：prev 是 None -> 不做 smear
c = FakeCache()
o0 = m._apply_smear(x[:, 0:1], c)
print(f"第 1 步: prev=None, 输出等于输入? {torch.equal(o0, x[:, 0:1])}")
print(f"        cache 现在存的是: {tuple(c.prev_embedding.shape)}  <- x[:, -1:, :]")

# 第二步：prev 是上一步的 x
c2 = FakeCache()
m._apply_smear(x[:, 0:1], c2)
o1 = m._apply_smear(x[:, 1:2], c2)
gate = 0.3 * torch.sigmoid(m.smear_gate(x[:, 1:2, :24]))
manual = x[:, 1:2] + gate * x[:, 0:1]
print(f"\n第 2 步: 等于 x[1] + gate·x[0]? "
      f"{torch.allclose(o1, manual, atol=1e-5)}")
print(f"  差异 {(o1 - manual).abs().max().item():.2e}")
```

**这一步测的是「读旧存新」的顺序。** 如果写反，
`prev` 会等于当前 `x`，`o1 = x[1] + gate·x[1]`，
和 `manual` 的差异会很大。

实测：

```
第 1 步: 输出等于输入? True
  cache 存的形状: (1, 1, 384)  <- 应为 (1,1,384)

第 2 步: 等于 x[1] + gate·x[0]? True
  差异 0.00e+00
```

**故障注入验证这个判据有效**：

```python
def broken(self, x, kv_cache):
    kv_cache.prev_embedding = x[:, -1:, :]      # ★ 先存
    prev = kv_cache.prev_embedding              # ★ 后读（错）
    ...
```

实测：

```
正确实现 vs 顺序反掉的实现: max|diff| = 0.6787
判据「等于 x[1]+gate·x[0]」在坏实现上: False
```

**判据抓得住**（差异 0.68，判据直接 False）。
注意这个 bug 的严重性 —— 它不会让 decode 崩，只会让
**每个 token 多加一次自己的嵌入**，推理质量悄悄变差。

---

## ★ 消融实验

```bash
# 基线（smear + backout 都开）
bash script/train_base.sh full --no-resume --model-tag full_sb

# 关 smear
bash script/train_base.sh full --no-resume --model-tag full_nosmear --no-smear

# 关 backout
bash script/train_base.sh full --no-resume --model-tag full_nobackout --no-backout

bash scratch/ablation.sh full_sb
```

**这两个开关是本章写作时补上的。** 原来只有 `--use-smear` /
`--use-backout`（打开），没有关闭的入口 —— 而 `full` 档默认
**开着**这两个 trick，于是「消融默认组合里的 trick」这件事
根本没法从命令行做，只能去改 `PRESETS`。

**这是一个真实的缺陷，不是「有意的教学噪音」。** 第 12 章我把
不给 `--n-kv-head` 开关辩护为「减少教学噪音」，那站得住 ——
因为 GQA 在本项目默认关闭，一个用不上的开关只会增加噪音。
但**默认在用的东西不给关闭开关，是纯粹的缺漏**。

现在补齐了 5 个关闭开关：

```
--no-resid-lambdas   --no-x0-lambdas   --no-value-embeds
--no-smear           --no-backout
```

加上 `--shared-value-embeds`（第 18 章）。

> ⚠ **开关的优先级有讲究**：`apply_overrides` 里**先处理 `--use-xxx`
> 再处理 `--no-xxx`**，所以 `--all-tricks --no-smear` 的含义是
> 「开全部，除了 smear」。顺序反过来写的话 `--all-tricks` 会
> 覆盖掉 `--no-smear` —— 一个静默失效的消融。
> `tests/test_core.py` 里有判据专门锁这个顺序。

**预期（诚实版）**：这两个 trick 的效应在 `full` 档上
很可能落在 bpb 噪声内（26 个参数能做的事很有限）。
nanochat 的消融里它们属于「小但正」的一类。

### 更有信息量的消融：`backout_lambda` 的取值

```python
# 改 init_weights 里的 0.2
torch.nn.init.constant_(self.backout_lambda, 0.0)    # 等于关闭
torch.nn.init.constant_(self.backout_lambda, 0.5)    # 减一半的中层
torch.nn.init.constant_(self.backout_lambda, 1.0)    # 完全抵消中层
```

**λ=1.0 特别有意思**：那时 `x_final = x_24 - x_12`，
**中层的信息被完全抵消**。这是个非常干净的对照 ——
如果 λ=1 训得动（bpb 不大幅变差），说明中层输出**没有携带
后续层需要的信息**；如果大幅变差，说明深层严重依赖中层。

**这个消融比「开/关」有信息量得多**，因为它测的是一个
**定量的因果关系**，而不是一个二元的开关。

---

## 常见坑

### 坑 1：`prev_embedding` 的读写顺序反了

见上面。`prev = kv_cache.prev_embedding` 和
`kv_cache.prev_embedding = x[:, -1:, :]` **必须先读后写**。

反过来就变成 `x + λ·gate·x`（自我放大），而且**不报错**。

### 坑 2：decode 分支和训练分支写得不一样

```python
# 训练：gate 用 x[:, 1:, :24]，乘 x[:, :-1]
# decode：gate 用 x[:, :, :24]（只有 1 个位置），乘 prev
```

区别在于**形状**（`T-1` 个 vs 1 个）。如果你顺手把 decode
也写成 `x[:, :-1]`，那 `T=1` 时就是空张量 —— 结果是
「decode 静默地不做 smear」，训练和推理不一致。

**这类 bug 极难查**：训练 loss 完全正常，只有推理质量差。

### 坑 3：`x[..., :24]` 硬编码

`n_embd < 24` 时静默降级。本项目最小档 `debug` 是 32，安全。

但如果哪天加一个 `n_embd=16` 的微型档做测试，smear 的门控
输入会变成 16 维 —— **参数形状还是 `(1, 24)`，不报错**，
只是 `x[..., :24]` 实际只切到 16 个。

**排查方法**：打印 `x.shape[-1]` 和 `smear_gate.in_features` 对比。

### 坑 4：以为 `smear_lambda=0` 时 `smear_gate` 也没用

`smear_gate` 的参数**照样有梯度**（虽然被 `λ=0` 乘成 0，
所以梯度也是 0）。Muon 会去正交化一批全零梯度。

这是**无害的浪费**（24 个参数的零梯度），但要知道：
「λ=0 短路」不等于「gate 不参与计算图」。

### 坑 5：`x0 = x` 放在 smear 之前

见上面。会让 `x0_lambdas` 和 smear 部分冗余。
**不一定有害**（是个设计选择），但**和 nanochat 不一样**，
别在「复刻」的名义下悄悄改掉。

---

## 延伸

**为什么是「前一个」而不是「前 K 个」**

smear 只用 `x[t-1]`。理论上加 `x[t-2]`, `x[t-3]`… 信息更多，
但那样就不再是「便宜的小把戏」了 —— 参数和计算都线性增长。

而且 **`x[t-1]` 是信息量/成本比最高的那一个**：
语言里 bigram 关联（"of the"、"in the"）已经能解释相当一部分
局部结构。更远的关联交给注意力（第 10 章）处理。

**smear 和注意力是分工，不是重叠**

| | 看多远 | 成本 |
|---|---|---|
| smear | 恰好 1 个 token | 24 参数 + 一次加法 |
| 注意力 | 全上下文 | 4 个投影矩阵 |

smear 用 24 个参数解决「上一词关联」这个最高频的模式；
注意力用几亿个参数解决任意模式。**让便宜的先做便宜的。**

**Backout 的其他实现方式**

| 方案 | 做法 |
|---|---|
| **本项目** | `x_L - λ·x_{L/2}` |
| 减多个中层 | `x_L - Σλᵢ·xᵢ`（更灵活，参数更多） |
| 减初始 | `x_L - λ·x_0`（等价于 `x0_lambdas` 取负） |
| LayerDrop | 训练时**随机丢弃**部分残差分支 |

LayerDrop 和 backout 的动机相似（防止过度依赖浅层特征），
但手段不同：backout 是**确定性的线性减法**，
LayerDrop 是**随机性正则**。前者更可控，后者更强。

**五个 trick 的成本-收益总表**（`full` 档实测）

| trick | 参数 | 占 full | 初始值 | 方向 |
|---|---|---|---|---|
| `resid_lambdas` | 24 | 0.000007% | 1.15→1.05 | × 放大 |
| `x0_lambdas` | 24 | 0.000007% | 0.20→0.05 | + 加回起点 |
| `value_embeds` | 150,994,944 | **43.64%** | Uniform | + 加身份 |
| `smear` | 25 | 0.000007% | λ=0 | + 加前词 |
| `backout` | 1 | 0.0000003% | 0.2 | **− 减中层** |

**这张表本身就是本章的结论**：

- **参数成本和影响力完全不成正比。** 26 个参数的 smear
  和 1.5 亿参数的 ve 在设计上是同等重要的。
- **但也不能说「参数不重要」** —— ve 的 1.5 亿参数是实打实的
  显存和通信开销。第 18 章的两个改动都作用在它身上：
  `--shared-value-embeds`（省 40% 参数）和 bf16 转换（省 0.98 GiB）
- **四个加、一个减**，构成残差流的完整动力学。

**五个 trick 的参数预算**（`full` 档实测）：

| trick | 参数 | 占 full | 初始值 | 方向 | CLI 关闭开关 |
|---|---|---|---|---|---|
| `resid_lambdas` | 24 | 0.000007% | 1.15→1.05 | × 放大 | `--no-resid-lambdas` |
| `x0_lambdas` | 24 | 0.000007% | 0.20→0.05 | + 加回起点 | `--no-x0-lambdas` |
| `value_embeds` | 150,994,944 | **43.64%** | Uniform | + 加身份 | `--no-value-embeds` |
| `smear` | 25 | 0.000007% | λ=0 | + 加前词 | `--no-smear` |
| `backout` | 1 | 0.0000003% | 0.2 | **− 减中层** | `--no-backout` |

另外两个改动：

| 改动 | CLI | 省 |
|---|---|---|
| ve 表共享 | `--shared-value-embeds` | 138,412,032 参数（40%） |
| 嵌入类降 bf16 | 默认开 | 约 0.98 GiB（`full` 估算） |

**这张表本身就是卷 3 的结论**：

- **参数成本和影响力完全不成正比。**
- **但参数显存是真实的** —— 43.6% 的参数决定了
  「同样的显存能训多大的 transformer 本体」
- **四个加、一个减**，加上「共享」和「降精度」两个工程开关，
  构成 nanochat 这套架构的全部可调项

（这张表里的 `value_embeds` 是 43.64%，但其中绝大部分是**嵌入表本身**，
不是门控。门控只占 `12 × 12 × 12 = 1728` 个参数。）

>
> ★ **2026-10 消融实测**（`ablation` 档 d6，3,096 步，每组 39 分钟）：
> `--use-smear` **−0.0049**、`--use-backout` **−0.0024** ——
> **两者都在 ±0.02 的噪声带内**，也就是说它们对这个规模的可测质量
> 没有影响（方向都是「开好一点」，但幅度不可区分）。
>
> 对照：`--use-value-embeds` 是 **−0.0226**（唯一超过噪声带的 trick），
> 而 `--all-tricks`（5 个全开）是 **−0.0294**。
> ⚠ **这些是 d6 档 / 2 亿 token 的结论，不能直接外推到 `full` 档 d24。**
>
> 完整消融表见 [README 的 ★ 消融总表](README.md#-消融总表)。

---

## 卷 3 最后一个 trick

[第 20 章：滑动窗口注意力](20-滑动窗口注意力.md) ——

本章的 trick 都是**改模型结构**。
最后一章的滑窗**不改任何数学**，只改「每层看多远」。

而且我会给你一个本项目实测的反直觉结论：
**在 `full` 档（T=1024）上，滑窗一点速度都省不下来。**
