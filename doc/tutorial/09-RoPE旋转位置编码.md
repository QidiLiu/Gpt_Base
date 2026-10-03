# 09 · RoPE 旋转位置编码

**卷 2 里最需要「想明白」的一章。** 前面两章都是「记住公式」，
这一章需要建立一个直觉：为什么旋转能编码位置。

---

## 本章目标

- 理解注意力对顺序的**完全无感**，以及为什么必须显式加位置
- 说清楚 RoPE 的核心性质：**只依赖相对位置**
- 亲手实现 `precompute_rope` 和 `apply_rotary_emb`

---

## 前置回顾

[第 08 章](08-RMSNorm.md) 完成了归一化。现在看注意力的输入
`q, k`，形状都是 `(B, T, H, D)`。

**这一章处理它们的位置信息。**

---

## 概念：注意力对顺序是完全无感的

先做一个思想实验。如果把一句话的 token 顺序打乱：

```
原句：  The  cat  sat  on  the  mat
打乱：  mat  The  on  cat  the  sat
```

用**没有位置编码**的注意力处理这两句话，会发生什么？

答案是：**两句话产生完全相同的注意力分布**（只是输出跟着换了个位置）。
因为注意力的计算是

```
attn(q_i, k_j) = q_i · k_j
```

它只看「第 i 个 query 和第 j 个 key 的内积」，完全不知道 `i` 和 `j`
分别排在第几个。所以模型无法区分「猫在垫子上」和「垫子在猫上」。

**这不是「训练得不够」，是架构上根本没有这个信息。**

> 这个性质有一个漂亮的推论：**没有位置编码的 Transformer 是置换等变的**
> （permutation-equivariant）。打乱输入 = 打乱输出。反过来说，
> 要打破这个等变性，就必须引入位置信息。

历史上有三代方案：

| 方案 | 做法 | 问题 |
|---|---|---|
| **绝对可学习嵌入** | 加一张 `T × D` 的位置表，位置 `t` 查第 `t` 行 | ① 表大小随 T 变，超不出训练长度 ② 相邻位置的表示没有结构关系 |
| **正弦特征编码**（原始 Transformer） | 用 sin/cos 频率 | 能外推，但**没有相对性**：位置 5 和 100 的表示没有简单关系 |
| **RoPE** | 对 q/k 做**旋转** | 相对位置天然内嵌，且可外推 |

RoPE 的关键洞察：**旋转的复合性质**。

---

## 概念：为什么旋转能编码「相对」位置

二维直觉先建立起来。把一个 2 维向量 `v` 旋转角度 `θ`：

```
R(θ) · v
```

关键性质（旋转的复合律）：

```
R(θ_a) · R(θ_b) = R(θ_a + θ_b)
```

也就是说，**先转 θ_b 再转 θ_a，等价于一次转 θ_a + θ_b**。

现在把位置 `t` 编码成角度 `t · ω`（`ω` 是每个通道对的角速度）。
两个位置 `t` 和 `t+k` 的向量之间的内积：

```
⟨R(tω)·q, R((t+k)ω)·k⟩
  = ⟨q, R(ω(t - (t+k)))·k⟩        （把 R(tω) 用 R(-tω) 消掉）
  = ⟨q, R(-kω)·k⟩
```

**中间那个绝对位置 `t` 消失了，只剩下差值 `k`。**

所以：注意力分数只依赖两个位置的**距离**，不依赖它们具体在哪。
这就是 RoPE 相对性的来源，而且它不是设计出来的，
而是旋转复合律的**副产品**。

> 这个推导也是第 11 章滑窗实现的理论基础：滑窗本质上是
> 「只保留 `k ∈ [-left, 0]` 的分数」。

---

## 概念：多频率 —— 为什么每个通道对用不同的 ω

上面是二维的情形。真实的 `head_dim` 是 64（`full` 档），
拆成 **32 对**二维平面。每一对用**不同的角速度**：

```python
inv_freq[i] = 1 / base^(2i / d)        i = 0, 1, ..., d/2-1
```

于是频率是几何级数分布的：

```
i=0  →  inv_freq = 1        （最快，管近距离依赖）
i=1  →  inv_freq = 1/10     (base=100000 时约 1/10^1)
...
i=31 →  inv_freq ≈ 1/base   （最慢，管长距离依赖）
```

**为什么要多个频率？**

单一频率只有一维信息（转了 θ 还是 θ+2π）。而位置编码要能区分
`k=1` 和 `k=2` 两个距离，单一频率下这两者可能给出同一个内积
（因为角度差小于 2π 时正弦函数非单调）。

多频率就像「用多把尺子量长度」：
- 高频尺子（ω 大）分辨 `k=1, 2, 3` 的微小差异
- 低频尺子（ω 小）分辨 `k=100, 200, 300` 的大差异

和 Fourier 分析是同一件事：位置编码就是「把离散位置变成一个
连续信号，然后用不同频率的投影去测量它」。

### `base` 控制什么

`base` 是「最高频率 / 最低频率」的比值。

- `base` 小 → 频率分布集中 → 只能分辨短距离
- `base` 大 → 最低频率更慢 → 能表达长距离关系

原始 RoPE 用 `10000`，LLM 普遍用 `100000`。本项目跟随
（`ModelConfig.rope_base = 100000.0`）。

> nanochat 有一条更激进的结论：**RoPE 可以不训练就外推到
> 远超训练长度的位置**（因为 `t·ω` 是线性的，外推时角度连续变化）。
> 但超过 4 倍训练长度后性能会明显退化 —— 因为低频通道转过太多圈，
> 数值精度不够了。

---

## 概念：GPT-NeoX 的「前后两半」配对方式

旋转要作用在**每一对**维度上。配对方式有两种，本项目用 GPT-NeoX 的：

```
方式 A（GPT-J / 交错）：(0,1), (2,3), (4,5), ...
方式 B（GPT-NeoX）：   (0, 32), (1, 33), (2, 34), ...   ← 本项目
```

方式 B 的实现是把最后一维**切前后两半**：

```python
x1, x2 = x[..., :d], x[..., d:]      # d = head_dim // 2
```

然后：

```
y1 = x1·cos - x2·sin
y2 = x1·sin + x2·cos
```

⚠ **本项目的符号和教科书相反**：

```
教科书： y1 = x1·cos - x2·sin    y2 = x1·sin + x2·cos
本项目： y1 = x1·cos + x2·sin    y2 = -x1·sin + x2·cos
```

差一个负号，等价于「用 `-θ` 旋转而不是 `+θ`」。

**为什么等价？** 因为只关心 q 和 k 的**相对**旋转量。
全体转 `+θ` 和全体转 `-θ`，内积里那个中间位置照样消掉：

```
⟨R(tω)·q, R((t+k)ω)·k⟩ = ⟨q, R(±kω)·k⟩
```

`±` 号在结果里体现为 cos 是偶函数、sin 是奇函数 —— 对内积的整体
符号变化不影响「只依赖 k」这个结论。

本项目沿用 nanochat 的约定（`-θ`）是为了与它的 checkpoint 兼容。

---

## 概念：为什么只对 q 和 k 旋转，不对 v

RoPE 的整个机制作用在 `q · k` 这个内积上。`v` 不参与内积计算 ——
它是被「加权求和」的那个量。旋转 `v` 不会改变任何注意力分数，
只会给输出引入一个没有意义的整体旋转。

所以：**只旋转 q 和 k**。这一点在代码里是：

```python
if self.use_rope and cos.size(1) >= T:
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
```

---

## 手抄代码

### 第 1 块：预算 cos/sin 表

写进 `src/model/layers.py`。

```python
def precompute_rope(seq_len: int, head_dim: int, base: float = 100000.0,
                    device=None, dtype=torch.float32):
    """
    预算 RoPE 的 cos / sin 表。

        inv_freq[i] = 1 / base^(2i/d)        i = 0..d/2-1
        angle[t][i] = t × inv_freq[i]
        cos/sin[t][i] = cos/sin(angle[t][i])

    ⚠ 三处容易写错：
      1. channel 用 arange(0, head_dim, 2) —— 因为配对是「前后两半」，
         偶数步长正好取出 d/2 个下标
      2. freqs 是 t（长度 L）和 inv_freq（长度 d/2）的**外积**，
         所以用 torch.outer 而不是逐元素乘
      3. 最后要加回 batch 维和 head 维（[None, :, None, :]），
         这样在 forward 里广播时不用再 reshape
    """
    if device is None:
        device = "cpu"
    channel = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)                 # (seq_len, head_dim/2)
    return (freqs.cos().to(dtype)[None, :, None, :],
            freqs.sin().to(dtype)[None, :, None, :])  # (1, seq_len, 1, head_dim/2)
```

**关于 `rotary_seq_len = sequence_len * 10`**：

`GPT.__init__` 里这张表按 **10 倍**序列长度预算：

```python
self.rotary_seq_len = cfg.sequence_len * 10
```

因为它极小（`(1, L, 1, d/2) × 2`，`full` 档 L=10240 时只有 5 MB），
多算点省得以后要动态增长的麻烦。真的不够时 `forward` 会 assert：

```python
assert T <= self.cos.size(1), f"序列长度超过 RoPE 表容量：{T} > {self.cos.size(1)}"
```

### 第 2 块：施加旋转

```python
def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    对 q 和 k 施加 RoPE。

    x 形状 (B, T, H, D) —— 注意是 head_dim 在最后一维，不是 SDPA 习惯的
    (B, H, T, D)。我们在 attention 里直接用 (B,T,H,D)，省掉两次 transpose。

    配对方式（GPT-NeoX 风格）：把最后一维切成前后两半
        x = [x1 | x2]，x1 = x[..., :d]，x2 = x[..., d:]
    教科书是 y1 = x1·cos - x2·sin，我们用 y1 = x1·cos + x2·sin，
    等价于用 -θ 旋转而不是 +θ。只影响符号约定，不影响「相对性」。

    ⚠ 为什么不用 x[..., :d], x[..., 1::2]？那是 GPT-J 的交错配对，
      和 NanoChat 的 checkpoint 不兼容。
    """
    assert x.ndim == 4, f"期望 (B,T,H,D)，得到 {tuple(x.shape)}"
    d = x.size(3) // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=3)
```

**为什么 `assert x.ndim == 4`**：这个函数假设形状是 `(B,T,H,D)`。
如果误传 `(B,H,T,D)`，`x.size(3)` 就变成 `T` 而不是 `D`，
`d` 算错，`torch.cat` 的最后一维长度也对不上 —— 会报一个
很难定位的形状错误。这个 assert 把它变成一条清晰的报错。

---

## 动手验证

### 验证 1：相对性质 —— 内积只依赖位置差

这是本章**最重要**的验证，它直接检验「旋转 = 相对编码」这个论点。

```bash
uv run pytest tests/test_core.py -k rope -v
```

自己也可以验 ⚠ **注意这里的写法**：平移不变性说的是「**同一对**
q/k 向量放在不同位置」，而不是「不同位置的 q 和 k」。写错就会得到
看起来「性质不成立」的结论：

```python
import sys, torch
sys.path.insert(0, "src")
from model.layers import precompute_rope, apply_rotary_emb

D, L = 8, 64
cos, sin = precompute_rope(L, D, base=10000.0)
torch.manual_seed(0)

# ★ 固定的一对「内容」向量 —— 它们本身不含位置信息
q_base = torch.randn(1, 1, 1, D)
k_base = torch.randn(1, 1, 1, D)

def score_at(t, gap):
    """q 在位置 t、k 在位置 t+gap 时的内积"""
    qt = apply_rotary_emb(q_base, cos[:, t:t + 1],     sin[:, t:t + 1])
    ks = apply_rotary_emb(k_base, cos[:, t + gap:t + gap + 1],
                                   sin[:, t + gap:t + gap + 1])
    return (qt * ks).sum().item()

for gap in (0, 1, 5, 17):
    vals = [score_at(t, gap) for t in (0, 7, 20, 40)]
    spread = max(vals) - min(vals)
    print(f"gap={gap:3d}: " + "  ".join(f"{v:+.6f}" for v in vals)
          + f"   最大离散={spread:.2e}")
```

实测输出（最大离散就是 float32 的舍入量级）：

```
gap=  0: -0.526053  -0.526053  -0.526053  -0.526053   最大离散=2.38e-07
gap=  1: -2.266718  -2.266718  -2.266718  -2.266718   最大离散=2.38e-07
gap=  5: +1.492137  +1.492136  +1.492136  +1.492136   最大离散=7.15e-07
gap= 17: +2.867661  +2.867661  +2.867661  +2.867661   最大离散=2.38e-07
```

**每一行的四个数应该完全相同**（差 1e-7 量级）。

注意 `cos[:, t:t+1]` 这个切片 —— 它只取位置 `t` 那一行，
用来模拟「同一个向量放在不同位置」。如果离散不是 1e-7 量级而是大
数量级，说明 `apply_rotary_emb` 的旋转没有正确作用，
或者 `cos`/`sin` 被广播到了错误的维度。

### 验证 2：频率确实是几何分布的

```python
import sys, torch
sys.path.insert(0, "src")
from model.layers import precompute_rope

for D in (8, 64):
    cos, sin = precompute_rope(4, D, base=100000.0)
    # 位置 1 的角度 = 1 × inv_freq，用 atan2 从 cos/sin 反解回来
    ang = torch.atan2(sin[0, 1, 0], cos[0, 1, 0])   # 注意 [0,1,0] 去掉 head 维
    r = ang[1:] / ang[:-1]
    print(f"D={D}  inv_freq[0]={ang[0]:.6f}  inv_freq[-1]={ang[-1]:.3e}")
    print(f"   相邻比值 min={r.min():.5f} max={r.max():.5f}"
          f"  理论={100000.0 ** (-2 / D):.5f}")
    print(f"   最高/最低频率比 = {(ang[0] / ang[-1]).item():.1f}  (应该正好是 base)")
```

实测：

```
D=8   inv_freq=[1.0, 0.056234, 0.003162, 0.000178]
   ratio min=0.05623 max=0.05623  theory=0.05623   top/bottom=5623.4
D=64  inv_freq=[1.0, 0.697831, 0.486968, 0.339821, ..., 0.000124, 8.7e-05, 6e-05, 4.2e-05, 2.9e-05, 2.1e-05, 1.4e-05]
   ratio min=0.69783 max=0.69783  theory=0.69783   top/bottom=69783.0
```

两个观察：

1. **相邻比值是常数**（min == max），证实了 `inv_freq` 是严格的几何级数。
2. **`head_dim` 决定了「多少个频率档位」**：`D=8` 只有 4 个频率（`d/2`），
   最高/最低比 5623；`D=64` 有 32 个频率，比值 69783（= base）。
   头维越大，能同时覆盖的「距离尺度」越宽。

### 验证 3：旋转保持范数（旋转是正交变换）

```bash
uv run pytest tests/test_core.py -k rope -v
```

判据里有「`apply_rotary_emb` 不改变 q 的范数」。自己也可以看一眼：

```python
import sys, torch
sys.path.insert(0, "src")
from model.layers import precompute_rope, apply_rotary_emb
# ⚠ precompute_rope 的第一个参数是 seq_len，必须 >= q 的 T，否则广播失败
cos, sin = precompute_rope(32, 64, base=100000.0)
q = torch.randn(1, 32, 2, 64)
print("|q| before", round(q.norm().item(), 5),
      " after", round(apply_rotary_emb(q, cos, sin).norm().item(), 5))
```

实测 `62.95295 → 62.95296`（差 7.6e-6，纯 fp32 舍入）。

如果范数差到 1e-2 量级，说明实现里有缩放、或者 dtype 转换丢了精度。

---

## ★ 消融实验

RoPE 的消融是本章唯一一个「关掉之后模型彻底学不了」的开关：

```bash
# 基线
bash script/train_base.sh ablation --no-resume --model-tag d6_rope

# 关掉 RoPE（完全没有位置信息）
bash script/train_base.sh ablation --no-resume --model-tag d6_norope --no-rope

bash scratch/ablation.sh d6_rope
```

**实测（2026-10）**：Δ bpb = **+0.0530，279σ** —— 全表最大的单项劣化。
（`ablation` 档实测 σ_Δ = 0.000190，显著性阈值 0.001。）
没有位置信息的语言模型在 bpb 上会明显更差，因为它连「这个词在
上一句话的末尾」这种最基本的局部结构都无法表达。

> ⚠ **本章最容易踩的坑就在这个消融里**：`--no-rope` 这个开关曾经
> **完全无效** —— 它只出现在 `describe()` 的打印字符串里，
> `forward` 里无条件施加 RoPE。两组实验的 bpb 一模一样，
> 结论会变成「RoPE 毫无影响」。
>
> 现在 `forward` 里是 `if self.use_rope and cos.size(1) >= T`，
> 而且这个坑在 `test_core.py -k rope` 里有专门的判据钉着。
> **如果你的 `--no-rope` 跑出来 Δ 接近 0，先怀疑开关没生效，
> 而不是相信「RoPE 没用」这个结论。**

### 顺便：base 的影响

```bash
bash script/train_base.sh ablation --no-resume --model-tag d6_base10k  # 需要改 config
```

`rope_base` 没有单独的 CLI 开关（它不在 `apply_overrides` 里）。
想测得改 `ModelConfig.rope_base`。预期影响很小（0.001 量级），
因为 LLM 对 base 在 1e4~1e6 之间的取值都不敏感。

---

## 常见坑

### 坑 1：把 cos/sin 表按错误的切片传给 `forward`

```python
T0 = 0 if kv_cache is None else kv_cache.get_pos()      # ✅ 推理时要偏移
cos_sin = (self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T])
```

**推理（decode）时必须按 `cache_seqlens` 偏移。** 忘了这一步的症状是：
训练时正常，推理时胡言乱语。而且不会报错 —— 只是位置编码全错位。

对应的判据是 `test_core.py -k kv_cache`（第 34 章会讲）。

### 坑 2：传了 `(B, H, T, D)` 而不是 `(B, T, H, D)`

`apply_rotary_emb` 的 `assert x.ndim == 4` 只能挡住维数错误，
挡不住前三维顺序搞反。搞反时 `x.size(3)` 变成 `T`，
`d = T // 2`，`torch.cat` 出来形状错，在下游 attention 里报一个
无关的错误。

**记住判据**：`x.size(-1)` 必须是 `head_dim`。

### 坑 3：用 `arange(head_dim)` 而不是 `arange(0, head_dim, 2)`

前者给出 `d` 个下标而不是 `d/2` 个，`inv_freq` 长度错一倍，
`torch.outer` 的结果形状变成 `(L, d)`，后面 `x1` 是 `(..., d/2)`，
广播时直接报错。

### 坑 4：在 meta device 上建的 RoPE 表是垃圾内存

`GPT.__init__` 里创建的 `cos`/`sin` 如果是在 `meta` device 上建的，
`to_empty()` 只会分配未初始化的内存，不填任何值 —— 实测会变成 NaN，
整个模型 loss 立刻是 nan。

所以 `init_weights()` 里**必须重算** RoPE 表（`gpt.py` 里有一段
专门的警告注释）。这是 meta device 三步法最容易踩的坑，第 16 章详讲。

---

## 延伸

**RoPE vs ALiBi**：ALiBi 是另一种方案 —— 不旋转，而是在注意力
分数上加一个与距离成正比的偏置（`score += -m · (i - j)`）。
更简单，也不需要位置表，但表达力略弱。nanochat 曾经测过，
RoPE 更好。

**RoPE 的可外推性**：因为 `t·ω` 对 `t` 是线性的，把表算到
`10 × sequence_len` 就有了外推余量。本项目正是这么做的。
但超过 4 倍训练长度后仍会退化 —— 低频通道转过太多圈，
bf16/fp32 的角度精度不够。

**nanochat 的一个额外技巧**：部分模型会在 `q` 上加一个衰减掩码
（`qk_post_softnorm_mask`）来抑制注意力衰减。本项目没有实现，
`ModelConfig` 里也没有对应开关。

---

## 下一章

[第 10 章：注意力三步曲](10-注意力三步曲.md) ——
有了 q、k、v 和位置编码，现在可以把注意力算出来了。