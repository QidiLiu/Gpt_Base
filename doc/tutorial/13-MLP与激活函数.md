# 13 · MLP 与激活函数

注意力的另一半。它占了整个模型约 2/3 的参数 —— 而且是最「笨」的一块，
没有 KV cache、没有 GQA、没有位置编码。

---

## 本章目标

- 知道 MLP 为什么要升维再降维，而不是原地变换
- 理解 ReLU² 为什么是 nanochat/modded-nanogpt 的选择，以及它和 GELU 的取舍
- 亲手实现 `MLP.__init__` 和 `MLP.forward`

---

## 前置回顾

[第 12 章](12-QK-Norm与GQA.md) 完成了注意力。Attention 块的结构是：

```
x = x + Attn(norm(x))
```

现在加第二个子层：`x = x + MLP(norm(x))`。这就是完整的 Block，
第 14 章会讲残差连接本身的性质。

---

## 概念：为什么需要升维

先看「原地变换」为什么不够。假设我们要一个 D→D 的矩阵
`x → W·x`（没有非线性）：

- 无论堆多少个这样的层，合起来仍然是**一个**线性变换
  （`W2(W1 x) = (W2W1) x`）
- 加了 bias 也只是仿射变换

**所以非线性是必需的。** 但如果只在 D 维空间里做非线性，
表达力有限（每层只能学一个 D→D 的非线性映射）。

升维到 4D 再降回 D 的好处：

```
D ──c_fc──> 4D ──非线性──> 4D ──c_proj──> D
           ↑ 这里是「非线性作用的地方」
```

非线性作用在 4D 空间上，能构造出多得多的决策边界。
**关键在于这个 4 倍的「中间空间」** —— 它是表达力的来源。

> **为什么是 4 倍？** 经验值。nanoGPT / nanochat 都用 4。
> 也有用 8 的（如一些 MoE 模型），参数量翻倍但收益不明显。
> 有用 `SwiGLU` 风格的（把 4D 拆成 3 份做门控），Gemma / LLaMA 用这个 ——
> 同样参数下效果更好但需要 3 个矩阵。

---

## 概念：参数量算一遍

`full` 档（D=768）：

```
Attention:  c_q, c_k, c_v, c_proj  =  4 × 768 × 768  =  2.36M
MLP:        c_fc (768→3072)        =  768 × 3072    =  2.36M
            c_proj (3072→768)       =  3072 × 768   =  2.36M
                                        合计 4.72M
```

所以 **MLP 占一个 Block 的 2/3**（4.72 / 7.08 = 66.7%）。

24 层的话：

```
Attention  2.36M × 24 = 56.6M
MLP        4.72M × 24 = 113.2M      ← 全部矩阵参数的 2/3
```

这就是为什么「优化器怎么分组」（第 27 章）时 MLP 是主要战场 ——
它占了参数量的大头，权重衰减、学习率的选择对它影响最大。

---

## 概念：ReLU² vs GELU

### 三个候选

**GELU**（原始 Transformer）：

```
GELU(x) = x · Φ(x)        Φ = 标准正态 CDF
```

平滑、处处可导。**但需要算 erf/exp** —— 这是一个超越函数，
在 GPU 上不是单条指令。

**ReLU²**（本项目默认）：

```
ReLU²(x) = max(0, x)²
```

分段二次。导数：

```
d/dx ReLU²(x) = 2x    (x > 0)
             = 0     (x <= 0)
```

**极其便宜** —— 就是 `relu` 加一次平方，两条指令。

**SwiGLU**（LLaMA / Gemma）：

```
SwiGLU(x) = Swish(x_gate) ⊙ x_up       拆成 3 个矩阵
```

### 为什么选 ReLU²

modded-nanogpt / nanochat 方向上的第三方整理给出的对比
（来源见文末，**不是 nanochat 官方仓库的数字**）：

| 激活 | 验证 loss | 训练速度 | 额外参数 |
|---|---|---|---|
| GELU | 2.845 | 1.00× | 0 |
| **ReLU²** | **2.838** | **1.15×** | **0** |
| SwiGLU | 2.822 | 0.85× | +50% |

三个数字各自说明一件事：

- **ReLU² 比 GELU 好一点点**（2.838 vs 2.845）**且快 15%** —— 纯赚
- **但 SwiGLU 质量最好**（2.822），代价是慢 15% 且参数多 50%
- 也就是说 ReLU² 不是「最优」，是**性价比最优**

> ⚠ 这张表来自一篇第三方博客对 modded-nanogpt 的整理，
> 我**没有**在本仓库里跑出这些数字，也不建议把它当成精确引用
> （不同配置下顺序可能变）。要自己确认就跑下面的消融。
> 可靠的部分是「ReLU² 和 GELU 差距在 1% 量级」这个量级判断。

不过要注意一个细节：`F.relu(x).square()` 是**两次 kernel 启动**，
而 `F.gelu` 是一次。MLP 的算力大头是那个 `D×4D` 的矩阵乘，
所以激活函数只占很小一块 —— 实测差距 15% 主要来自
**kernel 启动和内存带宽，不是算术量**。真正的优势在于
**没有超越函数**：`exp`/`erf` 在 bf16 上有精度损失，而平方是精确乘法。

> 一个容易忽略的点：`ReLU²` 的输出非负。这意味着 MLP 的输出
> 永远非负 —— 但因为外面接了 `c_proj`（带符号的权重）和残差连接，
> 所以不影响整个残差流的性质。

### 开关怎么用

```bash
# 基线（relu2）
bash script/train_base.sh ablation --no-resume --model-tag d6_relu2

# 换成 GELU
bash script/train_base.sh ablation --no-resume --model-tag d6_gelu --activation gelu
```

`ModelConfig.activation` 实际支持三个值：

| 值 | 对应 | 备注 |
|---|---|---|
| `"relu2"` | `F.relu(x).square()` | 默认 |
| `"gelu"` | `F.gelu(x)` | 精确 erf 版 |
| `"gelu_tanh"` | `F.gelu(x, approximate="tanh")` | tanh 近似 |
| 其他 | `ValueError` | 见验证 3 |

> ⚠ **写这一章时发现并修掉的一处不一致**：`config.py` 第 57 行的注释写的是
> `activation: str = "relu2"     # "relu2" | "gelu"`，
> `train_base.py` 的 CLI help 也是 `relu2|gelu` ——
> **两处都漏了 `gelu_tanh`**，而 `MLP.forward` 实际是支持的。
> 三处都已改。
>
> 这正是「同一件事描述了三遍」的典型代价：`forward` 是唯一
> 真正决定行为的代码，另两处都是**可能过时的副本**。
> 手写代码时照抄注释里的候选列表，会漏掉一个能用的选项。
>
> **记住：候选值的权威来源是 `forward` 的 `if/elif` 分支，
> 不是任何注释。** 现在 `config.py` 的注释里也写了这句。

---

## 手抄代码

### 第 1 块：`MLP`

```python
class MLP(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        d = cfg.n_embd
        self.activation = cfg.activation
        self.c_fc = Linear(d, 4 * d, bias=cfg.use_bias, device=device)
        self.c_proj = Linear(4 * d, d, bias=cfg.use_bias, device=device)

    def forward(self, x):
        x = self.c_fc(x)
        if self.activation == "relu2":
            x = F.relu(x).square()
        elif self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "gelu_tanh":
            x = F.gelu(x, approximate="tanh")
        else:
            raise ValueError(f"未知 activation: {self.activation}")
        return self.c_proj(x)
```

**三处容易写错的地方**：

1. **`c_proj` 的方向不能反**。必须 `4d → d`（降维）。
   写成 `d → 4d` 的话输出形状会变成 `(B,T,4D)`，在
   `x + self.mlp(...)` 的残差相加处报形状错误。
2. **激活放在两个线性之间**。放在 `c_fc` 之前的话，
   那就变成了「先激活再降维」，结构完全不同（且浪费参数）。
3. **`else` 分支要抛异常**。静默返回 `x`（等于没做非线性）
   是很难查的 bug —— loss 仍然在降，只是永远学不到东西。

---

## 动手验证

### 验证 1：确认 MLP 占了 2/3 的参数

```bash
uv run pytest tests/test_core.py -k mlp_activation -v
```

自己也可以数：

```python
import sys
sys.path.insert(0, "src")
from common.config import make_run_config
from model.layers import MLP, CausalSelfAttention

cfg = make_run_config("full").model
mlp = MLP(cfg)
attn = CausalSelfAttention(cfg, layer_idx=0)
n_mlp = sum(p.numel() for p in mlp.parameters())
n_attn = sum(p.numel() for p in attn.parameters())
print(f"MLP      {n_mlp:>10,}")
print(f"Attention{n_attn:>10,}")
print(f"MLP 占比  {n_mlp / (n_mlp + n_attn):.1%}")
```

**期望**：约 66.7%。

### 验证 2：三个激活函数的形状和数值差异

```python
import sys, torch
sys.path.insert(0, "src")
import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print("x        :", x.tolist())
print("relu2    :", F.relu(x).square().tolist())
print("gelu     :", [round(v, 4) for v in F.gelu(x).tolist()])
print("gelu_tanh:", [round(v, 4) for v in F.gelu(x, approximate="tanh").tolist()])
```

**观察**：
- `relu2` 在 `x <= 0` 时**精确为 0**，没有小尾巴
- `gelu` 在 `x = -1` 时是 −0.159（负值！），`x = -3` 时才接近 0
- 三者输出**形状完全相同** —— 切换 `activation` 不会改变模型结构，
  所以 checkpoint 可以跨 activation 加载（虽然那样没意义）

### 验证 3：非法 activation 会报错

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.layers import MLP

cfg = make_run_config("debug").model
cfg.activation = "swiglu"      # 没实现
mlp = MLP(cfg)
try:
    mlp(torch.randn(2, 4, cfg.n_embd))
except ValueError as e:
    print("正确报错:", e)
```

---

## ★ 消融实验

这是本章唯一、也是 nanochat 实际做过的消融：

```bash
# 基线：ReLU²
bash script/train_base.sh ablation --no-resume --model-tag d6_relu2

# GELU（精确 erf 版）
bash script/train_base.sh ablation --no-resume --model-tag d6_gelu --activation gelu

# GELU（tanh 近似版）
bash script/train_base.sh ablation --no-resume --model-tag d6_gelu_tanh --activation gelu_tanh

bash scratch/ablation.sh d6_relu2
```

**预期**：三者的 bpb **几乎一样**（Δ 很小）。>
> ★ **2026-10 提醒**：上面这个「预期」依赖的噪声假设是错的。
> `ablation` 档实测 σ_Δ = 0.000190（3 次同配置 + 1 次换初始化），
> 显著性阈值 0.001 —— 而不是早期文档里写的 ±0.02（从没测过）。
> **在 d6 上 0.001 量级的效应是测得出来的**，
> 真正测不出的是「训练步数不够」（`smoke` 档 896 步那种情况，
> 它会把 QK-Norm 消融的**符号测反**）。

**这个消融的价值不在 bpb 差值**，而在两点：

1. **它验证了「激活函数不是小模型上的关键因素」** —— 这是一个
   有价值的负结果。如果你想在 nanochat 上找一个「大效应」来做消融，
   激活函数不在名单里。
2. **它验证了 `--activation` 开关真的生效**。⚠ 这类开关最常见的
   失效模式是「只出现在打印字符串里」。`progress.sh` 里有
   `-k 'mlp_activation'` 的判据专门盯着这件事。

### 想看到真正的差异：放大参数量

⚠ **本项目从未测过激活函数的消融**，所以这里不给数字。

> ★ **2026-10 更正**：我原来在这里写的是「效应在 d6 上测不出来」。
> 那句话的依据是「噪声带 0.02」—— ★ 而那个噪声是拍的，
> 实测 σ_Δ 只有 0.000190。**所以「测不出来」这个理由不成立**，
> 正确的说法是「**没测**」。
>
> 这两者的区别正是本项目的核心纪律：**没测 ≠ 测不出。**
>
> 真要测，d6 档（39 分钟）就够，命令是：
>
> ```bash
> bash script/train_base.sh ablation --no-resume \
>      --model-tag d6_gelu --activation gelu
> bash script/train_base.sh ablation --no-resume \
>      --model-tag d6_gelu_tanh --activation gelu_tanh
> bash scratch/ablation.sh d6_base
> ```
>
> 本项目没跑，所以**不要在文档里写一个没实测过的「ReLU² 更好 0.0x bpb」** ——
> 这个要求不变，只是理由从「测不出」变成了「还没测」。

---

## 常见坑

### 坑 1：`c_proj` 方向写反

见上面。症状是残差相加时报形状不匹配，
错误信息里会出现 `4` 这个倍数，很容易定位。

### 坑 2：激活函数放在残差连接外面

```python
x = self.mlp(x) + x           # ❌ 错了
x = x + self.mlp(self.norm2(x))   # ✅
```

区别在于：前者的残差路径上有一个非线性，梯度回传时会被激活函数的
导数反复缩放 —— 深层会不稳定。

### 坑 3：以为 ReLU² 是「一次操作」

`F.relu(x).square()` 是两次（relu + 平方）。相比 `F.gelu` 的
一次（含 exp），算术量其实更多。

真正的优势是**没有超越函数**：bf16 的 `exp` 有精度问题，
而平方是精确的乘法。实测 `gelu` 和 `gelu_tanh` 的差异主要来自
`erf` vs `tanh` 的近似误差。

### 坑 4：`square()` 在 fp16 下溢出

`ReLU²` 的输出最大是输入的平方。如果 `c_fc` 的输出意外很大
（比如初始化问题），`x²` 在 fp16 下会溢出成 `inf`。

本项目用 bf16（动态范围更大）且有 QK-Norm + rms_norm 兜底，
所以不会遇到。但如果改成 fp16，需要给 `square` 之前加 clamp。

---

## 延伸

> **本章引用的第三方来源**：ReLU² 的对比数字来自
> José David Baena 的 *Modern Transformer Architecture: RoPE, QK Norm,
> and Design Choices*（2025-10），那篇文章是对 modded-nanogpt /
> nanochat 消融的整理。ReLU² 的原始出处是
> So et al. 2022, *Primer: Searching for Efficient Transformers*
> （arXiv:2202.08906）。

**为什么 ReLU 系在 LLM 上重新流行**

2023 年之前主流是 GELU / SwiGLU，因为 ReLU 被认为「在深层
Transformer 上表现不佳」（这是 2015 年一批实验的结论）。
但那些实验用的是 Post-LN 架构。

Pre-LN + 残差连接改变了方程：ReLU 的负半轴「死亡」问题
被残差旁路救回来了。modded-nanogpt 的实验证明 ReLU² 在
Pre-LN 架构上完全没问题，而且更快。

**参数量与计算量的比例**

MLP 的 FLOPs 是 `2 × D × 4D × 2`（前向两次矩阵乘，每 token）
= `16 D²`，而 Attention 的投影也是 `8D²`，但注意力的
`QK^T` 和 `AV` 又各加 `4Dh·T`。

所以 MLP 的**参数**是 2/3，但**计算量**占比更复杂 ——
它与序列长度无关，而注意力的 QK^T/AV 与 T 成正比。
长上下文时注意力反而占更多计算。

---

## 下一章

[第 14 章：残差流与 Pre-LN](14-残差流与Pre-LN.md) ——
把注意力 和 MLP 拼起来的那个 `+`。它决定了 24 层能不能训得起来。