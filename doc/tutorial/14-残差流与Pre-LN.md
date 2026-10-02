# 14 · 残差流与 Pre-LN

注意力 和 MLP 都有了。剩下的那个 `+` 决定了 24 层能不能训得起来。

---

## 本章目标

- 说清 Pre-LN 和 Post-LN 的差别，以及为什么现代 LLM 全用 Pre-LN
- 理解「残差流」这个比喻，以及它为什么让深层可解释
- 亲手实现 `Block`

---

## 前置回顾

- [第 10 章](10-注意力三步曲.md)：注意力
- [第 13 章](13-MLP与激活函数.md)：MLP

现在要把它们拼成 `Block`。

---

## 概念：两种写法

**Post-LN**（原始 Transformer 论文）：

```python
x = norm1(x + attn(x))
x = norm2(x + mlp(x))
```

**Pre-LN**（nanoGPT / nanochat / 现代 LLM 全部）：

```python
x = x + attn(norm1(x))
x = x + mlp(norm2(x))
```

差别只是 **norm 在残差连接的里面还是外面**。但这个位置选择
决定了两件根本性的事。

---

## 概念：Pre-LN 为什么能训深层

看梯度怎么回传。Pre-LN 下：

```
x_{l+1} = x_l + f(norm(x_l))
```

对 `x_l` 求导：

```
dx_{l+1}/dx_l = I + f'(norm(x_l)) · dnorm/dx_l
                  ↑
               单位矩阵
```

**恒等项 `I` 保证了梯度可以「原样」穿过任意多层** ——
一个 24 层的网络，最深那层的梯度里一定包含一条「直接从这里
回到输入」的路径，不会被任何非线性衰减掉。

Post-LN 下：

```
x_{l+1} = norm1(x_l + attn(x_l))
```

梯度必须穿过 `norm1` 的导数才能回到上一层。而 RMSNorm 的导数
含 `1/RMS(x)` 这个因子 —— 如果 `RMS(x)` 很大（激活值炸了），
这一层的梯度会被严重压缩。**层数越多，衰减越累积。**

这就是为什么 Post-LN **必须靠 warmup 救**：训练早期权重随机、
`RMS(x)` 偏大，梯度传不过去，先用小学习率把权重训稳。

> nanochat 用 Post-LN 时 warmup 要占训练步数的相当比例；
> Pre-LN 下本项目只 warmup 40 步（`TrainConfig.warmup_steps`）。

**代价**：Pre-LN 的表示能力略弱。因为 norm 在残差**外面**，
主路径上的 x 从不被重新归一化，多路信息相加时没有门控
（Post-LN 的每次 norm 相当于一次「重整」，带一点门控效果）。

现代 LLM 几乎清一色选 Pre-LN —— 因为深度带来的稳定性收益
远大于那一点表示力。

---

## 概念：「残差流」这个比喻

nanochat 的注释里把 `x` 叫 **残差流（residual stream）**。这个
比喻非常贴切，因为它描述了 x 的真实角色：

```
x_0  = 嵌入 + RoPE 信息
        │
        ├─→ norm1 → attn  ──┐
        │                     ├── 加到 x 上
        ├─→ norm2 → mlp   ──┘
        │
      x_1  = 上述累加
        │
       ...×24
        │
      x_24
        │
        └─→ norm → lm_head → logits
```

**x 是唯一贯穿全网的「信息总线」。** 每个 Block 只是往这条总线上
「加」自己算出来的东西，从不修改已有的内容。

这个视角带来的三个理解：

1. **注意力是「读」**：从总线上读上下文，写回总线
2. **MLP 是「写」**：从总线上读自己那一份，写回总线
3. **每个通道组都有「自己的解释」**。这是「可解释性」研究方向的基础：
   不同的通道组分别编码了语法信息、指代关系、位置信息……
   参见第 17-19 章的 resid_lambdas / x0_lambdas / value_embeds ——
   它们全都是在**操纵这条总线上的信号**。

### 为什么 x₀ 要立刻归一化

```python
x = self.transformer.wte(idx)
x = rms_norm(x.to(COMPUTE_DTYPE))      # ← 这一行
```

因为嵌入的尺度直接决定了整条总线的起点。`wte` 的初始化是
`std=0.8`（`init_weights` 里），而后续每层都会往上加东西 ——
如果起点尺度没钉住，24 层累积后的尺度可能差几十倍。

**总线的尺度只增不减**（残差都是加法），所以在起点归一化
特别重要。nanochat 在最后还有一次 `rms_norm`（在 `lm_head` 之前），
同样是为了把尺度拉回来。

---

## 概念：两个 norm 的位置

```python
def forward(self, x, cos_sin, window, kv_cache=None, ve=None):
    x = x + self.attn(self.norm1(x), cos_sin, window, kv_cache, ve)
    x = x + self.mlp(self.norm2(x))
    return x
```

`norm1` 在注意力之前，`norm2` 在 MLP 之前。**每个子层都有自己的
norm** —— 这不是可选项：

- 共享一个 norm：两个子层看到不同尺度的输入（因为第一个已经往
  总线上加过东西了）
- 只在末尾 norm：中间层的输入尺度会失控

本项目这两个 norm 都是**无参的 `rms_norm`**（第 08 章），
所以 `self.norm1` 存的是函数不是模块。

---

## 手抄代码

### 第 1 块：`Block`

```python
class Block(nn.Module):
    def __init__(self, cfg, layer_idx: int, device=None):
        super().__init__()
        self.norm1 = make_norm(cfg, cfg.n_embd, device)
        self.attn = CausalSelfAttention(cfg, layer_idx, device)
        self.norm2 = make_norm(cfg, cfg.n_embd, device)
        self.mlp = MLP(cfg, device)

    def forward(self, x, cos_sin, window, kv_cache=None, ve=None):
        # ★ 注意参数顺序：attn 拿 cos_sin（要算 RoPE），
        #   mlp 不需要 —— 所以不能写成 x = x + self.mlp(self.norm1(x))
        x = x + self.attn(self.norm1(x), cos_sin, window, kv_cache, ve)
        x = x + self.mlp(self.norm2(x))
        return x
```

**三个容易写错的地方**：

1. **`norm1` 给 attn，`norm2` 给 mlp**。反了的话虽然不会报错
   （维度相同），但每个子层拿到的输入不是「干净」的那份。
2. **`cos_sin` 只传给 attn**。MLP 不需要位置信息 —— 它是
   **逐 token 独立**的（没有跨 token 的运算）。
3. **残差加的是子层的输出，不是输入**。写成
   `x = self.attn(...) + x` 意思一样，但写成
   `x = self.attn(norm1(x))` 就丢了残差。

### 第 2 块：`BlockList` 的组装（在 `gpt.py`）

```python
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(padded, cfg.n_embd, device=device),
    "h": nn.ModuleList([Block(cfg, i, device) for i in range(cfg.n_layer)]),
})
```

⚠ 用 `nn.ModuleDict` 而不是直接 `nn.Embedding` + `nn.ModuleList`
两个属性 —— 命名空间的约定（`transformer.wte.*` / `transformer.h.0.*`）
让 checkpoint 的键名可读，而且 `named_parameters()` 的顺序稳定
（影响优化器分组，见第 27 章）。

---

## 动手验证

### 验证 1：残差流的尺度只增不减

这是本章最有价值的验证 —— 它能让你「看见」残差流。

```bash
uv run pytest tests/test_core.py -k causal -v
```

自己也可以追踪每层的输出尺度：

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config, OptimConfig
from model.gpt import build_model
from model.layers import precompute_rope
from optim.muon import setup_optimizer

cfg = make_run_config("ablation").model          # d6，6 层
model = build_model(cfg, device="cuda")
opt = setup_optimizer(model, OptimConfig())
cos, sin = precompute_rope(cfg.sequence_len, cfg.head_dim,
                           base=cfg.rope_base, device="cuda")
torch.manual_seed(0)

def trace(tag):
    x = torch.randn(1, cfg.sequence_len, cfg.n_embd, device="cuda")
    prev = x.pow(2).mean().sqrt().item()
    out = []
    with torch.no_grad():
        for i, b in enumerate(model.transformer.h):
            x = b(x, (cos, sin), cfg.window_sizes()[i])
            r = x.pow(2).mean().sqrt().item()
            out.append(r / prev)
            prev = r
    print(tag + " " + "  ".join("%.4f" % v for v in out))

print("层数  " + "  ".join("%5d" % i for i in range(cfg.n_layer)))
trace("init  ")
V = cfg.vocab_size
for _ in range(15):                                # 训 15 步
    xb = torch.randint(0, V, (4, cfg.sequence_len), device="cuda")
    loss = model(xb, xb); loss.backward(); opt.step(); model.zero_grad(set_to_none=True)
trace("step15")
```

实测：

```
层数      0      1      2      3      4      5
init   1.0000 1.0000 1.0000 1.0000 1.0000 1.0000
step15 1.0456 1.0469 1.0486 1.0490 1.0480 1.0471
```

**两个观察，第二个更重要**：

1. **训练后每层的 RMS 只增不减**（比值都 ≥ 1），每层只涨 4-5%。
   这是 Pre-LN + 最后那次 `rms_norm` 的效果 —— 尺度缓慢增长，
   最后被拉回来。

2. **初始状态下每一层的比值精确等于 1.0000**。这不是巧合 ——
   `init_weights` 里把 `c_proj` 初始化为 **0**：

   ```python
   torch.nn.init.zeros_(block.attn.c_proj.weight)   # 恒等
   torch.nn.init.zeros_(block.mlp.c_proj.weight)   # 恒等
   ```

   所以**刚建好的模型里，每个 Block 都是严格的恒等映射**，
   残差流完全不变化。第 16 章会讲这个初始化的用意。

> ⚠ 写这段验证时有两个坑：
> ① `precompute_rope` 要传 `device="cuda"`，否则 cos/sin 在 CPU 上，
>    和 cuda 上的 q/k 相乘时报 `Expected all tensors to be on the same device`。
> ② 随机输入的 `T` 必须和 `precompute_rope` 的 `seq_len` 一致，
>    否则报 `size of tensor a (64) must match tensor b (128)`。

### 验证 2：Pre-LN 的梯度能穿过任意深度

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import build_model

cfg = make_run_config("ablation").model       # d6，够深且快
model = build_model(cfg, device="cuda")

x = torch.randint(0, cfg.vocab_size, (2, 32), device="cuda")
emb = model.transformer.wte(x)
loss = model(x, x).log()
loss.backward()

# 看第一层和最后层的梯度范数之比
g0 = model.transformer.h[0].mlp.c_proj.weight.grad.norm().item()
gN = model.transformer.h[-1].mlp.c_proj.weight.grad.norm().item()
print(f"第 0 层梯度 {g0:.4e}")
print(f"第 {cfg.n_layer-1} 层梯度 {gN:.4e}")
print(f"比值 {g0 / gN:.2f}   （Pre-LN 下不应该差几个数量级）")
```

**期望**：比值在 `1e-2 ~ 1e1` 量级。如果是 `1e-4` 以下，
说明梯度在衰减 —— 那通常是 Post-LN 或者初始化有问题。

---

## ★ 消融实验

本章没有开关（Pre-LN 是硬编码的 `Block.forward` 结构）。
但有一个**结构对照**可以做：

```bash
# 基线：Pre-LN（默认）
bash script/train_base.sh ablation --no-resume --model-tag d6_prenorm

# 手动改成 Post-LN（需要编辑 src/model/layers.py 的 Block.forward）
# x = self.norm1(x + self.attn(x, cos_sin, window))
# x = self.norm2(x + self.mlp(x))
bash script/train_base.sh ablation --no-resume --model-tag d6_postnorm

bash scratch/ablation.sh d6_prenorm
```

**预期**：在 d6（6 层）上 Post-LN 也能训，甚至可能略好
（它的表示力更强）。**要到 d20+ 才看得出 Pre-LN 的稳定性优势**，
而且那时 Post-LN 通常需要更长的 warmup。

> 这个消融**不要在 `full` 档上做**（41 小时 × 2）。
> 而且它的结论依赖深度 —— 拿 d6 的结果去推断 d24 是错的。

---

## 常见坑

### 坑 1：把 `norm1` 和 `norm2` 反了

不报错，维度一样，只是每个子层拿到的输入不「干净」。
这类 bug 只能靠「删掉一个 norm 看看 bpb 变化」来发现。

### 坑 2：给 MLP 也传 `cos_sin`

```python
x = x + self.mlp(self.norm2(x), cos_sin)     # ❌ MLP.forward 没有这个参数
```

会报 `TypeError`。但如果你为了「统一接口」给 MLP 也加一个
`cos_sin` 参数却不用它，那会浪费计算（虽然只是传个引用）。

**MLP 是逐 token 独立的** —— 这是一个重要的结构性事实，
也是它比注意力便宜的原因之一。

### 坑 3：残差连接写成 `x + attn(...)` 时 `attn` 里又用了 `x`

看代码：`self.attn(self.norm1(x), ...)` —— 传进注意力的是
**归一化后的 x**，不是原 `x`。写成 `self.attn(x, ...)` 就丢了 norm。

### 坑 4：以为残差是「恒等映射所以不改变分布」

残差主干确实有恒等路径，但**每个子层的输出会往上叠加**。
实测训练 15 步后每层 `RMS` 涨 4.5%（验证 1）——
不多，但也不是零。⚠ 那个测量是在 `ablation` 档（6 层）上做的，
**24 层的累计增长我没有实测**（按每层 4.5% 线性外推约 2.9 倍，
但真实模型的每层增益会随训练变化，别把这个外推当数据）。

真正让它不失控的是三件事：
1. Pre-LN 的 norm 把每个子层的**输入**钉住
2. `c_proj` 初始化为 0（第 16 章），所以初始时每层输出为 0
3. 最后的 `rms_norm` 把总线的尺度拉回来给 `lm_head`

---

## 延伸

**残差流的「可解释性」视角**

Elhage et al. 的 *Anatomy of a Large Language Model*（2023）把
x 的各个通道组解释为「残差流上的不同信息」：
某些通道组编码语法，某些编码词性，某些编码位置。
后续工作（logit lens、activation patching）建立在这个视角上。

**DeepNet / ReZero / LayerScale**

几个「让深层更容易训」的方案：

- **ReZero**（微软）：每个残差分支乘一个可学标量，初始化为 **0**。
  本项目的 `resid_lambdas`（第 17 章）本质就是这个，
  只不过它初始化为 1.15 而不是 0
- **LayerScale**（Google）：分支乘一个每通道的可学缩放
- **DeepNet**（Meta）：把残差分支的输出在**最后**归一化

它们的共同思路：**给残差分支一个「音量旋钮」**，让网络自己
学每层该往上加多少。本项目选了最激进的版本（第 17-19 章）。

**为什么没有「残差分支的 dropout」**：有，但 LLM 很少用 ——
它会破坏「总线只增不减」的性质，而后面的 trick（第 17 章的
`resid_lambdas`）正是建立在这个性质上的。

---

## 下一章

[第 15 章：权重绑定与 logit softcap](15-权重绑定与logit-softcap.md) —
模型的最后一步：把 `(B,T,768)` 的残差流变成 `(B,T,16384)` 的 logits。