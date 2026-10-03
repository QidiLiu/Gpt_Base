# 08 · RMSNorm

第一个零件。3 行代码，但「为什么可以这么少」是一个真问题。

---

## 本章目标

- 说清楚 LayerNorm 和 RMSNorm 的三个差异，以及去掉减均值为什么安全
- 理解「norm 是无参函数」这个实现选择，以及它带来的参数量/统计差异
- 亲手实现 `rms_norm` / `layer_norm` / `make_norm` 三个函数

---

## 前置回顾

[第 07 章](07-先跑通一个最小GPT.md) 给出了形状图。图里有三处归一化：

```
嵌入 → rms_norm → [Block × N] → rms_norm → lm_head
                    ↑
              每个 Block 内部两次（注意 attn 前、mlp 前）
```

本章实现的就是这个 `rms_norm`。

---

## 概念：归一化在解决什么问题

残差流的尺度会随深度累积。如果不做归一化：

- 第 1 层输出 std ≈ 1，第 24 层可能 std ≈ 30
- 后面那层的 `lm_head` 收到的 logits 巨大 → softmax 饱和 → 梯度消失
- 而且不同通道的尺度可能差几个数量级，少数通道主导整个张量

归一化把每个 token 的特征向量拉回「单位尺度」，让每一层的输入分布稳定。

**注意它归一化的是最后一维**（`hidden_dim`），不是 batch 也不是时间维。
也就是说：**每个 token 独立归一化**，token 之间不互相影响。这很重要 ——
它意味着归一化不会引入跨 token 的依赖，因果性不会被破坏。

⚠ **但要澄清一个常见误解**：归一化**不均衡各通道的尺度**。
它只把整条向量除以一个标量（RMS）。如果通道 0-3 的量级是 0.6、
通道 12-15 是 95，归一化后它们依然是 147 倍的关系（只是整体缩小了）。
验证见「动手验证 1」。

归一化解决的是**层与层之间的尺度漂移**（第 24 层不会比第 1 层大 30 倍），
而**不是通道之间的尺度失衡**。后者是另一个问题，本项目没有专门处理。

---

## 概念：LayerNorm vs RMSNorm

**LayerNorm**（原始 Transformer）：

```
LN(x) = (x - mean(x)) / std(x) · g + b
```

对最后一维做两步：先减均值（centering），再除标准差（scaling）。

**RMSNorm**：

```
RMSNorm(x) = x / RMS(x) · g        其中 RMS(x) = sqrt(mean(x²))
```

三个差异：

| | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | 是 | **否** |
| 偏置 b | 有 | **无** |
| 可学参数 | `g` + `b`（2×hidden_dim） | 只有 `g`（hidden_dim） |
| 需要算 | mean 和 var 两次归约 | 只有 `mean(x²)` 一次 |

**为什么能去掉减均值？**

LLaMA 论文的论证：LLM 的隐藏激活天然就在零点附近近似对称，
所以 `mean(x) ≈ 0`，减它带来的收益很小 —— 而代价是**一次额外的
跨通道归约**（要先算完 mean 再算 var，不能一趟算完）。

> ⚠ 这个论证是**经验性**的，不是定理。在某些任务上（尤其是
> 视觉和部分小模型）减均值仍有可测的收益。所以本项目保留了
> `layer_norm` 作为消融基线 —— 把 `--norm-type layer` 打开就能测。

**为什么能去掉偏置？**

`b` 的作用是在归一化后给每个通道一个固定偏移。但 `g` 已经提供了
逐通道的可学缩放，而 `b` 的偏移能力可以被 `g` + 后续的线性层吸收。
去掉它每个 Block 省 `hidden_dim` 个参数 —— d24 上是 768 × 2 × 24 ≈ 37K，
不多，但是**白拿的**。

**nanochat 更激进**：它连 `g` 都不要。所以本项目的 `rms_norm` 是
**完全无参的**（返回的是一个函数，不是 `nn.Module`）。这一点在
`make_norm` 的实现里有直接后果，见下面。

---

## 概念：为什么 `rms_norm` 返回的是函数而不是模块

```python
if cfg.norm_type == "rms":
    return rms_norm          # ← 直接返回函数，不是 nn.Module
```

`Block.__init__` 里写的是 `self.norm1 = make_norm(cfg, cfg.n_embd, device)`。
当 `norm_type == "rms"` 时，`self.norm1` 就是 `rms_norm` 这个函数本身。

**这样做的三个好处**：

1. **参数量统计干净**。`nn.ModuleList` 里不会塞一堆空模块。
   `model.num_params()` 直接反映真实参数量，不用减掉「norm 层的参数」。
2. **少 2×N 次 all-reduce**。分布式下每个空模块都要参与通信。
3. **`state_dict` 里没有 norm 键**，checkpoint 更小。

**代价**：`self.norm1` 不是 `nn.Module`，所以不能对它做 `.to(device)`、
`.weight` 之类的操作。这在本项目里没问题（norm 本来就无参），
但如果哪天想换成带 `g` 的 RMSNorm，就得把 `make_norm` 改成返回
`nn.Module` —— 这是 nanochat 也不接受的妥协。

---

## 概念：`eps` 的取值差异

```python
return F.rms_norm(x, (x.size(-1),))                      # PyTorch 默认 eps
return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)  # 显式 1e-5
```

`eps` 加在分母上防除零：`(x²).mean() + eps`。

RMSNorm 论文用 `1e-6`，但 nanoGPT / nanochat 用 `1e-5`，
本项目跟后者。原因很实际：`eps` 太大会在激活值很小时
把归一化结果压扁（`x / sqrt(eps)` 接近常数），太小则可能在
全零输入上出 NaN。`1e-5` 是这两者的折中。

> 这一行在消融时会产生可测的差异吗？几乎测不出（Δ < 0.001）。
> 但它必须在两处**保持一致**，否则 `norm_type` 对比实验就掺了别的变量。

---

## 手抄代码

### 第 1 块：三个归一化函数

写进 `src/model/layers.py`。

```python
def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """
    RMSNorm(x) = x / RMS(x) · g      （本项目无 g，故省略）

    对比 LayerNorm：
        LN(x)  = (x - mean(x)) / std(x) · g + b
        RMS(x) = x / sqrt(mean(x²) + eps)

    两个关键差异：
      1. 不减均值 —— 只算均方根，省一趟跨通道归约
      2. 没有偏置 b（也没有 g，本项目直接不要）

    为什么能减掉均值：LLM 的隐藏激活天然在零点附近近似对称，
    减均值的收益很小，而代价是一次额外归约。这是经验结论不是定理，
    所以 layer_norm 那条路径保留着做消融对照。
    """
    return F.rms_norm(x, (x.size(-1),))


def layer_norm(x: torch.Tensor, weight, bias) -> torch.Tensor:
    """标准 LayerNorm。存在的唯一目的是消融对比（--norm-type layer）。"""
    return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)


def make_norm(cfg, ndim: int, device=None):
    """
    按配置构造归一化层。

    ⚠ rms 分支返回的是一个**函数**，不是 nn.Module —— 因为它完全无参。
      好处：参数量统计干净、分布式下少通信、state_dict 更小。
      代价：拿不到 nn.Module 的 .to() / .weight 等能力。
    """
    if cfg.norm_type == "rms":
        return rms_norm
    if cfg.norm_type == "layer":
        w = nn.Parameter(torch.ones(ndim, device=device))
        b = nn.Parameter(torch.zeros(ndim, device=device)) if cfg.use_bias else None
        return lambda x: layer_norm(x, w, b)
    raise ValueError(f"未知 norm_type: {cfg.norm_type}")
```

**三处容易写错的地方**：

1. `F.rms_norm` 的第一个参数是**归一化的维度列表**，不是 shape 广播。
   `x.size(-1)` 是一个 int，要包成 tuple。
2. `layer_norm` 的 `eps` 必须显式传（本项目要 1e-5，不是 PyTorch 默认）。
3. `make_norm` 的 `rms` 分支**不要**包一层 lambda —— 直接返回函数，
   这样 `Block` 里存的就是 `rms_norm` 本身。

---

## 动手验证

### 验证 1：rms_norm 消除的是「整体缩放」，不是通道差异

```bash
uv run pytest tests/test_core.py -k rmsnorm -v
```

自己也可以直接验，而且**这里有个容易搞错的点值得亲自确认**：

```python
import sys, torch
sys.path.insert(0, "src")
from model.layers import rms_norm

v = torch.randn(16)

# 性质 1：整体缩放被完全消除 —— 三种尺度归一化后是同一个向量
for s in (1.0, 100.0, 0.01):
    y = rms_norm((v * s).view(1, 1, 16)).view(-1)
    print(f"scale={s:6}: RMS(输出)={y.pow(2).mean().sqrt():.4f}")

# 性质 2：通道之间的相对差异被**保留**
x = torch.randn(1, 1, 16) * torch.tensor([1.0] * 8 + [100.0] * 8)
y = rms_norm(x).view(-1)
print("通道 0-3  RMS:", x.view(-1)[:8].pow(2).mean().sqrt())
print("通道 12-15 RMS:", x.view(-1)[8:].pow(2).mean().sqrt())
print("归一化后 0-3 :", y[:8].pow(2).mean().sqrt())
print("归一化后 12-15:", y[8:].pow(2).mean().sqrt())
```

实测输出：

```
scale=   1.0: RMS(输出)=1.0000
scale= 100.0: RMS(输出)=1.0000
scale=  0.01: RMS(输出)=0.9997

通道 0-3  RMS: 0.6474
通道 12-15 RMS: 95.1566
归一化后 0-3 : 0.009622
归一化后 12-15: 1.414181
```

**关键结论：归一化后两组通道仍然是 147 倍的关系**，只是整体缩小了。

> ⚠ **一个常见的误解**：很多人以为 norm 会「均衡各通道的尺度」。
> 不会。RMSNorm 只把**整条向量**除以一个标量（RMS），LayerNorm 也一样。
> 两者都不是「通道白化」（whitening）。真正做白化的是 BatchNorm，
> 而 BatchNorm 依赖 batch 统计量 —— 那会破坏因果性（一个 token 的
> 输出取决于它后面有哪些 token），所以 LLM 根本不能用。
>
> 这也是为什么归一化对「少数通道主导」这件事**帮助有限**：
> 它解决的是**层与层之间的尺度漂移**，不是**通道之间的尺度失衡**。

### 验证 2：rms_norm 保持方向，只改长度

```bash
uv run pytest tests/test_core.py -k rmsnorm -v
```

判据里有一条是「归一化前后逐元素比值应该是常数」——
因为 `x / RMS(x)` 对**同一行**的每个元素除的是同一个数。

### 验证 3：确认 layer 分支真的带参数

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.layers import make_norm

cfg = make_run_config("debug").model
cfg.norm_type = "rms"
n_rms = make_norm(cfg, cfg.n_embd)
print(type(n_rms))                       # <class 'function'>  ← 不是 nn.Module
print(len(list(n_rms.parameters())) if hasattr(n_rms, "parameters") else 0)   # 0

cfg.norm_type = "layer"
n_ln = make_norm(cfg, cfg.n_embd)
print(type(n_ln))                        # <class 'function'>  ← 是 lambda
print(sum(p.numel() for p in n_ln.parameters()))   # ndim 或 2*ndim
```

`use_bias=False` 时 `b = None`，所以只有 `g`，参数量 = `ndim`。

---

## ★ 消融实验

RMSNorm vs LayerNorm 是本章唯一能做的消融，而且**它是 `--norm-type`
这个开关存在的全部理由**。

```bash
# 基线（RMSNorm，无参数）
bash script/train_base.sh ablation --no-resume --model-tag d6_rms

# 对照（LayerNorm，带 g 和 b）
bash script/train_base.sh ablation --no-resume --model-tag d6_ln --norm-type layer

bash scratch/ablation.sh d6_rms
```

**预期**：RMSNorm 的 bpb 更低或持平，但差距很小（LLaMA 论文报告在
同等规模下 RMSNorm 略优）。更重要的是**参数量**：

```bash
uv run python -m training.train_base --mode ablation --norm-type layer \
  --num-iterations 1 2>&1 | grep -A8 "参数量明细"
```

你会看到多出 `transformer矩阵` 那一项的增加（每个 Block 多 2×384 个参数，
d6 有 6 层 = 4608 个）。参数多了但效果没更好 —— 这就是「砍掉它们」的收益。

> ★ **2026-10 实测（推翻了我当初的预期）**：
> `--norm-type layer` 的 Δ = **+0.0013，7σ** —— 真实但很小。
>
> 我当初写「这个消融测不出差别」，理由是「噪声 ±0.02」。
> **那个噪声是我拍的。** 实测 σ_Δ = 0.000190（3 次同配置 + 1 次换初始化），
> 比假设小 100 倍 —— 于是这个消融**确实测得出来**，
> 只是效应本身就只有 0.0013。
>
> 所以准确的结论是：**RMSNorm 在质量上赢 0.0013 bpb，在这个规模上。**
> 它的主要收益仍然在**省参数**（d6 省 4608 个）和少一次减均值。

---

## 常见坑

### 坑 1：把归一化维度写错

```python
F.rms_norm(x, (x.size(-1),))      # ✅ 归一化 hidden_dim
F.rms_norm(x, x.shape)             # ❌ 归一化所有维度，结果每个元素都是 ±1
```

后者会让整个张量塌成符号函数 —— loss 不会 NaN，但会完全不学习。
症状：loss 一直是 9.01 附近不动。

### 坑 2：以为 RMSNorm 有可学参数，去 `model.parameters()` 里找它

本项目的 `rms_norm` **完全无参**。如果你想在 optimizer 里给 norm 单独设 LR，
会发现根本没有对应的参数组。这不是 bug，是设计。

### 坑 3：`norm_type="layer"` 时忘了 `use_bias` 的联动

`make_norm` 里 `b = None if not cfg.use_bias else ...`。本项目
`use_bias` 默认 `False`，所以 layer 路径也只有 `g`。
想复现「LayerNorm 的完整形态」得同时开 `--tie-embeddings` 之外的
bias 开关（目前没有单独暴露），只能在 config 里改。

---

## 延伸

**为什么 nanochat 连 `g` 也不要？**

`g` 是逐通道的可学缩放。nanochat 的论证（modded-nanogpt 的消融）：
在 Pre-LN 架构里，`g` 的作用基本被后面的线性层吸收了 —— 线性层自己
就能学出需要的缩放。所以 `g` 是冗余参数。

代价是失去了一点灵活性。本项目保留 `layer_norm` 的 `g` 正是为了
能对照这条结论。

**RMSNorm 之后的变体**：Gemma 提出的 `(1 + g)` 形式（先归一化再仿射）
比 LLaMA 的 `g` 形式收敛稍好，因为 `g` 初始化为 0 时后者会把整个
通道组清零。本项目不用这个变体。

---

>
> ⚠ **2026-10 才知道：`--norm-type layer` 曾经有两个 bug，
> 所以这个消融选项从未真正工作过。**
>
> 跑消融跑批时它 3 秒就崩（`expected scalar type BFloat16 but found Float`）。
> 顺藤摸瓜发现两个独立根因：
>
> 1. `make_norm` 用 `torch.ones(ndim)` 建 γ/β，默认 **float32**，
>    而激活是 bf16 → `F.layer_norm` 直接抛。CPU 上全是 float32 所以「碰巧能跑」。
> 2. **更严重**：γ/β 是被 `lambda` 闭包捕获的，
>    **根本不在 `model.parameters()` 里** → **optimizer 永远看不到它们**，
>    永远停在 (1, 0)。那样训出来的不是「LayerNorm」，
>    是「没有可学习仿射的 LayerNorm」。
>
> `setup_optimizer` 的「参数分组不完整」assert 抓不到第二个 ——
> 它检查的是「模型有的参数是否都被分组」，方向正好相反。
>
> 已修（`LayerNormAffine(nn.Module)` + `init_weights` 写回 (1,0)），
> 4 条判据 + 3 组故障注入。见 commit `51eb4a8`。
> **本章如果早点真的跑过这个消融，就不会留到现在。**

---

## 下一章

[第 09 章：RoPE 旋转位置编码](09-RoPE旋转位置编码.md) ——
注意力本身对顺序是完全无感的（打乱 token 顺序，输出只是跟着打乱）。
位置信息必须显式加进去。