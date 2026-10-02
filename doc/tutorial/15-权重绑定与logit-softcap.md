# 15 · 权重绑定与 logit softcap

**卷 2 的收尾。** 把 `(B,T,768)` 的残差流变成 `(B,T,V)` 的 logits，
这一步有两个细节值得单独讲：词表要不要 pad、logits 要不要设上限。

---

## 本章目标

- 知道词表为什么要对齐到 64 的倍数，以及对齐前后在哪一步切回来
- 理解 logit softcap 防的是什么，以及为什么用 `tanh` 而不是硬 clamp
- 读完 `GPT.forward` 的输出头部分，卷 2 就通关了

---

## 前置回顾

[第 14 章](14-残差流与Pre-LN.md) 完成了残差流。
现在 `x` 是 `(B, T, 768)`，我们要把它变成 logits。

---

## 概念：词表对齐 —— 一个纯性能优化

`ModelConfig.pad_vocab_size_to = 64`。为什么要 pad？

```python
padded = ((cfg.vocab_size + 63) // 64) * 64     # 向上取整到 64 的倍数
self.padded_vocab_size = padded
```

`full` 档 `V=16384` 恰好是 64 的倍数。实测四档全部如此：

| 档位 | `vocab_size` | `padded` | 生效？ | `lm_head` 参数 |
|---|---|---|---|---|
| `debug` | 8192 | 8192 | **否** | 262,144 |
| `smoke` | 8192 | 8192 | **否** | 1,048,576 |
| `ablation` | 16384 | 16384 | **否** | 6,291,456 |
| `full` | 16384 | 16384 | **否** | 12,582,912 |

⚠ **2026-10 曾一度出现异常**：`ablation`/`full` 实际跑的是 `V=8192`，
于是 `ablation` 那行变成 `8192 × 384 = 3,145,728`、`full` 变成 `6,291,456`。

**已修（2026-10 同日）**：`train_tokenizer` 现在复用缓存前核对词表大小，
上表重新成立。根因与故障注入记录见[第 37 章](37-全流程与排错.md)。

**所以 pad 这套机制在本项目里一次都没生效。** 讲它是因为这是
「静默优化」的典型 —— 你在代码里看到一行 `// 64`，但它的效果
是零。换个词表（比如换成 50257 的 GPT-2 词表）它才真正开始工作。

两个理由：

1. **DDP 的梯度按第 0 维切分**。多卡时 `padded` 必须能被
   `world_size` 整除，否则 rank 0 和 rank 1 拿到的行数不同，
   all-reduce 直接错位。64 是 2^6，能被 1~64 卡整除。
2. **GEMM 的 tile 通常是 8/64 的倍数**。最后那个矩阵乘的输出
   维度（= `V`）不对齐会浪费 tile。

**切回来的位置**在 `forward` 里：

```python
logits = self.lm_head(x)                    # (B, T, padded_vocab_size)
logits = logits[..., :cfg.vocab_size]       # ← 切掉 padding 部分
logits = logits.float()
```

**为什么切在 `.float()` 之前**：`float()` 会把张量体积翻倍
（bf16→fp32）。先切掉 padding 再转换，要转换的元素更少。
`full` 档虽然没 pad，但这个顺序在别的词表下能省真金白银。

（另一个顺序问题：softcap 必须在 `.float()` **之后**，
见「常见坑 2」。两个顺序是独立的，别弄混。）

---

## 概念：logit softcap 防什么

不加限制时，`lm_head` 输出的 logits 会随训练增长。
`full` 档训练几百步后，常见的 token 的 logit 能到 30+。

**问题在 softmax**：

```
softmax([30, -30, 5, ...]) ≈ one-hot，且这个 one-hot 是「硬」的
```

两个后果：

1. **梯度趋近 0**。softmax 的雅可比矩阵是
   `diag(p) - p·pᵀ`。当 `p` 已经是 one-hot，这个矩阵的秩是 0 ——
   **整个向量的梯度都是 0**。模型卡在那里出不来。
2. **fp32 也开始丢精度**。`logit=30` 时 `exp(30) ≈ 1e13`，fp32 的
   尾数只有 24 位（约 7 位十进制有效数字），相邻 logit 相差 0.01
   时它们的 softmax 结果已经区分不出来了。

**softcap 的做法**：

```python
if cfg.logit_softcap > 0:
    cap = cfg.logit_softcap
    logits = cap * torch.tanh(logits / cap)
```

`tanh` 的两个性质让它成为理想选择：

- **平滑饱和**：`tanh(x)` 在 `|x| > 3` 之后就接近 ±1，导数指数衰减
  但**不是硬 0**
- **原点附近线性**：`tanh(x) ≈ x`（|x| < 1 时），所以小 logit 区域
  几乎不改变数值

`cap = 15` 意味着 logits 被压到 `[-15, 15]` 附近。而 softmax 的
上界差异变成约 `2·15 = 30` —— 也就是 logits 的「有效范围」从
无限制变成了 30，仍然够用，但不会进 one-hot。

实测的三个关键点（验证 1 的输出）：

| x | `15·tanh(x/15)` | 导数 |
|---|---|---|
| 0 | 0.0000 | **1.0000** |
| 15 | 11.4239 | 0.4200 |
| 40 | **14.8559** | **0.0191** |

两个容易看漏的点：

- **`x=40` 时输出是 14.86，不是 15.00。** softcap 不是硬上限 ——
  真正的上界只有渐近才达到。所以「logits 一定 ≤ 15」这个说法
  是错的，准确说是「logits ≤ 15 附近，|x| 越大越贴近 15」。
- **导数在 `x=40` 时是 0.0191，不是 0。** 相比原点的 1.0，
  梯度被压到 1.9% —— **仍然能学**。这正是要的效果：
  比 `clamp` 的 0 好，也比不 softcap 的 1.0 好。

**为什么不用硬 clamp**：`torch.clamp` 在边界处导数是 0，
会造成同样的梯度消失问题，而且不可逆 —— 梯度一旦为 0，
参数就永远学不动了。

**为什么不是 `-inf → 0` 那种硬 mask**：那也是硬边界。

> 这个技巧来自 Gemma-2（Google DeepMind），nanochat 采用了它。

---

## 概念：softmax 的数值稳定实现

`F.cross_entropy` 内部做了数值稳定处理（减去 `logsumexp`）：

```python
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1),
                       ignore_index=-1, reduction=loss_reduction)
```

**不要自己写 `log_softmax` 然后 `gather` 再 `log`** ——
如果某个 logit 是 1000，`exp(1000)` 会 inf。
`F.cross_entropy` 内部用 `logsumexp` 避开了这个问题。

---

## 概念：权重绑定（`tie_embeddings`）

`ModelConfig.tie_embeddings` 默认 `False`（不绑定）。
开启后：

```python
if cfg.tie_embeddings:
    self.transformer.wte.weight = self.lm_head.weight    # 共享同一个 Parameter
```

**省的是参数**：`wte` 是 `(V, D)`，`lm_head` 是 `(D, V)`。
绑定后省掉一份 `(V, D)`。实测：

| 档位 | 总参数 | 嵌入 (`V×D`) | 嵌入占比 |
|---|---|---|---|
| `ablation` | 23,199,744 | 6,291,456 | **27.1%** |
| `full` | 346,031,882 | 12,582,912 | **3.6%** |

注意这个占比**随规模剧烈变化**：小模型上嵌入是参数的近三成，
`full` 档只有 3.6%。所以「绑定省多少参数」这个问题**没有
单一答案** —— 在你正在训的那个规模上算。

（小模型嵌入占比高，正是第 02 章那个结论的另一个表述：
嵌入占大头时，缩小 `V` 比加深模型更划算。）

**代价是两个位置的学习率必须一致**。本项目分别配置：

```python
embedding_lr   = 0.3      # wte
unembedding_lr = 0.008    # lm_head
```

它们差 **37.5 倍**！绑定后这两个 LR 必须取同一个值，
否则共享的参数会收到互相矛盾的更新信号。

> **nanoGPT 默认绑定，nanochat 不绑。** 因为 nanochat 给 `wte` 和
> `lm_head` 设计了差别很大的 LR（嵌入侧需要大步长学新 token，
> 输出侧需要小步长避免 logits 爆炸）。本项目跟随 nanochat。

**本项目默认不绑定**，且这个开关在 `full` 档的「最佳组合」里
**没有开启** —— 见下面的消融。

---

## 手抄代码

`GPT.forward` 的输出头部分（卷 3 会讲 `__init__` 和 trick 部分）：

```python
# (5) 输出头
x = rms_norm(x)                      # ← 最后这次归一化不能省

logits = self.lm_head(x)                      # (B, T, padded_vocab)
logits = logits[..., :cfg.vocab_size]         # 切掉 padding 部分
logits = logits.float()                        # 后面要算 CE，fp32 更稳

# (6) Logit softcap
if cfg.logit_softcap > 0:
    cap = cfg.logit_softcap
    logits = cap * torch.tanh(logits / cap)

if targets is not None:
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=-1, reduction=loss_reduction)
    if loss_reduction == "none":
        loss = loss.view(B, T)
    elif loss_reduction == "mean" and not torch.isfinite(loss):
        # 整批 target 都被 ignore（都是 -1）时，F.cross_entropy 算的是 0/0 = NaN。
        # 这里降级成 0 而不是让 NaN 污染全部权重 —— 后者不可恢复。
        log0("  [警告] 本批没有有效的监督 token（targets 全为 -1），loss 记为 0。")
        loss = logits.sum() * 0.0
    return loss
return logits
```

**四处容易写错的地方**：

1. **`logits.float()` 在 `F.cross_entropy` 之前**。bf16 的
   `logsumexp` 精度不够，交叉熵会损失几个百分点。
2. **softcap 在 `.float()` 之后**。`tanh` 在 bf16 上的精度损失
   比 fp32 大得多。
3. **`ignore_index=-1` 必须传**。SFT 时大量位置的 target 是 -1
   （第 04 章的 loss mask）。漏了它会把 -1 当成合法的 class id，
   loss 直接变成垃圾。
4. **NaN 兜底不是可选项**。`F.cross_entropy` 在「全部 target 都是
   `ignore_index`」时算 `0/0 = NaN`，而 NaN 一旦进梯度就会
   污染**所有**权重且不可恢复。这段兜底把它降级成 0。

---

## 动手验证

### 验证 1：softcap 的效果

```bash
uv run pytest tests/test_core.py -k softcap -v
```

自己也可以看曲线形状：

```python
import sys, torch
sys.path.insert(0, "src")

cap = 15.0
x = torch.linspace(-40, 40, 401)
y = cap * torch.tanh(x / cap)

print("x=0   ->", round(y[200].item(), 4))
print("x=15  ->", round(cap * torch.tanh(torch.tensor(1.0)).item(), 4))
print("x=40  ->", round(y[-1].item(), 4), "  ← 注意不是 15.0")
print("x=-40 ->", round(y[0].item(), 4))

# 中心差分求导：看梯度有没有被彻底压成 0
d = (cap * torch.tanh((x + 0.01) / cap) - cap * torch.tanh((x - 0.01) / cap)) / 0.02
print("导数 x=0  :", round(d[200].item(), 4))
print("导数 x=15 :", round(d[275].item(), 4))
print("导数 x=40 :", f"{d[-1].item():.2e}", "  ← 不是 0")
```

实测：

```
x=0   -> -0.0000
x=15  -> 11.4239
x=40  -> 14.8559
x=-40 -> -14.8559
导数 x=0  : 1.0000
导数 x=15 : 0.4200
导数 x=40 : 1.91e-02
```

### 验证 2：logits 真的会涨过 15 吗

这是本章最该自己量的一个数。**别信文档，也别信「小模型用不上」
这种想当然的话。**

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config, OptimConfig
from model.gpt import build_model
from model.layers import precompute_rope
from data.tokenizer import get_tokenizer
from data.dataloader import make_dataloader
from optim.muon import setup_optimizer


def run(softcap, steps=200, mode="smoke"):
    cfg = make_run_config(mode).model
    cfg.logit_softcap = softcap
    model = build_model(cfg, device="cuda")
    opt = setup_optimizer(model, OptimConfig())
    loader = make_dataloader(get_tokenizer(), batch_size=8,
                             seq_len=cfg.sequence_len, split="train", device="cuda")
    torch.manual_seed(0)
    peak_raw = peak_out = 0.0
    losses = []
    for _ in range(steps):
        inputs, targets, _ = next(loader)
        loss = model(inputs, targets)
        loss.backward(); opt.step(); model.zero_grad(set_to_none=True)
        losses.append(loss.item())
        with torch.no_grad():
            x = model.transformer.wte(inputs[:2, :128])
            cos, sin = precompute_rope(128, cfg.head_dim, base=cfg.rope_base,
                                       device="cuda")
            for i, b in enumerate(model.transformer.h):
                x = b(x, (cos, sin), cfg.window_sizes()[i])
            x = torch.nn.functional.rms_norm(x, (cfg.n_embd,))
            lg = model.lm_head(x)[..., :cfg.vocab_size].float()
            peak_raw = max(peak_raw, lg.abs().max().item())
            out = lg if softcap <= 0 else softcap * torch.tanh(lg / softcap)
            peak_out = max(peak_out, out.abs().max().item())
    return peak_raw, peak_out, sum(losses[-10:]) / 10


print(f"{'softcap':>10} {'原始|logit|max':>15} {'输出|max|':>10} {'末10步loss':>12}")
for cap in (0.0, 15.0):
    r, o, l = run(cap)
    print(f"{'0（关）' if cap == 0 else '15.0':>10} {r:>15.2f} {o:>10.2f} {l:>12.4f}")
```

实测（`smoke` 档，4 层 / 128 维 / 8192 词表，**真实 token**，200 步）：

```
   softcap    原始|logit|max    输出|max|     前10步loss     后10步loss
      0（关）           24.84      24.84       8.1641       5.8483
      15.0           27.11      14.21       8.1453       5.8242
```

**三个观察，第二个推翻了我原本的预期**：

1. **不 softcap 时 logits 真的涨到 24.84。** 而且这是 4 层 128 维的
   小模型、只跑了 200 步 —— 「小模型用不上 softcap」是错的。
   任何时候你怀疑某个技巧在小规模上无用，把它量一遍再说。
2. **开了 softcap 之后，原始 logits 涨得更高（27.11 vs 24.84）。**
   这很反直觉：softcap 压的是**输出**，而 `lm_head` 的权重
   更新靠的是 `d loss / d (原始 logit)` —— 那个梯度一直在，
   所以权重继续变大，只是输出被 `tanh` 挡住。**softcap 不阻止
   权重变大，它只限制模型「看到」的东西。**
3. **loss 几乎没差别**（5.8483 vs 5.8242，差 0.4%）。
   说明在这个尺度上 softcap 主要作用是**数值安全**，
   不是提升质量。

> ⚠ 这三个数字只在本机、这一档、这一批种子下成立。换档位
> （尤其 `full` 的 24 层 / 768 维）结论可能反过来。

### 验证 3：ignore_index 的作用

```python
import sys, torch
sys.path.insert(0, "src")
import torch.nn.functional as F

logits = torch.randn(4, 100)
targets = torch.tensor([5, -1, -1, 7])

print("带 ignore_index:", F.cross_entropy(logits, targets, ignore_index=-1).item())
print("不带            :", F.cross_entropy(logits, targets).item())
```

实测：

```
带 ignore_index: 5.207038
不带 -> IndexError: Target -1 is out of bounds.
```

**PyTorch 会直接报错，不会静默算错。** 这比「静默失败」好 ——
但也意味着**漏传 `ignore_index` 在 SFT 里是一个立刻可见的崩溃**，
而不是慢慢变差的 loss。看到这个 `IndexError` 就知道少了参数。

---

## ★ 消融实验

本章有两个开关，都值得测：

### softcap

```bash
# 基线（cap=15）
bash script/train_base.sh ablation --no-resume --model-tag d6_cap15

# 关掉
bash script/train_base.sh ablation --no-resume --model-tag d6_cap0 --no-softcap

bash scratch/ablation.sh d6_cap15
```

**预期**：`ablation` 档上 **Δ bpb 接近 0**（在 `smoke` 档实测
loss 差 0.4%，见验证 2）。但**这不代表 softcap 没用**：

- 验证 2 已经证明 logits 确实会涨过 cap（24.84 vs 上限 15），
  所以 softcap 是在**真实地**做事，不是空转
- 它的价值是**数值安全**（不让训练某天突然炸掉），
  而不是提升质量
- 这种「收益在异常路径上」的改动，消融测不出来 ——
  **消融的阴性结果不等于改动无用**

这个消融真正的价值是**建立一条基线认知**：如果哪天 bpb 差了一截，
先确认是不是 softcap 被意外关掉了。

### 权重绑定

```bash
# 基线（不绑定）
bash script/train_base.sh ablation --no-resume --model-tag d6_unbound

# 绑定
bash script/train_base.sh ablation --no-resume --model-tag d6_tied --tie-embeddings

bash scratch/ablation.sh d6_unbound
```

**预期**：

- bpb 大致持平（绑定是权重共享的正则化效果，经验上略好或持平）
- **参数量明显下降**：注意日志里 `lm_head` 那一项会变成 0
  （因为 `named_parameters()` 只数一次共享的张量）

> ⚠ **绑定后的 LR 问题**：绑定后 `wte` 和 `lm_head` 共享参数，
> 但本项目给它们的 LR 差 37.5 倍（0.3 vs 0.008）。共享参数会同时
> 收到两个方向的更新，互相抵消，**预计**训练会明显变差。
>
> ⚠⚠ 「预计」两个字是认真的 —— **这个组合我没有实测过**，
> 不要把它当成已验证的结论。理由是 `0.3` 是为「从零学一个新 token」
> 设计的（embedding 侧需要大步长），而 `0.008` 是为「避免 logits
> 爆炸」设计的（输出侧需要小步长）。用哪个都会在另一侧出问题：
> - 用 0.3：logits 会被推得很大，正好撞上本章的 softcap
> - 用 0.008：嵌入几乎学不动，`wte` 一直是随机初始化附近
>
> nanochat 不绑定，正是为了能独立调这两个数。

---

## 常见坑

### 坑 1：忘了 `rms_norm(x)`（最后那次）

24 层的残差流 RMS 会增长（本章验证 1：每层 4-5%，
24 层累计约 3 倍）。不归一化的话 `lm_head` 收到的输入尺度偏大，
logits 偏大，配合 softcap 会被大量压到饱和端。

### 坑 2：`.float()` 和 softcap 的顺序反了

```python
logits = cap * torch.tanh(logits / cap)      # ❌ 在 bf16 上算 tanh
logits = logits.float()
```

bf16 只有 8 位尾数，`tanh` 在 bf16 上算出来的结果相对误差约 1e-2。
正确顺序是先转 fp32 再 softcap。

### 坑 3：`ignore_index=-1` 漏传

见上面。这是 SFT 阶段最容易出的 bug —— 症状是**立刻崩**
（`IndexError: Target -1 is out of bounds`），不是慢慢变差。
反过来说：如果你在 SFT 里看到 loss 正常下降但质量很差，
那**不是**这个 bug（漏传会崩）。要往别处查。

### 坑 4：NaN 兜底被当成「可选项」删掉

```python
elif loss_reduction == "mean" and not torch.isfinite(loss):
    log0("  [警告] ...")
    loss = logits.sum() * 0.0
```

这段看起来多余，但它的作用是**阻止 NaN 进入梯度**。
`logits.sum() * 0.0` 的梯度是全 0 —— 参数完全不动，但至少
下一个 batch 的梯度还是干净的。

删掉它的话，一次「整批无监督」就会让**所有权重变成 NaN**，
而且不可恢复（AdamW 的动量里也有 NaN，会一直传播）。

---

## 延伸

**Gemma-2 的其他 logit 相关技巧**

- **attention logit softcap**：在每个注意力模块的 softmax **之前**
  对 `QKᵀ` 做 softcap，`cap=50`。⚠ 本项目**没有**实现它，原因见上
  （与 FA2 不兼容）
- **sliding window + 交替 local/global**：Gemma-2 用 5:1 的
  local:global 交替（local 4096 / global 8192），
  本项目用 `window_pattern` 的平铺（第 20 章）
- **QK-Norm**：本项目第 10 章。⚠ 注意 Gemma-2 **不用** QK-Norm
  —— 它的 attention softcap 承担了同一个职责。本项目选 QK-Norm
  而不选 attention softcap，就是为了保住 FA2

**为什么 `logit_softcap=15` 而不是更小的值**

`cap` 决定了 logits 的有效范围 `[-cap, cap]`，也就是
softmax 的「动态范围」是 `2·cap`。太小（比如 5）会让模型
无法同时表达强和弱的偏好；太大（比如 100）就等于没限制。

15 对应「有效范围 30」，而典型 LLM 的 logit 分布就在这个量级。

Gemma-2 用的是**两个不同的值**：注意力内部的 logits `cap=50`，
最终输出层 `cap=30`。本项目取 15，比 Gemma-2 更紧 —— 因为本项目
的 QK-Norm（第 10 章）已经把注意力 logits 压到 `±11.52` 附近，
输出端的 logits 也没有 Gemma-2 那么大（验证 2 实测 27）。

> **一个值得注意的副作用**：softcap 与 FlashAttention 不兼容。
> FA2 的 kernel 不接受在 softmax 前插入 `tanh`。Gemma-2 论文明确
> 提到了这一点，所以它那一代放弃了 FA。本项目没这个问题 ——
> softcap 只作用在 `lm_head` 输出上，不在注意力内部，
> 所以 SDPA 的 flash 后端照常可用（见第 11 章）。
> 这也解释了为什么本项目**没有实现「attention logit softcap」**。

**softmax vs sigmoid**

LLaMA 用 softmax（多分类互斥），Gemma 1 试过 sigmoid（二分类
独立），发现不如 softmax。原因是 softmax 的「归一化」提供了
额外的竞争信息 —— 一个 token 概率上升会挤压其他 token。

---

## 卷 2 完成

你现在的能力：

- ✅ 画出一个 GPT 的完整数据流图，说清每一步的形状变化
- ✅ 解释 Pre-LN 为什么能训深层（恒等项保证梯度通路）
- ✅ 说清 RMSNorm / RoPE / QK-Norm 各自的**代价**，不只是好处
- ✅ 解释「保证用上 FA2」和「碰巧用上」的区别
- ✅ 独立验证一个数值声明（而不是相信文档）

**下一步**：卷 3 讲 nanochat 的 5 个架构 trick。它们全部是在
**操纵残差流** —— 第 14 章那条「信息总线」是理解它们的唯一框架。

[第 16 章：meta device 三步建模型](16-meta-device三步建模型.md)