# 12 · QK-Norm 与 GQA

两个「用一点东西换另一点」的机制：
QK-Norm 用**表达力**换**训练稳定性**，GQA 用**质量**换**显存/带宽**。

---

## 本章目标

- 算清 QK-Norm 的代价：它到底丢掉了什么信息
- 说清 GQA / MQA 的机制与代价，能自己算 KV cache 省了多少
- 亲手实现 `CausalSelfAttention.__init__` 里的 GQA 分支

---

## 前置回顾

- [第 10 章](10-注意力三步曲.md)：注意力五步走完，QK-Norm 讲了「防什么」
- [第 11 章](11-SDPA与FlashAttention.md)：`sdpa_bt` 和 FA2

本章把 QK-Norm 的**代价**补齐，再加一个新东西：GQA。

---

## 概念：QK-Norm 丢掉了什么

第 10 章说 QK-Norm 防 logits 爆炸。代价是：

```python
q = rms_norm(q) * self.qk_norm_scale    # 所有 q 的长度都变成 1.2·√Dh
k = rms_norm(k) * self.qk_norm_scale
```

**归一化之后，所有 query 和所有 key 的范数都相同。**

而注意力分数是 `q · k`。内积的大小完全由**方向**决定 ——
「这个 key 值不值得看」这种**强度**信息被抹掉了。

具体地说，一个 head 再也无法表达：

- 「key A 的重要程度是 key B 的 3 倍」
- 「只有 query 的某一类才应该关注某类 key」中的强度差异

**它只能表达方向。** 这是 RMSNorm + 内积这个组合的固有性质，
不是实现缺陷。

### 那为什么还要用

因为**稳定性比那一点表达力更值钱**。LLM 训练中 logits 饱和是
一个真实且致命的失败模式 —— 一旦某个 head 饱和，它基本不可恢复
（梯度趋近 0，参数不再更新）。

> QK-Norm 来自 PaLM（Dehghani et al. 2023，
> *Scaling Vision Transformers to 22 Billion Parameters*），
> 机制上等价于「把 q、k 的范数钉住，从而给 logits 一个上界」。
> 有资料说它在**超过 12 层**之后才成为必需 —— 这个门槛数字
> 我没有找到一手出处，**别当结论用**。

代价（表达力）小于收益（稳定性）。

> `qk_norm_scale = 1.2` 这个系数也是超参。太小则归一化过度
> （logits 完全没有区分度），太大则压不住。本项目默认 1.2，
> nanochat 也是 1.2。

---

## 概念：GQA —— 为什么 KV 不需要那么多头

**MHA（Multi-Head Attention）** 的假设是「每个头学不同的东西」。
`n_head = 12` 意味着 12 组独立的 q、k、v。

但仔细看 KV cache 的开销：每个 token 都要缓存
`n_layer × 2 × n_head × head_dim` 个数（k 和 v 各一份）。

实测 `full` 档：

```
bytes/token = n_layer × 2 × n_kv_head × head_dim × 2(bf16)
            = 24 × 2 × 12 × 64 × 2
            = 73,728 bytes ≈ 72 KiB / token
```

T=1024 的上下文，**每行 batch 就是 72 MiB**。

而且 decode 阶段是**纯 memory bound**（第 11 章）：
每生成一个 token 要读整个模型的权重 + 整个 KV cache。
KV cache 越大，生成越慢。

### 三种方案

| 方案 | `n_kv_head` | KV cache | 质量 |
|---|---|---|---|
| **MHA** | = `n_head` | 100% | 最好 |
| **MQA**（Multi-Query） | 1 | 8.3% | 明显下降 |
| **GQA**（Grouped-Query） | 分组，如 `n_head/4` | 33.3% | 几乎不降 |

实测 KV cache（`full` 档，T=1024，每行 batch）：

| 配置 | bytes/token | KV cache/T | 相对 MHA |
|---|---|---|---|
| MHA（`n_kv_head=12`） | 73,728 | 72 MiB | 100% |
| **GQA（`n_kv_head=4`）** | **24,576** | **24 MiB** | **33.3%** |
| MQA（`n_kv_head=1`） | 6,144 | 6 MiB | 8.3% |

GQA 的思路是**折中**：把 12 个 query 头分成 3 组，每组 4 个头
**共享一组 kv**。

```
q:  [h0 h1 h2 h3 | h4 h5 h6 h7 | h8 h9 h10 h11]     12 个头
     ───────────   ───────────   ────────────
k:  [   k0 k1     |   k2 k3     |   k4 k5   ]        3 组
```

同组内的 4 个 query 头看到相同的 kv，但 q 不同 → 仍然学出不同的
注意力模式。**损失很小，但 kv 少了 3 倍。**

MQA 是 GQA 的极端情况（每组 1 个头）。GQA 的原始论文
（*GQA: Training Generalized Multi-Query Transformer Models
from Multi-Head Checkpoints*, Ainslie et al. 2023）还展示了
**从 MHA 权重 up-training 到 GQA** 的方法 —— 先训 MHA 再分组，
质量损失可以做到几乎为零。

---

## 概念：本项目为什么默认关掉 GQA

```python
n_kv_head: int = 6      # kv 头数，< n_head 即为 GQA
```

`build_model_config` 里默认：

```python
n_kv_head=n_embd // head_dim,   # 默认不做 GQA
```

**默认 `n_kv_head == n_head`（MHA）**，原因有三：

1. **训练时 GQA 省不到显存。** KV cache 只在推理（decode）时用。
   训练时 k/v 是当场算出来就用掉，不缓存。省 KV cache 对训练速度
   只有间接影响（更少的通信量，多卡才有意义）。
2. **`enable_gqa=True` 在 SDPA 内部会把 kv head 展开成 q head 的数量** ——
   **临时反而多占一份显存**。GQA 省的是 cache，不是计算。
3. **本项目是单卡、训练为主**。GQA 的收益场景是多卡大模型推理，
   两个条件都不满足。

**但接口留着**：`n_kv_head` 可配，`assert n_head % n_kv_head == 0`
保证分组合法，第 11 章提到的 `enable_gqa=True` 会自动生效。

> 想在 `full` 档上开 GQA：`--depth 24` 保持，然后改 config 里
> `n_kv_head=4`。**注意这会改变参数量**（`c_k`/`c_v` 从 768→768
> 变成 768→256），不是纯推理优化。

---

## 手抄代码

本章的 `__init__` 已经在第 10 章写过了。这里只强调 GQA 相关的部分。

### 第 1 块：kv 维度的推导

```python
def __init__(self, cfg, layer_idx: int, device=None):
    super().__init__()
    self.n_head = cfg.n_head
    self.n_kv_head = cfg.n_kv_head
    self.head_dim = cfg.head_dim
    self.d_model = cfg.n_embd

    # ★ GQA 的合法性检查：每个 kv 头要被 n_head/n_kv_head 个 q 头共享，
    #   所以必须整除。不整除的话 SDPA 的 enable_gqa 无法展开。
    assert cfg.n_head % cfg.n_kv_head == 0, "n_head 必须是 n_kv_head 的整数倍"

    # ★ c_k / c_v 的输出维度用 kv_dim 而不是 n_head * head_dim ——
    #   这就是 GQA 省参数的机制。no GQA 时 kv_dim == d_model，
    #   两者相等，所以这段代码在默认配置下「看不出差别」。
    kv_dim = cfg.n_kv_head * cfg.head_dim

    self.c_q = Linear(self.d_model, self.n_head   * self.head_dim, bias=cfg.use_bias, device=device)
    self.c_k = Linear(self.d_model, kv_dim,                          bias=cfg.use_bias, device=device)
    self.c_v = Linear(self.d_model, kv_dim,                          bias=cfg.use_bias, device=device)
    self.c_proj = Linear(self.d_model, self.d_model,                   bias=cfg.use_bias, device=device)
```

**最容易写错的地方**：`c_k` 和 `c_v` 的输出维度。

```python
self.c_k = Linear(self.d_model, self.n_head * self.head_dim)    # ❌ 忘了改成 kv_dim
```

在默认配置（无 GQA）下这**测不出来** —— `kv_dim == d_model`，
两者数值相同。开了 GQA 之后，`view(B,T,n_head,head_dim)` 会因为
元素总数不匹配而报错。

### 第 2 块：forward 里的 GQA 分支

```python
# (1) 投影 —— k/v 用 n_kv_head
q = self.c_q(x).view(B, T, self.n_head,   self.head_dim)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
```

`attend` 内部通过 `gqa=(q.size(2) != k.size(2))` **自动**启用
`enable_gqa`：

```python
def attend(q, k, v, window, attn_impl="flex"):
    left = window[0]
    T = q.size(1)
    if left is None or left <= 0 or left >= T:
        return sdpa_bt(q, k, v, causal=True, gqa=(q.size(2) != k.size(2)))
    ...
```

**注意 `q.size(2) != k.size(2)` 这个判断**：它比较的是**头数**
（`view` 之后的第 2 维），不是总元素数。无 GQA 时两者相等 →
`gqa=False`；有 GQA 时不等 → `gqa=True`。

---

## 动手验证

### 验证 1：GQA 真的能跑，且形状正确

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.layers import CausalSelfAttention, precompute_rope

cfg = make_run_config("full").model          # n_head=12, n_embd=768, head_dim=64
cos, sin = precompute_rope(16, cfg.head_dim, base=cfg.rope_base)

for n_kv in (12, 4, 1):
    cfg.n_kv_head = n_kv
    attn = CausalSelfAttention(cfg, layer_idx=0)
    x = torch.randn(2, 16, cfg.n_embd)
    y = attn(x, (cos, sin), window=(16, 0))
    n_q = sum(p.numel() for n, p in attn.named_parameters() if n.startswith("c_q"))
    n_k = sum(p.numel() for n, p in attn.named_parameters() if n.startswith("c_k"))
    print(f"n_kv_head={n_kv:2d}: c_q={n_q:>7,}  c_k={n_k:>7,}  "
          f"比值={n_q / n_k:4.1f}  out={tuple(y.shape)}")
```

实测：

```
  n_kv_head=12: c_q= 589824 c_k= 589824 比值= 1.0 out=(2, 16, 768)
  n_kv_head= 4: c_q= 589824 c_k= 196608 比值= 3.0 out=(2, 16, 768)
  n_kv_head= 1: c_q= 589824 c_k=  49152 比值=12.0 out=(2, 16, 768)
```

**两个观察**：

1. **输出形状恒为 `(2, 16, 768)`** —— 与 `n_kv_head` 无关。因为
   `c_proj` 的输入维度固定是 `n_head × head_dim = 768`。
2. **参数比正好是 `n_head / n_kv_head`**（12/12=1、12/4=3、12/1=12）。
   这就是 GQA 省参数的全部机制：`c_k`/`c_v` 变小了。

> ⚠ 手写这段验证时容易踩三个坑：
> ① `cos/sin` 必须用 `precompute_rope(16, cfg.head_dim)` 生成，且
>    `seq_len` 参数（第一个）要 ≥ 输入的 `T`，否则报
>    `size of tensor a (32) must match tensor b (64) at dimension 3`。
> ② `cfg.n_kv_head` 必须能被 `n_head` 整除。`full` 档 `n_head=12`，
>    所以 `(12,4,1)` 合法、`(6,2,1)` 也合法；但如果 `n_head=2`，
>    `4` 和 `2` 都非法（`build_model_config(4, 32, 64, ...)` 的 `n_head`
>    是 2 不是 4，写 `(4,2,1)` 会在第一个就撞 assert）。
> ③ `cfg` 是 dataclass，循环里改 `cfg.n_kv_head` 会**一直生效**
>    （同一个对象）。这正好方便，但如果之后又 `CausalSelfAttention(cfg)`
>    一次，拿到的就是最后那个 `n_kv_head`。

### 验证 2：KV cache 的字节数

```python
import sys
sys.path.insert(0, "src")
from common.config import make_run_config

cfg = make_run_config("full").model
print(f"默认 n_kv_head={cfg.n_kv_head}: {cfg.kv_bytes_per_token():,} bytes/token")
for n_kv in (12, 4, 1):
    cfg.n_kv_head = n_kv
    b = cfg.kv_bytes_per_token()
    print(f"n_kv_head={n_kv:2d}: {b:>7,} bytes/token"
          f"   (T=1024 时 {b * 1024 / 1024**2:.0f} MiB/batch行)")
```

实测：

```
默认 n_kv_head=12: 73,728 bytes/token
n_kv_head=12:  73728 bytes/token   (T=1024 时 72 MiB/batch行)
n_kv_head= 4:  24576 bytes/token   (T=1024 时 24 MiB/batch行)
n_kv_head= 1:   6144 bytes/token   (T=1024 时  6 MiB/batch行)
```

比值正好是 `1 : 1/3 : 1/12`，对应 `n_head/n_kv_head`。

### 验证 3：SDPA 的 `enable_gqa` 真的可用

```bash
uv run pytest tests/test_core.py -k flash -v
```

判据里有一个 case 是 `(12, 64, 1024, 4)` —— 12 个 q 头配 4 个 kv 头，
用它验证 FA2 后端在 GQA 下仍然可选。

---

## ★ 消融实验

GQA 是**推理期**的优化，训练时质量不变（因为 `c_k`/`c_v` 的
参数量变了，其实是不同的模型）。所以本章的消融和前面几章不同 ——
它测的是「质量 vs 显存」的权衡曲线。

```bash
# MHA（默认，n_kv=6）
bash script/train_base.sh ablation --no-resume --model-tag d6_mha

# GQA（n_kv=2，需要改 config 里的 n_kv_head）
# —— CLI 没有这个开关，见下面的说明
```

> ⚠ **本项目没有 `--n-kv-head` CLI 开关。** 想测 GQA 得改
> `build_model_config` 里的 `n_kv_head=` 那一行。这是有意的：
> GQA 在本项目的目标硬件（单卡训练）上没有收益，
> 暴露一个用不上的开关只会增加教学噪音。

### 真正值得做的消融：QK-Norm 的强度

```bash
# 默认 1.2
bash script/train_base.sh ablation --no-resume --model-tag d6_qk12

# 关掉
bash script/train_base.sh ablation --no-resume --model-tag d6_qk0 --no-qk-norm

# 调大到 5（看会不会反而变差 —— 归一化过度导致 logits 无区分度）
# 需要改 config 的 qk_norm_scale

bash scratch/ablation.sh d6_qk12
```

**预期**：
- `qk=0` 的 bpb 略差或持平，**但早期 loss 曲线更抖**
- `qk=5` 的 bpb 明显变差（logits 区分度不足）

第三条是这个消融的真正价值：**QK-Norm 不是「越大越好」**，
1.2 是一个平衡点。

> ⚠ 在 `ablation` 档（d6）上，QK-Norm 的效应往往落在噪声内
> （±0.02）。真正能测出差别的是 d12 以上。

---

## 常见坑

### 坑 1：`c_k` 用了 `n_head` 而不是 `n_kv_head`

见上面。默认配置下测不出来，GQA 一开就炸。

### 坑 2：以为 GQA 也能在训练时省显存

**不能。** 训练时 k/v 当场算出来就用掉，不缓存。
GQA 省的是推理时的 KV cache 占用，以及 decode 的带宽。

而且 `enable_gqa=True` 在 SDPA 内部要把 kv head **展开**成 q head
的数量 —— **临时反而多占一份显存**。所以 GQA 在训练时是纯亏。

### 坑 3：以为 `n_head % n_kv_head == 0` 自动成立

`build_model_config` 里 `n_kv_head = n_embd // head_dim`，
而 `n_head = n_embd // head_dim` —— 两者相等，自动成立。
但如果你手改 `n_kv_head=5` 而 `n_head=6`，`assert` 会抓住。

### 坑 4：把 QK-Norm 应用到 v 上

`rms_norm(v)` 是错的 —— v 不参与内积计算（第 09 章解释过）。
归一化 v 只会给输出引入一个没有意义的缩放。

---

## 延伸

**QK-Norm 的替代方案**：Learned Query Scaling（LQS）——
保留 q/k 的原始范数，但除以一个可学的标量 `γ`（每头一个）：

```
q' = q / (1 + |q| · γ)
```

这样保留了「强度」信息（通过 `|q|`），又限制了 logits。
⚠ 我**没有**找到 nanochat 实测 LQS 的一手数据，也没有在本仓库
测过 —— 别把这个替代方案当已验证结论。它的理论动机是成立的
（保留了范数信息），但收益未知。

**GQA 的中间形态**：`n_kv_head` 可以取任意 `n_head` 的因子 ——
`n_kv_head = 2` 是「12 头配 2 组」，`n_kv_head = 3` 是「4 头一组」。
实践中 `n_kv_head = n_head/4`（每组 4 个头）最常用，
因为质量损失在这个比例下基本为零。

**MLA（Multi-head Latent Attention）**：DeepSeek 的方案，更激进 ——
把 k/v 联合低秩压缩成一个 latent 向量再缓存。`full` 档的
KV cache 能压到 5% 以下。代价是需要额外的投影矩阵和更复杂的
attention kernel。本项目没实现。

---

## 下一章

[第 13 章：MLP 与激活函数](13-MLP与激活函数.md) ——
注意力的另一半。它占了整个模型约 2/3 的参数。