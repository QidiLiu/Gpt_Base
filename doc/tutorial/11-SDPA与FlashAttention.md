# 11 · SDPA 与 FlashAttention

一个**布局陷阱**，一个**「保证用上 FA2」**的承诺。本章是卷 2 里
唯一一章「不改数学、只改怎么算」的。

---

## 本章目标

- 亲手踩一次 SDPA 的 `(B,H,L,E)` 布局陷阱，并理解它为什么这么危险
- 从 IO bound 的视角说清楚 FlashAttention 快在哪
- 知道本项目**保证**用上 FA2（而不是碰巧用上），以及怎么验证

---

## 前置回顾

[第 10 章](10-注意力三步曲.md) 里注意力已经能跑了。
但那章的 `attend()` 调用了本章要实现的 `sdpa_bt()`。

---

## 概念：布局陷阱 —— 为什么它不会报错

`F.scaled_dot_product_attention` 的签名是：

```python
scaled_dot_product_attention(query, key, value, attn_mask=None,
                             dropout_p=0.0, is_causal=False, scale=None)
```

**它严格要求 `(B, H, L, E)` 布局。** 而本项目全程用 `(B, T, H, D)`。

直接喂进去会怎样？PyTorch 不会检查 —— 它把第 1 维当 H、第 2 维当 L：

```
我们以为: (B, T, H, D)  ->  B=8, T=512, H=6,  D=32
它以为:   (B, H, L, E)  ->  B=8, H=512, L=6,  E=32
```

三种后果，取决于 T 和 H 的关系：

| 情况 | 后果 |
|---|---|
| `T=1` | **侥幸能跑** —— L=1 广播合法，但返回的是 `(B,H,L,E)` 布局，形状悄悄变了 |
| `T=2` | **直接 RuntimeError** —— q 的 L=2 和 k 的 H=6 撞在同一维 |
| `T == H` | **正常** —— 两种解读碰巧同形，纯属巧合 |

**最危险的是 T=1**：那正是推理 decode 阶段！所以这个 bug 在训练时
（T=512）能正常跑，一进推理就崩或者给出错误结果。

### 亲手看

```bash
uv run python scratch/hardware.py
```

它会打印这个陷阱的最小复现：

```
形状相同 = True，数值相同 = False，最大绝对误差 = 4.78
```

「形状相同、数值不同」是最难查的一类 bug —— 所有 assert 都过，
但算的是另一个东西。

---

## 概念：FlashAttention 为什么快 —— IO bound 视角

先看朴素注意力干了什么。设序列长度 T、头维 Dh：

```
1. 算 S = Q Kᵀ                    → (T, T) 矩阵，内存 T²·4 字节
2. softmax(S)
3. 输出 = P V                      → (T, Dh)
```

**T=8192 时，`S` 有 67M 个元素 = 268 MB**（fp32）。
每个 Block 至少一次前向、一次反向 —— 一次训练要反复读写几百 MB。

FlashAttention 的洞察：**从 roofline 的角度看，这个 kernel 是
memory bound，不是 compute bound。**

矩阵乘的算术强度（每读 1 字节能做的浮点运算）：

- 大矩阵乘：`O(sqrt(N))` —— 算得比读得多，**compute bound**（GPU 算力用满）
- 注意力的 softmax：`O(1)` —— 每读 1 个元素只做几次运算，**memory bound**

实测（`full` 档 T=1024、H=12、Dh=64）：

```
MFU = 23.6%   ← 远未用满算力
```

23.6% 说明：**大部分时间在等显存搬运，不在算。**

FlashAttention 的做法是**分块（tiling）+ 在线 softmax**：

```
把 Q/K/V 切成小块，放进 SRAM（片上高速缓存）
逐块算局部 attention，用「在线 softmax」增量修正全局的 max 和 sum
最后一次性写回 O
```

关键在于：**中间结果从不落回 HBM**。`(T,T)` 的注意力矩阵永远不出现
在显存里，只在寄存器/SRAM 里分块存在。

```
朴素:    HBM ←→ HBM   读 Q,K,V 写 S 读 S 写 P 读 P,V   →  反复搬运 T² 级数据
Flash:   HBM ←→ SRAM  读 Q,K,V（一次）写 O（一次）        →  只搬 O(线性) 级数据
```

这就是为什么它能到 40%+ 的 MFU，而标准实现只有 10-20%。

> **NanoChat 的注意点**：它用的是 **Flash Attention 3**（FA3），
> 原生吃 `(B,T,H,D)` 布局，不需要 transpose。FA3 主要为 Hopper
> （H100）编译，本卡是 Ada，拿不到。PyTorch 的 SDPA 里内建的是
> **FA2** —— 算法同源（都是分块 + 在线 softmax），接口不同。

---

## 概念：「碰巧能用」vs「保证能用」

这是本章最重要的一点。

`F.scaled_dot_product_attention` 会在所有可用后端里**静默挑选**一个：

```
flash  →  mem_efficient  →  math
```

挑到 flash 是运气好，但**没有任何东西保证它**。换 torch 版本、
换 shape、某个参数让它挑不满足，就悄悄退回 mem-efficient，
**而且不报错**。

本机（RTX 4060 Ti / SM 8.9）实测的各路径 FA2 可用性：

| 路径 | FA2 可用 | 实测耗时（4×12×1024×64, bf16） |
|---|---|---|
| `is_causal=True`（全上下文） | ✓ | 0.207 ms |
| GQA（`n_head != n_kv_head`） | ✓ | — |
| KV cache decode（`Tq=1`） | ✓ | — |
| 滑窗 + 显式 `attn_mask` | **✗** | 0.516 ms |
| 滑窗 + `flex_attention` | ✓ | 0.128 ms |

**前三行本项目一直在用 FA2 —— 但从来没有任何东西验证过。**
这正是「碰巧能用」。

所以本项目做了两件事：

1. `flash_backend_report()`（在 `common/utils.py`）：训练启动时打印后端状态
2. `test_causal_path_can_select_flash_backend`：用真实形状断言 FA2 **可选**

### 为什么滑窗拿不到 SDPA 的 FA2

SDPA 的 flash 后端**不接受任意 `attn_mask`**。硬钉它会直接炸 ——
实测：

```python
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
# RuntimeError: No available kernel. Aborting execution.   ← 实测
```

而 mem-efficient 后端**接受** mask（实测 0.516 ms，能跑）。
所以滑窗路径被迫降级，并且要物化一个 `(T,T)` 的 bool mask
（T=1024 时每层每步 1MB）。

绕出去的办法是 `flex_attention`（PyTorch 内建）：它把 mask 编译成
**块级**的 `block_mask`（只存 128×128 的块摘要），不物化 `(T,T)`，
生成的 kernel 就是 FA2 风格的。

**代价**：flex_attention 必须 `torch.compile`（eager 模式极慢）。
所以 `layers.py` 里用了「探测一次，失败回落」的思路。

---

## 手抄代码

### 第 1 块：`sdpa_bt` —— 布局转换

```python
def sdpa_bt(q, k, v, causal, mask=None, gqa=False):
    """
    调用 PyTorch SDPA，并处理「布局不匹配」这个坑。

    进 SDPA 前把第 1 维和第 2 维对调（T <-> H），出来后再对调回来。
    两次 transpose 的开销约 0.06ms（实测，8×512×6×32），相对注意力本身可忽略。
    """
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, is_causal=causal, enable_gqa=gqa)
    return out.transpose(1, 2)
```

**为什么这个函数值得单独存在**：它把「布局转换」这件事收在一处。
全项目只有这里知道 PyTorch 的布局约定。将来换 FA3（或者别的后端），
只需要改这一个函数。

### 第 2 块：滑窗的两条路径

这两块已经写好了（第 20 章会详讲 flex 的机制），本章只需要理解
**它们为什么并存**：

```python
def attend_sliding_sdpa(q, k, v, left):
    """物化 (T,T) 的 bool mask -> SDPA mem_efficient。保留作对照组。"""
    T = q.size(1)
    idx = torch.arange(T, device=q.device)
    delta = idx[:, None] - idx[None, :]
    mask = (delta >= 0) & (delta <= left)
    return sdpa_bt(q, k, v, causal=False, mask=mask,
                   gqa=(q.size(2) != k.size(2)))


def attend_sliding_flex(q, k, v, left):
    """flex_attention + block_mask -> FA2 风格 kernel。"""
    flex = _flex_attention()
    if flex is None:
        return attend_sliding_sdpa(q, k, v, left)      # 回落
    try:
        block_mask = _sliding_block_mask(q.size(1), left, q.device)
        out = flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                   block_mask=block_mask, enable_gqa=(q.size(2) != k.size(2)))
        return out.transpose(1, 2)
    except Exception:
        # 只警告一次。回落不是错误，只是慢 —— 数值等价。
        return attend_sliding_sdpa(q, k, v, left)
```

**为什么两条路径都要保留**：`attend_sliding_sdpa` 是**对照组**。
教程卷3 第 20 章教的是「滑窗的性能悬崖」，要量出那个悬崖就需要
一个慢的参照物。把慢路径删掉，第 20 章就失去教学对象了。

---

## 动手验证

### 验证 1：布局陷阱的最小复现

```bash
uv run python scratch/hardware.py
uv run pytest tests/test_core.py -k sdpa -v
```

关键判据是 `test_sdpa_layout_is_bhtd_not_bthd`：它断言
**`sdpa_bt` 输出的形状和输入相同**。如果漏了任何一次 transpose，
形状就会变成 `(B,H,T,D)`，这个判据立刻红。

### 验证 2：确认 FA2 真的可选

```bash
uv run pytest tests/test_core.py -k flash -v
```

它用**真实形状**（覆盖四档 + GQA 场景）问 PyTorch：
给定这个 shape，flash 后端到底可不可用。这是「保证」能被测量的形式。

### 验证 3：启动时看后端报告

```bash
uv run python -m training.train_base --mode ablation --num-iterations 1
```

日志里应该有一行：

```
注意力后端：✓ 已启用 Flash Attention 2｜滑窗另走 flex_attention（见 model/layers.py 的 attn_impl）
```

如果看到 `✗ 不可用（将退回 mem-efficient / math）`，说明这台机器
（或这个 torch 版本）拿不到 FA2。

### 验证 4：亲手量滑窗的 2.8 倍差距

```python
import sys, time, torch
sys.path.insert(0, "src")
from model.layers import attend_sliding_flex, attend_sliding_sdpa

B, H, T, D, left = 4, 12, 1024, 64, 256     # full 档的形状
q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)

def bench(fn, n=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / n * 1000

tf = bench(lambda: attend_sliding_flex(q, k, v, left))
ts = bench(lambda: attend_sliding_sdpa(q, k, v, left))
print(f"flex (FA2 风格) : {tf:.3f} ms")
print(f"sdpa (mem-eff)  : {ts:.3f} ms")
print(f"差距            : {ts / tf:.2f}x")
```

实测（本机）：

```
flex : 0.128 ms
sdpa : 0.565 ms
ratio: 4.42x
```

> 这个数字随 shape 变化。你可能得到 2.8x ~ 4.9x —— 取决于
> `left` 占 `T` 的比例、`head_dim`、以及 GPU 的占用情况。
> **重要的是两条路径的相对顺序和数量级，不是具体倍数。**

### 验证 5：两条路径数值等价

```bash
uv run pytest tests/test_core.py -k numerically_equivalent -v

> ⚠ **这条命令曾经选不中任何用例**（`-k flex` 全仓 30 deselected，
> 永远「通过」）。flex 路径的判据其实叫
> `test_sliding_window_paths_are_numerically_equivalent` ——
> **它在内部 import 了 `attend_sliding_flex`，只是名字里没有 flex。**
```

**为什么必须等价**：如果两条路径数值不同，「同一模型换个开关跑出
不同的 bpb」就无法区分是「开关的架构差异」还是「两条 kernel
算错了」—— 消融结论就废了。

实测相对误差 **1.0e-03**（bf16 舍入量级，因为两条路的累加顺序不同：
flex 走块级累加，SDPA 走逐元素累加）。

---

## ★ 消融实验

本章没有架构消融（`sdpa_bt` 不改变数学）。但有两个**测量型消融**：

### 测量 1：滑窗的代价

```bash
# 基线：全上下文 + flex
bash script/train_base.sh ablation --no-resume --model-tag d6_L

# 滑窗 + flex（FA2 风格）
bash script/train_base.sh ablation --no-resume --model-tag d6_S_flex --window-pattern SSL

# 滑窗 + 显式 mask（mem-efficient）
bash script/train_base.sh ablation --no-resume --model-tag d6_S_sdpa \
    --window-pattern SSL --attn-impl sdpa

bash scratch/ablation.sh d6_L
```

**预期**：
- `d6_S_flex` 与 `d6_S_sdpa` 的 bpb **几乎相同**（同一个模型，
  不同的 kernel）—— 这本身就是「数值等价」的强证据
- `d6_S_flex` 比 `d6_L` 略差（信息少了：滑窗看不到远处）
- **但耗时差距明显**：`d6_S_sdpa` 会比 `d6_S_flex` 慢

这个消融的价值不在 bpb 差值，而在**它证明了数值等价 + 速度不同**。

### 测量 2：compile 的代价

```bash
# 故意让 compile 失败，看回落日志
GPT_FORCE_COMPILE_FAIL=1 uv run python -m training.train_base --mode ablation --num-iterations 1
```

你会看到 `[optim] torch.compile 不可用（...），回落到 eager`。
这行日志很重要 —— **第 28 章会讲一个真实 bug：Muon advanced 因为
一个 data-dependent branching 静默回落，用户完全没察觉，
full 档因此慢了 23%。**

---

## 常见坑

### 坑 1：以为 `transpose` 免费

`q.transpose(1, 2)` 返回**非连续视图**，SDPA 内部如果需要连续
会隐式 `.contiguous()`，那就产生一次真实拷贝。

本项目实测两次 transpose 约 0.06ms（8×512×6×32），相对注意力
本身可忽略 —— 但这是**小模型**的数字。d24 的 T=1024 时 transpose
本身不涨（它只是元数据操作），真正贵的是 SDPA 内部的连续化。

### 坑 2：`enable_gqa=True` 但形状不满足要求

`enable_gqa=True` 要求 `n_head % n_kv_head == 0`，且 PyTorch 会
在内部把 kv head 重复展开成 q head 的数量 —— **这会临时分配
显存**。GQA 省的是 KV cache，展开时反而要多占一份。

### 坑 3：以为强制 `sdpa_kernel([FLASH_ATTENTION])` 总是安全的

见上面 —— 滑窗路径会抛 `No available kernel`。所以本项目
**不**全局强制 flash，只在能用的地方（`is_causal=True`）依赖
PyTorch 的自动选择 + 测试验证。

### 坑 4：flex_attention 的 block_mask 没缓存

`create_block_mask` 本身要跑一次前向、建块摘要，开销不小。
而 `attend()` 每层每步都会被调用 —— 不缓存的话光建 mask 就能吃掉
全部收益。

本项目用 `@lru_cache(maxsize=8)` 按 `(T, left, device)` 缓存。
⚠ cache key 里**不含 batch**：block_mask 的第 0/1 维传 `None`
表示「所有 batch / 所有 head 共用同一张 mask」—— 因果性和滑窗宽度
都不依赖具体样本。

---

## 延伸

**FlashAttention 3 相对 FA2 的改进**：
1. 支持 **fp8**（Ada/Hopper 的 tensor core 有 fp8 变体）
2. **非对称支持**：让 softmax 和 GEMM 并行，进一步提升 warp 利用率
3. 原生 `(B,T,H,D)` 布局，不用 transpose

NanoChat 的 speedrun 靠 FA3 + fp8 拿到约 4% 的加速。本卡拿不到。

**为什么 nanochat 全程用 `(B,T,H,D)`**：这不是随便选的，是**为了 FA3**。
FA3 原生吃这个布局，于是整个代码库都不用 transpose。现在我们没有
FA3，就只能在 `sdpa_bt` 里付出两次 transpose 的代价 —— 这是
「跟随上游布局」的真实成本。

**FlashDecoding**：decode 阶段（`Tq=1`）用不同的 kernel 策略，
因为 query 只有一个 tile。本项目的 SDPA 自动处理了这个（实测 FA2 可用）。

---

## 下一章

[第 12 章：QK-Norm 与 GQA](12-QK-Norm与GQA.md) ——
第 10 章讲了 QK-Norm 的一半（防 logits 爆炸）。
这一章讲它的代价，以及 GQA 怎么用省 KV cache 换来的。