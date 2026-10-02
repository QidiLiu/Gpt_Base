# 32 · 推理引擎与 KV cache

从「训练」转到「推理」。两者的数据流完全不同：
训练是 teacher forcing（一次给完整序列），推理是自回归（一次一个 token）。

本章会看到 KV cache 的**理论**收益（`O(L²)` → `O(L)`），
以及一个本项目实测的结果：**在 `full` 档 batch=1 上，
这个收益在墙钟时间上几乎看不到**。

---

## 本章目标

- 说清 KV cache 到底缓存了什么、为什么能省
- 理解「预分配 + 原地写入」这个设计（以及 `torch.cat` 的锯齿问题）
- 知道 `T=1` 在朴素路径上会撞 assert —— 这决定了 API 的形状

---

## 前置回顾

- [第 11 章](11-SDPA与FlashAttention.md)：`(B,H,L,E)` vs `(B,T,H,D)` 布局陷阱
- [第 19 章](19-Smear与Backout.md)：smear 的三个分支（训练/prefill/decode）
- [第 12 章](12-QK-Norm与GQA.md)：GQA 与 KV cache 的字节数

> **本章是只读教程。** 代码在 `src/inference/engine.py`
> 和 `src/model/gpt.py` 的 `forward(kv_cache=...)` 路径。

---

## 概念：为什么需要 KV cache

朴素生成（`model.generate`，第 32 章末尾会看到）每生成一个 token
就把整条序列重新前向一次：

```
生成第 1 个 token：前向 1 个位置   -> 算 1 次
生成第 2 个 token：前向 2 个位置   -> 算 2 次
...
生成第 L 个 token：前向 L 个位置   -> 算 L 次
总计 O(L²) 次矩阵乘
```

**关键观察**：第 2 次前向里，**前 L-1 个位置的 k 和 v 与第 1 次完全相同**
（输入没变，模型是因果的）。这部分计算是纯浪费。

KV cache 把每个位置算好的 k、v 存下来，下次只算新增的那个：

```
生成第 t 个 token：只前向 1 个位置
总计 O(L) 次矩阵乘
```

**对 L=1000，理论上这是 1000 倍的计算量差距。**

---

## 概念：形状约定

```python
# engine.py 的模块 docstring
形状约定
    q      (B, T, H,  D)   T 是「本次要算的位置数」
    k_cache(v_cache)  (B, S, H, D)   S 是「cache 容量（含已缓存的）」
    cache_seqlens      (B,)   每个 batch 元素已填到第几个位置

与 nanochat / FA3 的约定一致：head 维在最后，T 在第 1 维。
```

⚠ **注意 docstring 写的 `(B, S, H, D)`，但实际代码是
`(n_layers, B, S, H, D)`**（层在最外层）：

```python
# 形状 (n_layers, B, S, H, D)：层放最外层，方便一次 get_layer 切片
self.k_cache = torch.zeros(n_layers, batch_size, max_seq_len,
                           n_kv_head, head_dim, device=device, dtype=dtype)
```

**docstring 少写了最外层的 `n_layers`** —— 它描述的是
`get_layer()` 返回的那个视图，不是整个张量。

这是个小不一致，不影响功能，但**读 docstring 会以为
`self.k_cache` 是 `(B,S,H,D)`**。

### `cache_seqlens` 为什么是 tensor 不是 int

```python
# 已填到第几个位置（FA3 需要 int32）
self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
```

注释说「FA3 需要 int32」。

但本项目**没有 FA3**（第 11 章：滑窗走 flex_attention）。
不过 `flash_attn_with_kvcache` 的 API 确实要求这个形状和 dtype ——
**保留是为了将来接 FA3**。

⚠ 实际用它是**当成 Python int 用**的：

```python
def get_pos(self) -> int:
    """当前写指针位置（假设 batch 内所有行同步前进）。"""
    return int(self.cache_seqlens[0].item())

def advance(self, n_tokens: int) -> None:
    self.cache_seqlens += n_tokens
```

⚠ **`get_pos()` 里的 `.item()` 会触发 GPU→CPU 同步。**
每层调一次（24 次/step）—— 又一个固定开销。

**docstring 里那句「假设 batch 内所有行同步前进」很重要**：
本项目的 cache **不支持 batch 内不同长度**。
第 34 章（`generate_batch`）会用 `prefill_from` 把 prompt 复制成 N 份，
但那之后**所有行还是同步的**。

---

## 概念：预分配 + 原地写入

```python
def __init__(self, batch_size, n_kv_head, head_dim, n_layers,
             max_seq_len, device, dtype=None):
    dtype = dtype or COMPUTE_DTYPE
    self.k_cache = torch.zeros(n_layers, batch_size, max_seq_len, ...)
    self.v_cache = torch.zeros(n_layers, batch_size, max_seq_len, ...)
```

**docstring 的解释**：

> 关键设计：**预分配 + 原地写入**，生成过程中零显存分配。
> 如果每个 token 都 `torch.cat` 一次，不仅慢，还会因为
> 「旧 tensor 不能释放」而在显存上出现**锯齿状峰值**。

`torch.cat` 的问题：每次都分配一块新的、更大的内存，
而**旧的还在被引用**（要等这次前向结束才能释放）。
所以峰值是 `2 × 当前大小` 左右，而且每次增长都要搬一遍。

预分配则：
- 一次分配 `max_seq_len` 那么多
- 每次只往固定位置写
- **峰值 = max_seq_len 那一块，从头到尾不变**

### `prefill_from` —— RL rollout 的关键优化

```python
def prefill_from(self, other: "KVCache") -> None:
    """
    把另一个 cache 的内容复制进来并扩容 batch 维。

    用途：batch=1 prefill 一次 prompt，然后复制成 N 份并行采样 N 条回复。
    这是 RL rollout 的关键优化 —— prompt 的前向只算一次，
    而不是 N 次。见 generate() 的第 1、2 步。
    """
    assert self.get_pos() == 0, "目标 cache 必须还是空的"
    assert self.n_layers == other.n_layers
    assert self.n_kv_head == other.n_kv_head
    assert self.head_dim == other.head_dim
    assert self.max_seq_len >= other.max_seq_len
    pos = other.get_pos()
    self.k_cache[:, :, :pos] = other.k_cache[:, :, :pos]
    self.v_cache[:, :, :pos] = other.v_cache[:, :, :pos]
    self.cache_seqlens.fill_(pos)
    if other.prev_embedding is not None:
        # 扩 batch：(1, 1, D) -> (B, 1, D)
        self.prev_embedding = other.prev_embedding.expand(
            self.batch_size, -1, -1).clone()
```

**四条 assert 全部是形状检查** —— 目标 cache 必须空，
层数/头数/维度必须一致，容量必须够大。

**`.clone()` 那一行值得注意**：`expand` 返回视图，
`clone()` 之后才是独立内存。第 19 章讲过 `prev_embedding`
是 smear trick 的依赖（decode 阶段要用它）—— **漏了这个
`clone`，多行采样时会共享同一块内存。**

---

## 概念：★ T=1 在朴素路径上会撞 assert

这是本章最实用的一个发现。实测：

```python
model(torch.randint(0, 8192, (1, 1), device="cuda"))
```

**报错**：

```
File "src/model/gpt.py", line 278, in forward
    x = self._apply_smear(x, kv_cache)
File "src/model/gpt.py", line 351, in _apply_smear
    assert T > 1, "训练时序列长度必须 > 1"
AssertionError: 训练时序列长度必须 > 1
```

**原因**（第 19 章讲过的三个分支）：

```python
if kv_cache is None:
    assert T > 1, "训练时序列长度必须 > 1"
```

**没有 kv_cache 时，smear 只能靠切片 `x[:, :-1]` 拿前一个 token，
而 `T=1` 时那是空张量。**

所以：

> **单 token 前向必须走 `kv_cache` 路径**，
> 那条路径靠 `kv_cache.prev_embedding` 拿到前驱（第 19 章）。

**这直接决定了 API 的形状**：`generate` 必须是
「prefill 一次 + 逐 token decode」的形态，
不能是「每次都调 `model(整个序列)`」。

---

## 概念：★★ 实测 —— KV cache 在本项目规模上省不下时间

这是本章最重要的一段。理论是 `O(L²)` → `O(L)`，
但**实测不是**。

### 单次前向的时间与位置数无关

```python
import sys, time, statistics, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import build_model
from inference.engine import KVCache

cfg = make_run_config("full").model
model = build_model(cfg, device="cuda")
B, K, D = 1, cfg.n_kv_head, cfg.head_dim
L0 = 128
buf = torch.randint(0, cfg.vocab_size, (B, L0 + 8), device="cuda")
one = buf[:, -1:].contiguous()
seq = buf[:, :L0].contiguous()

def med(fn, n=25, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)

with torch.no_grad():
    c = KVCache(B, K, D, cfg.n_layer, 512, device="cuda")
    model(seq, kv_cache=c)
    print(f"  L=1   : {med(lambda: (model(one, kv_cache=c), c.advance(1))):.3f} ms")
    for L in (8, 32, 128, 256):
        print(f"  L={L:>3} : {med(lambda: model(buf[:, :L])):.3f} ms")
```

实测（`full` 档，24 层 / 768 维，batch=1，**中位数**）：

```
     位置数       中位数 ms      最小 ms
       1       26.549     21.915
       8       22.809     16.671
      32       24.119     15.360
     128       23.896     15.096
     256       25.083     19.887
```

**L=1 和 L=256 的时间基本相同（26.5 vs 25.1 ms）。**

**位置数差 256 倍，时间差 0.9 倍。**

（⚠ 这组数据噪声不小 —— 中位数 22.8~26.5，最小值 15.1~21.9。
但「L=1 不比 L=256 快」这个结论在噪声下依然成立。）

### 为什么

KV cache 的收益假设是「注意力/前向是 **FLOP bound**」。
但实测说明**它是 overhead bound**：

```
24 层 × ~12 个 kernel/层 ≈ 288 个 kernel
每次 launch 5-10 μs  ->  1.4~2.9 ms
```

实测 25 ms 里 launch 只占 7% —— **所以还有别的东西**。

### ★ 找到真正的开销：权重每步都重新转 dtype

`model/layers.py` 的 `Linear.forward`：

```python
class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
```

**矩阵参数是 fp32（第 18 章：只有嵌入类降了 bf16），
而激活是 bf16。所以每次前向都要把权重转一次。**

实测这个转换的成本：

```
fp32 矩阵参数: 169,871,064
每次 forward 要读 648 MiB、写 324 MiB
按 288 GB/s 带宽，光 cast 就要 3.54 ms

单次 weight.to(bf16): 0.0315 ms
240 次（每层 4 个投影 x 24 层 x 2.5）: 7.55 ms
```

**7.55 ms，占 25 ms 前向的 30%。**

**而这部分开销与序列长度完全无关** —— 所以它进一步稀释了
KV cache 的相对收益。

> ⚠ **这个 cast 在训练里是必须的**（第 28 章：
> 「master weights stay fp32 for optimizer precision,
> but matmuls run in the activation dtype」）。
> 训练时它被 fwd+bwd 的开销摊薄了；推理时它就是纯浪费。
>
> **推理的正确做法是提前把权重转好并常驻 bf16。**
> 本项目没有做 —— 因为本项目是训练项目，
> `model.generate` 只是训练末尾的「采样看看」（第 29 章 6.5 节），
> 跑 16 个 token，性能完全不重要。

---

## 概念：KV cache 的显存（这部分是精确的）

虽然时间收益不明显，**显存计算是精确的**：

```
  L=  1024:      72.0 MiB   kv_bytes_per_token 推算      72.0 MiB   一致=True
  L=  4096:     288.0 MiB   kv_bytes_per_token 推算     288.0 MiB   一致=True
  L= 16384:    1152.0 MiB   kv_bytes_per_token 推算    1152.0 MiB   一致=True
  L= 65536:    4608.0 MiB   kv_bytes_per_token 推算    4608.0 MiB   一致=True
```

**`memory_bytes()` 和 `kv_bytes_per_token() × max_seq_len` 精确相等。**

第 12 章算过 `kv_bytes_per_token() = 73728` 字节（`full` 档）：

```
24 层 × 2 (k和v) × 12 kv 头 × 64 head_dim × 2 字节 (bf16) = 73,728
```

**`L=65536` 时 4.5 GiB** —— 单条序列。
这就是为什么长上下文推理需要 GQA（第 12 章）或 MLA。

### decode 阶段的带宽账

```
  L=  1024: 权重 984.0 MiB + KV    72.0 MiB =  1056.0 MiB -> 理论下限 3.84 ms
  L=  4096: 权重 984.0 MiB + KV   288.0 MiB =  1272.0 MiB -> 理论下限 4.63 ms
  L= 16384: 权重 984.0 MiB + KV  1152.0 MiB =  2136.0 MiB -> 理论下限 7.78 ms
```

**权重 984 MiB 在所有长度下都是常数**，KV cache 随长度增长。

所以：
- **短序列**：瓶颈是权重（读 984 MiB）
- **长序列**：瓶颈变成 KV cache

`L=16384` 时 KV cache 已经和权重一样大了 —— **这时
第 12 章的 GQA 才开始有决定性价值**（`n_kv_head` 减半省一半 KV）。

---

## 读这段代码时注意什么：`generate` 的两个版本

**朴素版**（`gpt.py`）：

```python
@torch.inference_mode()
def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
    """
    朴素自回归：每生成一个 token 就把整条序列重新前向一次。
    O(L^2) 的重复计算。存在的意义是让你看清「KV cache 到底省了什么」
    —— inference/engine.py 会做同样的事但复用缓存，教程卷7 会对比。
    """
    assert isinstance(tokens, list)
    device = self.get_device()
    rng = torch.Generator(device=device).manual_seed(seed) if temperature > 0 else None
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    for _ in range(max_tokens):
        logits = self.forward(ids)[:, -1, :]
        next_id = sample_from_logits(logits, rng, temperature, top_k)
        ids = torch.cat((ids, next_id), dim=1)
        yield next_id.item()
```

**存在的意义是教学**（docstring 明说）：
「让你看清『KV cache 到底省了什么』」。

⚠ 而本章实测说：**在 `full` 档 batch=1 上它没省什么。**
所以这个对比在**这个规模上不成立**，在更大模型/更长序列上才成立。

**高效版**（`engine.py`）用 `KVCache`，并有 `prefill_from` 做批量采样。

---

## 动手验证

### 验证 1：单次前向与位置数无关

见上面「实测」那段。

**结论要记住**：`L=1` 不比 `L=256` 快。
这说明**前向不是 FLOP bound**。

### 验证 2：权重 cast 的成本

见上面。**7.55 ms / 25 ms = 30%。**

```bash
# 可以用 profiler 确认
uv run python -m torch.profiler \
    -c "import sys; sys.path.insert(0,'src'); ..." --profile_memory
```

### 验证 3：KV cache 显存公式

见上面那张表，四个长度全部精确一致。

### 验证 4：`T=1` 撞 assert

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import build_model
cfg = make_run_config("debug").model
m = build_model(cfg, device="cpu")
try:
    m(torch.randint(0, 64, (1, 1)))
except AssertionError as e:
    print("确认报错:", e)
```

**期望**：`确认报错: 训练时序列长度必须 > 1`

### 验证 5：cache 的行为判据

```bash
uv run pytest tests/test_engine.py -k "kvcache or prefill or cache" -v
```

覆盖：

- `test_kvcache_starts_empty_and_advances`
- `test_kvcache_layer_slicing`
- `test_prefill_from_copies_content`
- `test_prefill_from_rejects_nonempty_target`
- `test_prefill_from_rejects_shape_mismatch`
- `test_cache_capacity_exactly_fits`

---

## 常见坑

### 坑 1：以为 KV cache 一定大幅加速

**本项目规模上不成立。** 实测 1.09-1.15x，不是 192x。

**FLOPs 省了但时间没省** —— 因为 batch=1 的前向是 overhead bound。

### 坑 2：以为 `T=1` 能走朴素路径

**不能，会撞 assert。** 必须带 `kv_cache`。

### 坑 3：docstring 说 `k_cache` 是 `(B,S,H,D)`

**实际是 `(n_layers, B, S, H, D)`。** 少写了最外层。

### 坑 4：`get_pos()` 的 `.item()` 会同步

每层调一次（24 次/step），每次都是 GPU→CPU 同步。

**这是固定开销，与序列长度无关。**

### 坑 5：`prefill_from` 漏了 `.clone()`

`expand` 返回视图，`.clone()` 才是独立内存。
漏了会让多行采样共享 `prev_embedding`（第 19 章的 smear 会出错）。

### 坑 6：以为 cache 支持 batch 内不同长度

`get_pos()` 的 docstring 明说「**假设 batch 内所有行同步前进**」。
`prefill_from` 复制后所有行一样长。

---

## 延伸

**什么时候 KV cache 才真正重要**

| 条件 | KV cache 的收益 |
|---|---|
| batch=1，小模型，短序列 | **几乎没有**（本章实测） |
| batch=1，大模型（≥7B） | 明显（权重读取占主导，cast 开销占比下降） |
| 长序列（L ≥ 16384） | 明显（KV cache 本身变成主要带宽） |
| 大 batch | 明显（GPU 算力被用起来，时间接近 FLOP bound） |

**本项目的 `full` 档是 346M 参数 —— 在「小模型」这一档。**

**FlashDecoding**

decode 阶段 query 只有一个 tile，长度 1 的 GEMM 效率极低。
FlashDecoding 把 KV 分成多块并行算 —— 本项目没实现。

**PagedAttention（vLLM）**

`torch.cat` 的锯齿问题（第「预分配」那节）在**变长 batch**上更严重
（每条序列长度不同）。vLLM 的解法是把 KV cache 分成固定大小的「页」，
按需分配 —— 彻底避免碎片。

本项目的 `preallocate` 是同一思路的简化版（预分配最大长度），
**但要求所有行等长**。

**Speculative decoding**

KV cache 让「验证多个 token」变得便宜（它们共享同一个 prompt 前缀）。
配合 `prefill_from`（prompt 只算一次），
这正是 RL rollout 采样 N 条回复的标准做法。

**MQA / GQA 在推理侧的价值**

第 12 章说过 GQA 在本项目（单卡训练）没有收益。
但推理侧的账完全不同：

| `n_kv_head` | KV @ L=16384 | decode 带宽 |
|---|---|---|
| 12（MHA） | 1152 MiB | 2136 MiB |
| 4（GQA） | 384 MiB | 1368 MiB |
| 1（MQA） | 96 MiB | 1080 MiB |

**`n_kv_head=4` 时 decode 带宽少 36%。**
这就是所有现代 LLM 都用 GQA 的原因 —— **纯粹为推理而生**。

---

## 下一章

[第 33 章：生成循环与终止控制](33-生成循环与终止控制.md) ——

本章讲了 `KVCache` 的**数据结构**。下一章讲**控制流**：
生成怎么停下来、批量采样怎么对齐、工具调用怎么注入。

其中一个问题是本章埋下的伏笔 —— `src/` 里有**两个采样函数**：

| 文件 | 函数 | 在哪个维度上做 softmax |
|---|---|---|
| `model/gpt.py` | `sample_from_logits` | 完整 vocab（其余置 `-inf`） |
| `inference/engine.py` | `sample_next_token` | 只有 top-k 的 k 个值 |

**两者的 docstring 都说「先 top-k 再除温度」，但给的理由不同** ——
一个说「先裁性能更好」，一个说「先裁数值更准」。

**实测结论是：分布完全相同，但同一个 seed 下取到的 token 不同。**
这个区别很实际：**换了 sampler 实现，旧 seed 的实验就不能复现了。**
