# 16 · meta device 三步建模型

卷 3 的第一课。前面 15 章都在讲「模型长什么样」，从这一章开始讲
**「怎么把它造出来」** —— 而造它的第一步就藏着本项目最阴的一个 bug。

---

## 本章目标

- 说清 meta device 三步法到底跳过了什么（以及它没跳过的什么）
- 手抄 `GPT.__init__` 里「注册 buffer」那几行，和 `init_weights` 的整体结构
- 亲手复现 RoPE 表变垃圾的故障，并搞明白为什么它比 NaN 更难查

---

## 前置回顾

[第 15 章](15-权重绑定与logit-softcap.md) 走完了 `GPT.forward` 的输出头。
一个 GPT 的数学部分到此为止完整了 —— 从**读 token id** 到**出 logits**。

卷 3 剩下 4 章讲 nanochat 的 5 个架构 trick。但在那之前，得先解决
一个工程问题：**上面那些代码跑在什么设备上、内存从哪来、初始值怎么填。**

---

## 概念：普通建模型的两段浪费

最直觉的写法：

```python
model = GPT(cfg)          # ① 在 CPU 上建：分配内存 + 填随机数
model = model.to("cuda")  # ② 搬到 GPU
```

`full` 档有 346M 参数。fp32 下是 **1.29 GiB**。这段代码干了：

1. 在 CPU 上分配 1.29 GiB，**并写入 346M 个随机数**（慢）
2. 通过 PCIe 拷贝 1.29 GiB 到 GPU（更慢）
3. **峰值占用：CPU 1.29 GiB + GPU 1.29 GiB，同一瞬间**

第 3 点是真正的痛点：普通机器上你得有两份。16 GB 显存的机器
训练时，那 1.29 GiB CPU 内存虽然不多，但**对更大的模型就是致命的**
（70B 模型的权重是 140 GB）。

### meta device 三步法

```python
with torch.device("meta"):   # ① 只记形状，不分配内存，不产生随机数
    model = GPT(cfg)
model.to_empty(device)       # ② 直接在目标设备上分配（内容是垃圾）
model.init_weights()         # ③ 在 GPU 上原地初始化
```

关键在第 ① 步。`meta` 是 PyTorch 的一个**虚拟设备**：

```python
with torch.device("meta"):
    m = GPT(cfg)
m.transformer.h[0].mlp.c_fc.weight
# 形状        : (3072, 768)
# device      : meta
# 元素个数    : 2,359,296
# 读第一个元素: RuntimeError: Tensor.item() cannot be called on meta tensors
```

注意最后一行：**形状和元素个数都是真的，唯一缺的是「值」**。
meta 张量知道自己是 235 万个元素，但不知道任何一个是几。

所以 `nn.Linear` 的 `reset_parameters()` 在 meta 上调用
`init.kaiming_uniform_` 时，**不会产生任何随机数**（没有内存可写）。

### 实测对比

`full` 档（346M 参数）：

| | `GPT(cfg).to("cuda")` | meta 三步法 | 差距 |
|---|---|---|---|
| 建模型耗时 | 1.50 s | — | |
| `.to("cuda")` 耗时 | 0.24 s | — | |
| **合计** | **1.74 s** | **0.66 s** | **2.64×** |
| GPU 峰值 | 1.29 GiB | 1.30 GiB | **持平** |
| CPU 侧峰值 | 1.93 GiB | 不分配 | 省下 |

**两个反直觉的诚实结论**：

1. **GPU 显存峰值几乎不变（1.29 vs 1.30 GiB）。**
   meta 三步法**不省显存** —— 它省的是 CPU 侧的内存和时间。
   源码 docstring 里写「峰值内存从 (CPU 16G + GPU 16G) 降到
   (GPU 16G)」这句话是对的，但容易让人误以为 GPU 那半边变小了。
   它没有。
2. **加速比 2.64× 主要来自「不生成 CPU 随机数」。**
   1.50s 里的绝大部分是 CPU 上初始化 346M 个随机数 ——
   `.to("cuda")` 的拷贝只占 0.24s。

> ⚠ 本项目只有 346M 参数，所以**这个技巧在本机上不是必需的**
> —— 1.74s 对训练（41 小时）可以忽略。
> 它的价值在 scale：模型大 10 倍时，「CPU 生成随机数」那一步
> 会变成训练启动的主要等待项。**先学它，是因为它便宜，不是因为
> 本项目现在就需要它。**

---

## 概念：第 ② 步留下的「垃圾」是本章的主角

```python
model.to_empty(device=device)
```

`to_empty` 只做一件事：**分配内存，不管内容**。

它把 1.29 GiB 显存划给模型，然后**不去初始化它**。那些字节里
是上次这块显存被释放时留下的旧数据。

这就是问题所在。**模型的所有权重都成了垃圾。**

那为什么还要这么干？因为第 ③ 步会重填它们 —— 前提是第 ③ 步
**能覆盖到每一个需要覆盖的东西**。

---

## 概念：★ 本项目最阴的 NaN 坑（实际比 NaN 更阴）

`GPT.__init__` 里有这么一段：

```python
# RoPE 表。超算 10 倍长度：它很小（seq×head_dim/2×2），
# 多算一点省得以后要动态增长的麻烦。真的不够时 forward 会 assert。
self.rotary_seq_len = cfg.sequence_len * 10
cos, sin = precompute_rope(self.rotary_seq_len, cfg.head_dim,
                           base=cfg.rope_base, device=device,
                           dtype=COMPUTE_DTYPE)
self.register_buffer("cos", cos, persistent=False)
self.register_buffer("sin", sin, persistent=False)
```

**问题**：这里的 `device` 就是外层 `with torch.device("meta")` 的 meta！
所以 `cos` / `sin` 是**只有形状的 meta 张量**。

`init_weights` 里有一段专门救场：

```python
# ── 必须在这里重算 RoPE 表 ─────────────────────────────
# __init__ 里算的 cos/sin 是在 **meta device** 上创建的（只有形状）。
# to_empty() 只给它们分配「未初始化的垃圾内存」，不填任何值。
# 这就是 meta device 三步法最容易踩的坑：
#   凡是「在 __init__ 里创建的、依赖数据的量」，都必须在 init_weights 里重算。
cos, sin = precompute_rope(
    self.rotary_seq_len, cfg.head_dim, base=cfg.rope_base,
    device=self.get_device(), dtype=COMPUTE_DTYPE)
self.cos, self.sin = cos, sin
```

**漏掉这三行的后果是什么？** 直觉会以为 `loss = nan` —— 实测不是，
这正是它比 NaN 更阴的地方。

### 故障注入：四种情况实测

我故意把「重算 RoPE」这一步换成不同种类的垃圾：

| 注入的垃圾 | `max|cos|` | `isfinite` 判据 | 实际后果 |
|---|---|---|---|
| **NaN** | `nan` | **抓到** | loss = nan |
| **有限垃圾 `4.4e35`** | `4.39e+35` | **漏掉** | 训练正常跑，RoPE 全错 |
| **全零**（CUDA 全新页） | `0.0` | **漏掉** | 训练正常跑，**完全没有位置信息** |
| 真表（对照） | `1.0` | 通过 | 正常 |

（这张表是 solution 分支上做的四路真实注入；早期版本的判据只能
抓住 1/4，后来加固成四层检查 —— 见验证 1。）

**第二和第三行是重点。**

- 有限垃圾：值大到离谱，但 `isfinite` 放行
- 全零：最「干净」的垃圾，但也最致命 —— `cos=sin=0` 意味着
  **旋转矩阵变成了全零矩阵**，q 和 k 被乘成 0，注意力分数全是 0，
  softmax 输出均匀分布。**模型失去了全部位置感知**，但
  loss 曲线仍然平滑下降，你完全看不出来

> ⚠ 顺便一提：这也解释了为什么这三种情况**取决于「哪块内存」**。
> 全新进程的 CPU 分配通常是 NaN；内存 churn 之后的回收块
> 是上一次的数据；CUDA 的全新页通常是全零。
> **所以「这个 bug 会不会炸」取决于机器当时的内存状态** ——
> 换一台机器、同一个 commit，症状可能完全不同。

### 为什么全零时 loss 不 nan

`apply_rotary_emb` 是：

```python
return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=3)
```

`x * 0 = 0`，不产生 NaN。

而且注意 `CausalSelfAttention.forward` 里的**顺序**：

```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
q, k = rms_norm(q) * self.qk_norm_scale, rms_norm(k) * self.qk_norm_scale
```

**RoPE 之后是 QK-Norm。** 而 `rms_norm(全零向量)` = `全零向量`
（`0 / sqrt(0 + eps)` = 0）。所以下游看到的是「q 和 k 全是 0」，
注意力分数 0，softmax 均匀。

这个 bug 会一路「安静地」传播下去。**报错点（loss 下降但质量差）
和病根（cos 表没算）在完全不同的地方，中间没有一个变量是坏的。**

---

## 手抄代码

✍️ **本章的作业在 `src/model/gpt.py`，共三处待实现：**
`GPT.__init__` 的主体（`# ❗ 待实现：self.transformer…`）、
`init_weights`（六步全写在骨架 docstring 里）、模块级 `build_model`。

> 📌 三个块的代码标记：
> - `✍️` 要你敲进 `src/`（填骨架里的函数体）
> - `🆕` 自己写一个新文件（`scratch/` 里的实验脚本）
> - `📦` solution 分支已写好，直接读或直接跑

第 2 块里的五个 trick 开关本节先不写 —— 它们的注册代码在
第 17-19 章随开关出现逐个补；本章先让「最简模型」能建出来。
（骨架注释里那句「以下都是残差流 trick，默认全部关闭」位置留着即可。）

如果你在 `solution` 分支，这些全都已经是成品 —— 当注释读即可。

### 第 1 块：`build_model` —— 三步法的入口

<details>
<summary><b>👀 展开参考答案（先自己想 20 分钟）</b></summary>

```python
def build_model(cfg, device="cuda") -> GPT:
    """
    【meta device 三步法】建模型。

    为什么不用普通的 GPT(cfg) 然后 .to(device)？
      因为普通初始化会在**源设备**（CPU）上分配并写入所有随机数，
      再整体搬到 GPU。这意味着：
        · CPU 上先生成全部随机数（慢）
        · 再走 PCIe 搬过去（更慢）
        · 峰值内存 = CPU + GPU 两份

    meta device 三步法完全跳过这两步：
        1) with torch.device("meta"): 只算形状，不分配内存，不产生随机数
        2) model.to_empty(device)  : 直接在目标设备上分配（内容是垃圾数据）
        3) model.init_weights()    : 在 GPU 上原地初始化
    """
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    return model
```

</details>

### 第 2 块：`GPT.__init__` 的主体

<details>
<summary><b>👀 展开参考答案（先自己想 20 分钟）</b></summary>

```python
        # 注册模型的主体结构。词表用 padded（64 对齐），
        # forward 末端会把 logits 切回真实词表大小。
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded, cfg.n_embd, device=device),
            "h":   nn.ModuleList([Block(cfg, i, device)
                                  for i in range(cfg.n_layer)]),
        })
        self.lm_head = Linear(cfg.n_embd, padded, bias=False, device=device)
```

</details>

⚠ **RoPE 表那几行不在你敲的范围里** —— 骨架已经写好
（`self.rotary_seq_len`、`register_buffer("cos"/"sin", …, persistent=False)`），
本章的概念节讲的是它们**为什么会变成垃圾**，注意别挪动它们。

### 第 3 块：`init_weights` —— 六步初始化

骨架 docstring 把六步、为什么乘 `sqrt(3)`、每一步的口径都写全了。
下面的答案是 solution 对齐版；**与 solution 源码有两处刻意不同**，
已按 main 骨架的约定处理：

- **签名不带 `seed` 参数**（main 骨架：`def init_weights(self):`）——
  形状派生种子直接写死在函数里。solution 版为支持 `--seed` 加了参数；
  main 骨架态跟 AGENTS.md 的约定走（`GPT.init_weights` 默认按形状派生）。
- **没有 `share_value_embeds` 分支**（main 的 config 没有这个开关）。
- solution 源码在末尾还有一步「查表类参数降到 `COMPUTE_DTYPE` 省显存」；
  main 骨架的六步不要求它。想要完全对齐 solution，看
  `git show solution:src/model/gpt.py` 的 init_weights 末尾。

<details>
<summary><b>👀 展开参考答案（先自己想 20 分钟）</b></summary>

```python
    @torch.no_grad()
    def init_weights(self):
        cfg = self.config
        # ★ 按「形状」派生种子：形状相同的模型拿到同一个初始化，
        #   消融的 Δ 里不含初始化噪声；重跑历史结果逐位不变。
        torch.manual_seed(cfg.n_layer * 1000 + cfg.n_embd)

        # 1) 嵌入与反嵌入
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # 2) Uniform(-s, s) 的标准差是 s/√3，所以要乘 √3 才能和 std=s 的
        #    Normal 一样 —— 这是骨架 docstring 让你想通的那一行
        s = math.sqrt(3) * cfg.n_embd ** -0.5

        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)      # 输出投影 = 0 -> 恒等
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4 倍
            torch.nn.init.zeros_(block.mlp.c_proj.weight)      # 输出投影 = 0 -> 恒等

        # 3)★ norm_type="layer" 时 γ/β 是 nn.Parameter，meta + to_empty 之后
        #    是未初始化垃圾 —— 必须显式写回 (1, 0)。
        #    RMSNorm 无参，默认档位这里是 no-op（但要写上，第 08 章）。
        for block in self.transformer.h:
            for norm in (block.norm1, block.norm2):
                if isinstance(norm, nn.Module):
                    torch.nn.init.ones_(norm.weight)
                    if getattr(norm, "bias", None) is not None:
                        torch.nn.init.zeros_(norm.bias)

        # 4) tie_embeddings：wte 和 lm_head 共享同一份参数
        if cfg.tie_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

        # 5)★ 必须在这里重算 RoPE 表 —— 本章的主坑
        # __init__ 里算的 cos/sin 是在 **meta device** 上创建的（只有形状）。
        # to_empty() 只给它们分配「未初始化的垃圾内存」，不填任何值。
        cos, sin = precompute_rope(
            self.rotary_seq_len, cfg.head_dim, base=cfg.rope_base,
            device=self.get_device(), dtype=COMPUTE_DTYPE)
        self.cos, self.sin = cos, sin     # ★ 是「换掉」，不是 copy_
```

</details>

---

## 动手验证

### 验证 1：判据真的抓得住那三种垃圾

`test_meta_device_weights_are_all_finite`（`tests/test_core.py`）
是「四层检查」的粒度判据（形状/有限性/设备/初始 loss）。
**照做一次故障注入，你才能确认它有用**：

> 🔒 **需要先完成**：本章手抄代码（`build_model` + `__init__` 主体 +
> `init_weights`）。`-k meta` 的判据走 `build_model`，骨架态下挂的
> 就是这三处（报哪个补哪个）。

```bash
# 正常：应该通过
uv run pytest tests/test_core.py -k meta -v

# 故障注入 A：在 gpt.py 的 init_weights 末尾，把重算那三行注释掉
uv run pytest tests/test_core.py -k meta -v      # 应该红

# 撤销注入
git checkout src/model/gpt.py
```

**如果你只做了 A 而它没红，那这个判据是无效的。**
（solution 分支写作本章时做过完整的四路注入验证 —— 见概念节的表
—— 并据此把判据从「只能抓 1/4」加固到四层。）

### 验证 2：亲手看 meta 是什么

> 🔒 **需要先完成**：卷 2 全部 + 本章第 2 块（`GPT.__init__` 主体）。
> `GPT(cfg)` 会实例化全部 Block —— layers 没填完就建不出来。

```python
import sys, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import GPT

cfg = make_run_config("debug").model
with torch.device("meta"):
    m = GPT(cfg)

w = m.transformer.h[0].mlp.c_fc.weight
print(f"形状     : {tuple(w.shape)}")
print(f"device   : {w.device}")
print(f"元素个数 : {w.numel():,}")
try:
    w[0, 0].item()
except RuntimeError as e:
    print(f"读一个元素: {e}")

# 对比：真实设备上的形状一模一样
m2 = GPT(cfg)
print(f"真实张量 device: {m2.transformer.h[0].mlp.c_fc.weight.device}")
print(f"两者形状相同  : "
      f"{m.transformer.h[0].mlp.c_fc.weight.shape == m2.transformer.h[0].mlp.c_fc.weight.shape}")
```

实测：

```
形状     : (128, 32)
device   : meta
元素个数 : 4,096
读一个元素: Tensor.item() cannot be called on meta tensors
真实张量 device: cpu
两者形状相同  : True
```

### 验证 3：两条路线的耗时对比

> 🔒 **需要先完成**：同验证 2 + 本章第 1/3 块（`build_model` 与 `init_weights`）。

```python
import sys, time, torch
sys.path.insert(0, "src")
from common.config import make_run_config
from model.gpt import GPT, build_model

cfg = make_run_config("full").model

torch.cuda.synchronize(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.device("cpu"):
    m = GPT(cfg)
t_build = time.time() - t0
t0 = time.time(); m = m.to("cuda"); torch.cuda.synchronize()
peak_a = torch.cuda.max_memory_allocated() / 1024**3
t_a = t_build + time.time() - t0
del m; torch.cuda.empty_cache()

torch.cuda.synchronize(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time(); build_model(cfg, device="cuda"); torch.cuda.synchronize()
t_b = time.time() - t0
peak_b = torch.cuda.max_memory_allocated() / 1024**3

print(f"A: GPT(cfg).to(cuda)  {t_a:6.2f}s   GPU 峰值 {peak_a:.2f} GiB")
print(f"B: meta 三步法         {t_b:6.2f}s   GPU 峰值 {peak_b:.2f} GiB")
print(f"时间 {t_a / t_b:.2f}x，显存 {(peak_a - peak_b):+.3f} GiB")
```

实测：

```
A: GPT(cfg).to(cuda)   1.74s   GPU 峰值 1.29 GiB
B: meta 三步法          0.66s   GPU 峰值 1.30 GiB
时间 2.64x，显存 -0.005 GiB
```

**记住「显存持平」这个结果** —— 这是本项目实测的，
和「meta device 省显存」的常见说法不同。

---

## ★ 消融实验

本章没有模型架构消融（`build_model` 只影响启动，不影响数学）。
但有一个**工程消融**，而且它的结论恰恰是「别改」：

```bash
# 基线：meta 三步法
time bash script/train_base.sh smoke --no-resume

# 对照：临时把 build_model 换成普通两段式
#   def build_model(cfg, device="cuda"):
#       m = GPT(cfg)          # ← 注意：没有 with torch.device("meta")
#       return m.to(device)
time bash script/train_base.sh smoke --no-resume
```

**预期**：

- **bpb 完全一样**（逐位相同 —— 同样的种子，同样的初始值。
  `init_weights` 里第一行就是 `torch.manual_seed(...)`）
- 启动时间差 1 秒左右

> 🔒 **需要先完成**：卷 1-4 全部 + 本章（`build_model`）。
> 实测数字来自 solution 分支（见上表与本节输出）。

**这个消融的价值在于证明一件事：meta 三步法是一个纯粹的工程优化，
不改变任何数学。** 如果两条路线的 bpb 不同，那说明种子控制有问题 ——
那才是真 bug。

> ⚠ 别在 `full` 档上做（41 小时一次，测个 1 秒的差异）。
> 而且要注意：`--no-resume` 才会真的重建模型；不加的话它会加载
> checkpoint，走不到 `build_model`。

---

## 常见坑

### 坑 1：漏掉「在 `init_weights` 里重算依赖数据的量」

本章的主题。这是 meta device 三步法唯一的、也是致命的陷阱。

**记住规则**：`__init__` 里除了「算形状」还做了别的事，就必须想清楚
那件事在 `init_weights` 里要不要再来一遍。本项目有这些：

| `__init__` 里创建的 | 是纯形状壳吗 | `init_weights` 里要重算吗 |
|---|---|---|
| `nn.Linear` / `nn.Embedding` 权重 | 是 | ✅ `init_weights` 全部初始化 |
| `cos` / `sin`（RoPE 表） | **是**（但**算过**，只是算在 meta 上） | ✅ **必须重算** |
| `resid_lambdas` / `x0_lambdas` / `smear_lambda` / `backout_lambda` | 是 | ✅ 按 `use_*` 开关重新填 |
| `value_embeds` 权重 | 是 | ✅ |
| `smear_gate` / `ve_gate` 权重 | 是 | ✅ |

**自检方法**：写完 `init_weights` 后，逐条问「`__init__` 里算过的每个东西，
现在的值对吗」。答不上来的就是要重算的。

### 坑 2：以为 `register_buffer` 的东西会被 `init_weights` 覆盖

不会。`init_weights` 只碰 `nn.Parameter`（它用
`torch.nn.init.*` 和 `.data`）。**buffer 必须手动重新赋值**：

```python
self.cos, self.sin = cos, sin    # ✅ 直接换掉整个张量
```

注意是**换掉**而不是 `copy_` —— 因为 meta 阶段的 `self.cos` 只是个
形状壳，对它 `copy_` 会报错。

顺带一句：`persistent=False` 意味着 cos/sin **不进 checkpoint**。
这是对的 —— 它是常量，加载时重算即可。但这又是一个「
必须在 `init_weights` 里算对」的理由：checkpoint 救不了你。

### 坑 3：以为 meta device 也能前向

不能。meta 张量上没有数据，任何运算都会在「取到值」那一步报错。
所以 `build_model` 必须三步都做完才返回，`forward` 才可用。

### 坑 4：`to_empty` 之后忘了 `.data` 的语义

`to_empty` 之后就调 `init_weights`，所以要用 `@torch.no_grad()`。
`init_weights` 已经带了这个装饰器。**如果你手抄时漏了它**，
在 346M 参数上会直接 OOM（因为每个叶子张量都要存梯度）。

---

## 延伸

**meta device 的其他用法**

1. **`torch.func.functional_call` + meta** 做「只算 FLOPs 不分配内存」
   的分析 —— 本项目的 `estimate_flops_per_token` 走的是配置公式，
   不是真建模型。真要精确数，可以建一个 meta 模型来遍历结构
2. **模型并行 / FSDP 的 `init_empty_weights`**：DeepSpeed 和 FSDP
   都用同一套思路（空参数 + 分片初始化），meta device 三步法
   是它单机的简化版
3. **`accelerate.init_empty_weights()`** 是 HuggingFace 生态的封装

**为什么不直接在 meta 上跑前向**

「不分配内存就能算 FLOPs」听起来很美，但 shape 之外的东西全都拿不到：
权重值、batch 里的 token、`kv_bytes_per_token` 那种依赖实际 dtype 的计算。
所以「meta 建结构 + 真设备算数值」这个分工是必然的。

**torch 2.x 的相关能力**

- `torch.device("meta")` 从 2.0 起稳定
- `model.to_empty(device)` 从 1.13 起
- `torch.func` 系列（vmap / functional_call / meta 组合）是新解法，
  但对「建一个 70B 模型」这种朴素需求，三步法已经够简单

---

## 下一章

[第 17 章：resid_lambdas 与 x0_lambdas](17-resid-lambdas与x0-lambdas.md) ——

卷 3 正式进入 5 个 trick。第一个是最直接的：**在残差上加一个可学标量**。

第 14 章那条「信息总线」的比喻在这里第一次真正派上用场。
