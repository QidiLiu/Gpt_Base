"""
模型的零件。

这个文件刻意把每个零件拆成独立的小类/小函数，对应教程卷2 的每一章：

    Linear              -> 精度控制：为什么不用 autocast
    rms_norm/layer_norm -> 为什么可以去掉 bias 和缩放参数
    apply_rotary_emb    -> RoPE 旋转位置编码
    sdpa_bt/attend      -> 注意力三步曲 + SDPA 布局陷阱
    CausalSelfAttention -> QK-Norm 与 GQA
    MLP                 -> ReLU 平方 vs GELU
    Block               -> 残差流与 Pre-LN

▓▓ 这一章要你敲的部分 ▓▓
    除「🔶 规格」和「📖 只读」之外的全部。

    建议顺序（每个都能独立验证）：
      1. rms_norm            (3 行)   教程 08
      2. precompute_rope     (10 行)  教程 09
      3. apply_rotary_emb    (8 行)   教程 09
      4. sdpa_bt             (5 行)   教程 11  ★ 有个必须知道的大坑
      5. attend              (20 行)  教程 10
      6. MLP                 (10 行)  教程 13
      7. CausalSelfAttention (40 行)  教程 10/12
      8. Block               (10 行)  教程 14
"""

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import log0

from common import COMPUTE_DTYPE


# ===========================================================================
# ❗ 1. Linear：精度控制的核心（教程卷2 第 15 章相关）
# ===========================================================================
class Linear(nn.Linear):
    """
    一个会「在 forward 时把权重转成激活值 dtype」的 nn.Linear。

    ── 为什么需要这个类？────────────────────────────
    混合精度的朴素做法是 torch.amp.autocast：包住前向，框架自动把
    支持低精度的算子转到 bf16。但这样做的代价是：
      1. 「哪些算子跑在低精度」由框架的注册表决定，不在代码里，很难追踪；
      2. 想临时改精度只能靠嵌套 context manager；
      3. 权重（优化器要更新的对象）和激活值的精度纠缠在一起。

    我们的做法是把精度控制点收敛到一处：
      · 参数永远存 fp32 -> 优化器的数值精度有保证（不会出现 bf16 的 1-beta2 归零）
      · forward 时把权重 cast 成 COMPUTE_DTYPE 再做矩阵乘 -> 走 bf16 tensor core

    结果和 autocast 等价，但「哪里在降精度」是显式的一行代码。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 权重是 fp32 master copy；x 通常已经是 COMPUTE_DTYPE（来自 embedding）
        # ❗ 一行。提示：F.linear(x, W.to(x.dtype))
        raise NotImplementedError("待实现：return F.linear(x, self.weight.to(dtype=x.dtype))")


# ===========================================================================
# ❗ 2. 归一化（教程卷2 第 08 章）
# ===========================================================================
def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """
    RMSNorm（论文：https://arxiv.org/abs/1910.07467）

        RMSNorm(x) = x / RMS(x) * g

    对比 LayerNorm：
        LayerNorm(x) = (x - mean(x)) / std(x) * g + b

    三个关键差异：
      1. 不减均值 —— 只算均方根
      2. 没有偏置 b
      3. 只有一个可学习缩放 g（nanochat 连 g 都不要）

    为什么可以减掉均值和偏置？
      实证结论：LLM 的隐藏激活本来就在零点附近近似对称，
      减均值带来的收益很小，去掉均值 + 偏置能省参数、省一次归约。

    ── 你要写的 ──
    PyTorch 有内置的 F.rms_norm，但它要求「归一化 + 缩放」一步做完。
    我们这里**不要缩放参数**（可学习 g 的省略是刻意的，见教程 08），
    所以要手写：先算均方根，再除。

    要点：
      · 沿最后一维（hidden dim）算，均方根要 **keepdim**
      · 归一化的分母加 eps 防止 0 除
      · 注意 x 可能是 bf16；实测量级下 bf16 做归一化是安全的
        （RMSNorm 原论文就是为低精度设计的）
    """
    raise NotImplementedError(
        "待实现：rms_norm ——\n"
        "  提示：\n"
        "    var = x.pow(2).mean(-1, keepdim=True)     # 沿 hidden 维的均方\n"
        "    return x * torch.rsqrt(var + eps)         # 或 x / (var+eps).sqrt()\n"
        "  eps 取 1e-6（参考 RMSNorm 论文）。\n"
        "  参考实现：git show solution:src/model/layers.py")


def layer_norm(x: torch.Tensor, weight, bias) -> torch.Tensor:
    """
    标准 LayerNorm，带可选 bias。存在的唯一目的是**消融对比**：
    把 ModelConfig.norm_type 改成 "layer"，就能测出 RMSNorm 到底赢了多少。

    ── 你要写的 ──
    F.layer_norm(input, normalized_shape, weight, bias, eps)
    normalized_shape 用 x.size(-1) 的 tuple 形式。
    eps 用 1e-5（与 nanoGPT / nanochat 一致）。
    """
    raise NotImplementedError(
        "待实现：return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)")


def make_norm(cfg, ndim: int, device=None):
    """
    按配置构造归一化层。

    RMSNorm 完全无参 -> 返回的是一个**函数**，不是 nn.Module。
    这样 nn.ModuleList 里就不会有一堆空模块，参数量统计也更干净。
    """
    if cfg.norm_type == "rms":
        return rms_norm
    if cfg.norm_type == "layer":
        # ❗ 两行：造一个全 1 的 weight（+ 可选全 0 的 bias），
        #    再返回一个 lambda 把它包起来
        raise NotImplementedError(
            "待实现：make_norm 的 layer 分支 ——\n"
            "  w = nn.Parameter(torch.ones(ndim, device=device))\n"
            "  b = nn.Parameter(torch.zeros(ndim, device=device)) if cfg.use_bias else None\n"
            "  return lambda x: layer_norm(x, w, b)")
    raise ValueError(f"未知 norm_type: {cfg.norm_type}")


# ===========================================================================
# ❗ 3-4. RoPE（教程卷2 第 09 章）
# ===========================================================================
def precompute_rope(seq_len: int, head_dim: int, base: float = 100000.0,
                    device=None, dtype=torch.float32):
    """
    预算 RoPE 的 cos / sin 表。

    公式（RoFormer 论文 https://arxiv.org/abs/2104.09864）：
        inv_freq[i] = 1 / base^(2i/d)        i = 0, 1, ..., d/2-1
        angle[t][i] = t × inv_freq[i]        位置 t 在第 i 个频率通道上的角度
        cos[t][i] = cos(angle[t][i])
        sin[t][i] = sin(angle[t][i])

    直观理解：把 head_dim 维切成 d/2 对，每对当成一个二维平面上的点，
    用不同频率的角速度旋转它。位置越靠后，转得越多。
    低频通道转得慢（管长距离依赖），高频通道转得快（管近距离依赖）。

    ── 你要写的 ──
    返回值形状必须是 (1, seq_len, 1, head_dim/2)，这样在
    apply_rotary_emb 里能和 (B, T, H, D) 广播。

    三个容易搞错的点：
      1) 通道维度是 `arange(0, head_dim, 2)`（**步长 2**），
         不是 arange(head_dim)。因为两个通道配成一对旋转。
         长度 = head_dim // 2。
      2) 频率是 inv_freq = base^(-channel/head_dim)，
         分母是 head_dim 不是 head_dim/2。
      3) freqs = outer(位置, 频率)，形状 (seq_len, head_dim/2)。
         然后 cos/sin 各加两维变成 (1, seq_len, 1, head_dim/2)：
         第 0 维是 batch，第 2 维是 head（为了广播）。
    """
    raise NotImplementedError(
        "待实现：precompute_rope ——\n"
        "  channel = torch.arange(0, head_dim, 2, dtype=float32, device=device)\n"
        "  inv_freq = 1.0 / (base ** (channel / head_dim))\n"
        "  t = torch.arange(seq_len, dtype=float32, device=device)\n"
        "  freqs = torch.outer(t, inv_freq)                    # (S, D/2)\n"
        "  cos = freqs.cos().to(dtype)[None, :, None, :]      # (1,S,1,D/2)\n"
        "  sin = freqs.sin().to(dtype)[None, :, None, :]\n"
        "  return cos, sin\n"
        "验证：uv run pytest -k rope -v")


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    对 q 和 k 施加 RoPE。

    x 形状 (B, T, H, D)  —— 注意是 **head_dim 在最后一维**，
    不是 SDPA 习惯的 (B, H, T, D)。我们在 attention 里直接用 (B,T,H,D)，
    省掉两次 transpose（nanochat 也是这么做的）。

    ── 你要写的 ──
    把最后一维切成前后两半，配对旋转（GPT-NeoX 风格）：
        x = [x1 | x2]           x1 = x[..., :d],  x2 = x[..., d:]
        y1 = x1·cos + x2·sin     <- 我们实现的是这个符号约定
        y2 = x1·(-sin) + x2·cos
    最后 cat 回去。

    差一个负号，等价于「用 -θ 旋转」。因为只关心 q 和 k 的**相对**旋转量，
    两者等价，但为了和已有 checkpoint 兼容，我们沿用 nanochat 的约定。

    验证：uv run pytest -k rope -v
      · test_rope_is_rotation_preserves_norm   旋转必须保持 L2 范数
      · test_rope_relative_position            内积只依赖相对距离
    """
    raise NotImplementedError(
        "待实现：apply_rotary_emb ——\n"
        "  assert x.ndim == 4\n"
        "  d = x.size(3) // 2\n"
        "  x1, x2 = x[..., :d], x[..., d:]\n"
        "  return torch.cat([x1*cos + x2*sin, x1*(-sin) + x2*cos], dim=3)\n"
        "  参考实现：git show solution:src/model/layers.py")


# ===========================================================================
# ❗ 5. SDPA 布局陷阱（教程卷2 第 11 章）★ 本项目最危险的一个坑
# ===========================================================================
def sdpa_bt(q, k, v, causal, mask=None, gqa=False):
    """
    调用 PyTorch SDPA，并处理「布局不匹配」这个坑。

    ── 坑在哪？───────────────────────────────────────────
    PyTorch 的 scaled_dot_product_attention **严格要求 (B, H, L, E) 布局**。
    我们全模型用的是 (B, T, H, D)，直接喂进去会被当成 (B, H, T, D)：

        后果 A：q 的 T 和 k 的 S 撞车时，直接 RuntimeError
        后果 B：q/k/v 内部自洽时，**不报错、输出形状也对，
                但算的是另一个东西**（头数和长度被对调了）

    亲手看：
        uv run python scratch/hardware.py
        → 「形状相同 = True，数值相同 = False，最大绝对误差 = 4.78」

    ── 为什么会这样？─────────────────────────────────────
    因为 nanochat 这类代码全程用 (B,T,H,D) 并不是 PyTorch 的约定，
    而是 **Flash Attention 3** 的约定（FA3 原生吃 (B,T,H,D)）。
    我们没有 FA3，退回 SDPA，就只能迁就 PyTorch 的 (B,H,L,E)。

    ── 你要写的 ──
    进 SDPA 前把第 1 维和第 2 维对调，出来后再对调回来。
    两次 transpose 的开销约 0.06ms（实测 8×512×6×32），可忽略。

    验证：uv run pytest -k sdpa -v
    """
    raise NotImplementedError(
        "待实现：sdpa_bt ——\n"
        "  out = F.scaled_dot_product_attention(\n"
        "      q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),\n"
        "      attn_mask=mask, is_causal=causal, enable_gqa=gqa)\n"
        "  return out.transpose(1, 2)\n"
        "验证：uv run pytest -k sdpa -v")


def attend(q, k, v, window, attn_impl="flex"):
    """
    训练时的注意力。

    window: (left, right) 元组
      (T, 0)  = 全上下文
      (W, 0)  = 滑动窗口，只看最近 W 个
      (0, 0)  = 只看自己

    ── 两条路径，以及为什么要分 ────────────────────────────────
    **全上下文（left >= T）永远走 SDPA 的 is_causal=True。**
    这条路径本来就选中 flash kernel（= Flash Attention 2），不物化
    (T,T) 的注意力矩阵，是最快的路。attn_impl 开关对它**没有影响**。

    **滑窗才需要选择：**
      attn_impl="flex"  flex_attention + 块级 block_mask → FA2 风格 kernel
                       实测 0.18 ms（SDPA 显式 mask 是 0.50 ms）
      attn_impl="sdpa"  物化 (T,T) 的 bool mask，SDPA 退回 mem-efficient

    为什么 SDPA 没法直接做滑窗 FA2：它的 flash 后端**不接受 attn_mask**，
    硬钉会抛 `No available kernel`。所以要走 FA2 风格的滑窗 kernel，
    只能换一条路（flex_attention），而不是换 SDPA 的参数。

    这不是 bug，是硬件现实 —— 教程卷3 第20章会让你亲眼看到这个性能悬崖，
    而 `attn_impl` 就是量它的那把尺子（把它设成 "sdpa" 就能量到 2.8×）。

    ── 你要写的 ──
    1) left = window[0];  T = q.size(1)
    2) 如果 left 为 None / <= 0 / >= T（全上下文）
         -> sdpa_bt(q, k, v, causal=True, gqa=(q.size(2) != k.size(2)))
    3) 否则按 attn_impl 分派（这两条已经写好了，直接调）
         if attn_impl == "flex": return attend_sliding_flex(q, k, v, left)
         return attend_sliding_sdpa(q, k, v, left)

    验证：uv run pytest -k "attend or causal" -v
      · test_attend_matches_manual_reference   与手写 reference 逐元素一致
      · test_sliding_window_only_sees_recent  滑窗确实限制了可见范围
      · test_attn_impl_does_not_change_full_context_result  开关不影响全上下文
    """
    raise NotImplementedError(
        "待实现：attend —— 见 docstring 的三步\n"
        "参考实现：git show solution:src/model/layers.py")


def attend_with_kvcache(q, k_cache, v_cache, k, v, kv_cache, window,
                        attn_impl="flex"):
    """
    推理时带 KV cache 的注意力。**卷 7 会详细讲，但这里先实现掉。**

    形状约定 (B, T, H, D)，T 是**本次要算的位置数**（decode 时 T=1）。
    k_cache / v_cache 形状 (B, S, H, D)，S 是 cache 容量。

    ── 你要想清楚的四个点 ──────────────────────────────

    (1) 怎么把新的 k/v 写进 cache？
        提示：**不要用 scatter_**。曾经用
            `pos.expand(-1, Tq, -1, -1).scatter_(1, pos, k)`
        结果只写进去了 batch 的第 0 行，第 1 行及以后全是未初始化垃圾
        （stride-0 索引的坑）。
        正确做法：切片赋值。前提是 batch 内所有行同步前进
        —— 这本来就是 KVCache 的约定。

    (2) 有效长度 S 是多少？
        提示：不是 k_cache.size(1)（那是容量），而是
        「已填长度 + 本次新增」。尾部是未初始化数据，
        读了会「看到未来」。

    (2b) ★ 本函数**只读** cache_seqlens，不推进它。
        推进由调用方（CausalSelfAttention.forward）在**最后一层**
        统一做一次：kv_cache.advance(T)。
        如果这里自己 advance，或者每层都 advance，n_layer 层就会
        推进 n_layer 次 —— 各层写到不同位置，attention 读到别人的 k/v。

    (3) causal 参数传什么？
        提示：这里传 False。因为因果性已经由「只取 0..S-1 且 S 正好
        等于当前位置+1」保证了。

    (4) 滑窗时 mask 里的位置怎么算？
        提示：query 的绝对位置 = cache_seqlens + [0..Tq)，
        key 的绝对位置 = 0..S-1。delta = qpos - kpos。

    验证：uv run pytest -k kv_cache -v
      · test_kv_cache_writes_all_batches    每行都写进去了
      · test_kv_cache_prefill_equals_naive  与朴素前向结果一致（黄金测试）
    """
    raise NotImplementedError(
        "待实现：attend_with_kvcache ——\n"
        "  B, Tq, H, D = q.shape\n"
        "  pos = int(kv_cache.cache_seqlens[0].item())\n"
        "  k_cache[:, pos:pos+Tq] = k.to(k_cache.dtype)\n"
        "  v_cache[:, pos:pos+Tq] = v.to(v_cache.dtype)\n"
        "  S = int(kv_cache.cache_seqlens[0].item()) + Tq\n"
        "  kk, vv = k_cache[:, :S], v_cache[:, :S]\n"
        "  dtype 不一致时 cast 成 q.dtype\n"
        "  gqa = (q.size(2) != kk.size(2))\n"
        "  全上下文 -> sdpa_bt(q, kk, vv, causal=False, gqa=gqa)\n"
        "  滑窗 -> 构造 (B,Tq,S) 的 mask 再 sdpa_bt(...)\n"
        "  ★ 本函数**不推进** cache_seqlens！推进由调用方\n"
        "    CausalSelfAttention.forward 在最后一层做一次 kv_cache.advance(T)。\n"
        "    漏掉的话生成结果像胡言乱语但没有任何报错（见上面 (2b)）。\n"
        "参考实现：git show solution:src/model/layers.py")


# ===========================================================================
# 🔶 规格部分：Value Embedding 的位置约定
# ===========================================================================
def has_value_embed(layer_idx: int, n_layer: int) -> bool:
    """
    Value Embedding 只加在「隔一层」和「最后一层」上。

    为什么不每层都加？
      残差流的容量有限，每加一路信号就多一份需要维护的信息。
      隔层加 + 末层必有，是 ResFormer 论文的折中：足够提供身份信息，
      又不至于淹没原始的上下文信号。
    """
    return layer_idx % 2 == (n_layer - 1) % 2


# ===========================================================================
# 🔶 规格：注意力后端（Flash Attention 2）
# ===========================================================================
# 本项目保证在支持的后端上真的用上 **Flash Attention 2**，而不是碰巧能用。
#
# ── 「碰巧能用」和「保证用上」差在哪？───────────────────────────────
#   `F.scaled_dot_product_attention` 会在所有可用后端里**静默挑选**一个。
#   挑到 flash 是运气好，但没有任何东西保证它：换个 torch 版本、换个 shape、
#   或者哪天某个参数让它挑不满足，就悄悄退回 mem-efficient 或 math，
#   而且**不报错**。教程卷3 第20章讲的「性能悬崖」就是这么发生的。
#
# ── 本机（RTX 4060 Ti / SM 8.9）实测的可用性 ──────────────────────
#   路径                                    FA2 可用   实测耗时
#   is_causal=True 全上下文                    ✓      0.22 ms
#   GQA（n_head != n_kv_head）                 ✓      —
#   KV cache decode                           ✓      —
#   滑窗（显式 attn_mask）                      ✗      0.50 ms
#   滑窗（flex_attention + block_mask）        ✓      0.18 ms  ← 2.8×
#
# ── 为什么滑窗要绕一圈用 flex_attention？──────────────────────────
#   SDPA 的 flash 后端**不接受任意 attn_mask**。硬钉它会直接炸：
#       RuntimeError: No available kernel. Aborting execution.
#   而 SDPA 的 mem-efficient 后端接受 mask，于是滑窗路径被迫降级，
#   并且要物化一个 (T,T) 的 bool mask（T=1024 时每层每步 1MB）。
#
#   flex_attention 是 PyTorch 内建的另一条路：它把 mask 编译成
#   **块级**的 block_mask（只存 128×128 的块摘要），不物化 (T,T)，
#   生成的 kernel 就是 FA2 风格的。所以它既有 FA2 的速度，又支持滑窗。
#
#   代价：flex_attention **必须 torch.compile**（eager 模式极慢）。
#   所以下面用「探测一次，失败回落」的思路 —— 回落目标是显式 mask 路径
#   （上表里的 0.50 ms）。数值上两者等价（实测最大误差 0.0039 = bf16 舍入），
#   所以回落不改变结果，只改变速度。

# 编译后的 flex_attention。缓存起来，全局只编译一次。
_FLEX = None
_FLEX_FALLBACK_LOGGED = False


def _flex_attention():
    """惰性编译 flex_attention。编译不了就返回 None（调用方负责回落）。"""
    global _FLEX
    if _FLEX is None:
        try:
            from torch.nn.attention.flex_attention import flex_attention
            _FLEX = torch.compile(flex_attention, dynamic=False)
        except Exception:
            _FLEX = False          # 编译不了 -> 永久走回落路径
    return _FLEX or None


@lru_cache(maxsize=8)
def _sliding_block_mask(T: int, left: int, device):
    """
    构造滑窗的 block_mask，并**缓存**。

    为什么要缓存：create_block_mask 本身要跑一次前向、建块摘要，开销不小。
    而 attend() 每层每步都会被调用 —— 不缓存的话，光建 mask 就能吃掉
    全部的注意力收益。

    cache key 里不含 batch：block_mask 的第 0/1 维传 None 表示
    「所有 batch / 所有 head 共用同一张 mask」。这正是我们要的 ——
    因果性和滑窗宽度都不依赖具体样本。
    """
    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        delta = q_idx - kv_idx
        return (delta >= 0) & (delta <= left)

    return create_block_mask(mask_mod, None, None, T, T, device=str(device))


def attend_sliding_sdpa(q, k, v, left):
    """
    滑窗注意力的原始实现：物化 (T,T) 的 bool mask，交给 SDPA。

    SDPA 的 flash 后端不接受 attn_mask，所以这条路径必然落到
    mem-efficient 后端。它被保留下来有两个原因：

      1. 它是 flex_attention 的回落目标（没有 C 编译器时也能跑）
      2. 它是教程卷3 第20章的**对照组** ——「滑窗的性能悬崖」这个教学点
         需要一个慢的参照物。两条路径都在，2.8× 的差距才可测量。
    """
    T = q.size(1)
    idx = torch.arange(T, device=q.device)
    delta = idx[:, None] - idx[None, :]          # (T, T)，delta[j,i] = j - i
    mask = (delta >= 0) & (delta <= left)         # 因果 ∧ 窗口
    return sdpa_bt(q, k, v, causal=False, mask=mask,
                   gqa=(q.size(2) != k.size(2)))


def attend_sliding_flex(q, k, v, left):
    """
    滑窗注意力走 flex_attention（FA2 风格 kernel）。

    形状约定沿用全项目：(B, T, H, D)。flex_attention 要 (B, H, T, D)，
    所以进出各 transpose 一次 —— 和 sdpa_bt 是同样的两次 transpose 开销。

    任何一步不可用（无 CUDA / 编译失败 / block_mask 建不出来）都回落到
    `attend_sliding_sdpa`。
    """
    flex = _flex_attention()
    if flex is None:
        return attend_sliding_sdpa(q, k, v, left)
    try:
        block_mask = _sliding_block_mask(q.size(1), left, q.device)
        out = flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                   block_mask=block_mask, enable_gqa=(q.size(2) != k.size(2)))
        return out.transpose(1, 2)
    except Exception:
        # 只警告一次。回落不是错误，只是慢 —— 训练结果依然正确。
        global _FLEX_FALLBACK_LOGGED
        if not _FLEX_FALLBACK_LOGGED:
            _FLEX_FALLBACK_LOGGED = True
            log0("  [attn] flex_attention 不可用，滑窗回落到显式 mask"
                 "（mem-efficient 后端，约慢 2.8 倍，数值等价）")
        return attend_sliding_sdpa(q, k, v, left)


# ===========================================================================
# ❗ 6-7. Attention（教程卷2 第 10、12 章）
# ===========================================================================
class CausalSelfAttention(nn.Module):
    """
    因果自注意力。

    ── 一步一个动作地看 ──────────────────────────────
    1) 投影：x (B,T,D) -> q,k,v
    2) 位置编码：对 q,k 施加 RoPE（**只对 q,k，不对 v**）
    3) 归一化：QK-Norm，防止 logits 爆炸
    4) 加权：softmax(QK^T/√d)·V，因果 mask
    5) 输出投影：合并所有头 -> (B,T,D)

    ── 关于形状 ─────────────────────────────────────────
    全程用 (B, T, H, D)，不用 SDPA 惯用的 (B, H, T, D)。
    原因：H 维度挨着特征维，取 head 的时候是 view(B,T,H,D) 一步到位；
    而 (B,H,T,D) 需要先 transpose 两次，最后再转回来。
    （代价是 SDPA 那边要额外转置两次 —— 见 sdpa_bt 的注释。）

    ── 关于四个投影分开写（而不是 nanoGPT 那种融合的 c_attn）──
    原因：Muon 要求同一组里的参数形状相同才能堆叠通信，
    而且分开后能对 q/k/v 做不同的处理（QK norm 只作用在 q,k）。

    ── QK-Norm 为什么需要 ────────────────────────────────
    q·k 的点积在 d 维上求和，logits = q·k/√d。
    训练早期权重随机，logits 可能很大 -> softmax 饱和成 one-hot
    -> 梯度消失。QK-Norm 把 q,k 拉回单位球面，logits 天然有界。
    代价：每个 head 只能表达「方向」，不能表达「强度」。
    （消融：`--no-qk-norm`，smoke 档实测 Δbpb 约 -0.004，即噪声量级）

    ── 你要写的 ──
    __init__：4 个 Linear（q/k/v/proj）+ 可选的 ve_gate
              ⚠ ve_gate 的输出维度是 **n_kv_head**（每个 kv 头一个标量门），
                不是 n_kv_head * head_dim。写成后者会广播失败。
    forward：上面 5 步
    """

    def __init__(self, cfg, layer_idx: int, device=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.d_model = cfg.n_embd
        self.qk_norm_scale = cfg.qk_norm_scale
        self.use_rope = cfg.use_rope
        # 滑窗走哪条路径："flex"（FA2 风格）| "sdpa"（显式 mask，当对照组用）
        self.attn_impl = getattr(cfg, "attn_impl", "flex")

        # ❗ 注意：use_rope 必须真的被 forward 检查。
        #   曾经的 bug：这个开关只出现在 GPT.describe() 的打印字符串里，
        #   forward 无条件施加 RoPE，于是 --no-rope 这个消融开关**完全无效**，
        #   两组 bpb 一模一样，消融表会得出「RoPE 毫无影响」的错误结论。
        #   （smoke 档实测：去掉 RoPE 后 Δbpb = +0.0877，是最大的单项影响）

        assert cfg.n_head % cfg.n_kv_head == 0, "n_head 必须是 n_kv_head 的整数倍"
        raise NotImplementedError(
            "待实现：CausalSelfAttention.__init__ ——\n"
            "  kv_dim = cfg.n_kv_head * self.head_dim\n"
            "  self.c_q = Linear(d_model, n_head  * head_dim, bias=cfg.use_bias, device=device)\n"
            "  self.c_k = Linear(d_model, n_kv_head* head_dim, ...)\n"
            "  self.c_v = Linear(d_model, n_kv_head* head_dim, ...)\n"
            "  self.c_proj = Linear(d_model, d_model, ...)\n"
            "  self.ve_gate_channels = 12\n"
            "  self.ve_gate = Linear(ve_gate_channels, cfg.n_kv_head, bias=False, device=device) \\\n"
            "                   if (cfg.use_value_embeds and has_value_embed(layer_idx, cfg.n_layer)) else None\n"
            "参考实现：git show solution:src/model/layers.py")

    def forward(self, x, cos_sin, window, kv_cache=None, ve=None):
        """
        x       (B,T,D)
        cos_sin RoPE 的 cos/sin（已按当前偏移切好）
        window  (left, right)
        ve      Value Embedding（(B,T,kv_dim)），仅当 use_value_embeds 开启

        返回 (B,T,D)

        ── ★ KV cache：谁负责推进写指针？─────────────────────
        attend_with_kvcache 只**读** kv_cache.cache_seqlens 来定位写入位置，
        它自己不推进。推进必须由本函数负责，而且**只能在最后一层做一次**：

            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        为什么必须是最后一层？
          每一层都要把本层的 k/v 写到 cache 的同一个位置（pos 由
          cache_seqlens 决定，所有层共享）。如果每层都 advance，
          n_layer 层就会推进 n_layer 次，写指针直接跑飞，
          而且各层写到的位置互不相同 —— attention 读到的是别人的 k/v。
          只有最后一层 advance 一次，才能保证「所有层写完，指针恰好前进 T」。

        漏了这一步会怎样？
          cache_seqlens 恒为 0 -> 每次 decode 都写到位置 0，
          有效长度 S 恒等于 1 -> 模型永远只看得见自己那一个 token。
          症状是「生成出来的东西像胡言乱语，但没有任何报错」。
          验证：uv run pytest -k kv_cache_prefill -v
        """
        raise NotImplementedError(
            "待实现：CausalSelfAttention.forward ——\n"
            "  B, T, _ = x.shape\n"
            "  1) q = self.c_q(x).view(B, T, self.n_head,    self.head_dim)\n"
            "     k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)\n"
            "     v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)\n"
            "  2) ve 不为 None 时：\n"
            "       ve = ve.view(B, T, self.n_kv_head, self.head_dim)\n"
            "       gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))\n"
            "       v = v + gate.unsqueeze(-1) * ve\n"
            "  3) if self.use_rope and cos.size(1) >= T:\n"
            "         q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)\n"
            "  4) if self.qk_norm_scale > 0:\n"
            "         q = rms_norm(q) * self.qk_norm_scale\n"
            "         k = rms_norm(k) * self.qk_norm_scale\n"
            "  5) kv_cache is None -> attend(q,k,v,window, attn_impl=self.attn_impl)\n"
            "     否则：\n"
            "         kc, vc = kv_cache.get_layer(self.layer_idx)\n"
            "         y = attend_with_kvcache(q, kc, vc, k, v, kv_cache, window,\n"
            "                                 attn_impl=self.attn_impl)\n"
            "         ★★ 别漏了这一步（见上面 docstring 的「谁负责推进写指针」）：\n"
            "         if self.layer_idx == kv_cache.n_layers - 1:\n"
            "             kv_cache.advance(T)\n"
            "  6) return self.c_proj(y.contiguous().view(B, T, self.d_model))\n"
            "参考实现：git show solution:src/model/layers.py")


# ===========================================================================
# ❗ 8. MLP（教程卷2 第 13 章）
# ===========================================================================
class MLP(nn.Module):
    """
    前馈网络：升维 -> 非线性 -> 降维。尺寸 C -> 4C -> C。
    参数占了整个模型的大约 2/3。

    ── 为什么是 ReLU 平方而不是 GELU？────────────────────
    GELU(x) = x · Φ(x)，需要算 erf/exp，是一条平滑曲线。
    ReLU²(x) = max(0,x)²，是分段二次，导数在 x>0 时就是 2x，极其便宜。

    经验结论（消融数据见 tutorial/00 章）：
      smoke 档 d4：relu2 1.7136 vs gelu 1.7240，**relu2 好 0.0104**
      nanochat 的消融也是 relu2 略好，而且快得多。
      本项目默认 relu2，你可以 `--activation gelu` 做对比。

    ── 你要写的 ──
    __init__：两个 Linear（c_fc 升到 4C，c_proj 降回 C）
    forward：c_fc -> 激活 -> c_proj，激活按 self.activation 分支
    """

    def __init__(self, cfg, device=None):
        super().__init__()
        self.activation = cfg.activation
        # ❗ 两行
        raise NotImplementedError(
            "待实现：MLP.__init__ ——\n"
            "  self.c_fc   = Linear(d, 4*d, bias=cfg.use_bias, device=device)\n"
            "  self.c_proj = Linear(4*d, d, bias=cfg.use_bias, device=device)")

    def forward(self, x):
        raise NotImplementedError(
            "待实现：MLP.forward ——\n"
            "  x = self.c_fc(x)\n"
            "  relu2 -> F.relu(x).square()\n"
            "  gelu  -> F.gelu(x)\n"
            "  gelu_tanh -> F.gelu(x, approximate='tanh')\n"
            "  return self.c_proj(x)")


# ===========================================================================
# ❗ 9. Block（教程卷2 第 14 章）
# ===========================================================================
class Block(nn.Module):
    """
    一个 Transformer 块：注意力 + MLP，各自带一个残差连接。

        x = x + Attn(norm(x))
        x = x + MLP(norm(x))

    ── Pre-LN vs Post-LN ─────────────────────────────────
    Pre-LN（上面这种，nanochat 和 nanoGPT 都是）：
        残差主干上永远是「干净」的 x，梯度可以直接穿过 norm 前的旁路回去，
        所以深层也能稳定训练。代价是表示能力略弱（多路信息在相加时没有门控）。

    Post-LN（原始 Transformer 论文的写法）：
        x = norm(x + Sublayer(x))
        每一层的输入都被重新归一化，表示能力强，但梯度要穿过 norm 才能回去，
        深层容易不稳定，必须靠 warmup 救。

    现代 LLM 几乎清一色 Pre-LN。

    ── 你要写的 ──
    __init__：norm1 / attn / norm2 / mlp
    forward：先 attn 后 mlp，各自残差相加
    """

    def __init__(self, cfg, layer_idx: int, device=None):
        super().__init__()
        # ❗ 四行
        raise NotImplementedError(
            "待实现：Block.__init__ ——\n"
            "  self.norm1 = make_norm(cfg, cfg.n_embd, device)\n"
            "  self.attn  = CausalSelfAttention(cfg, layer_idx, device)\n"
            "  self.norm2 = make_norm(cfg, cfg.n_embd, device)\n"
            "  self.mlp   = MLP(cfg, device)")

    def forward(self, x, cos_sin, window, kv_cache=None, ve=None):
        raise NotImplementedError(
            "待实现：Block.forward ——\n"
            "  x = x + self.attn(self.norm1(x), cos_sin, window, kv_cache, ve)\n"
            "  x = x + self.mlp(self.norm2(x))\n"
            "  return x")
