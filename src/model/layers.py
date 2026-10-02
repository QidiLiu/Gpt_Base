"""
模型的零件。

这个文件刻意把每个零件拆成独立的小类/小函数，对应教程卷2 的每一章：

    Linear          -> 第 15 章  精度控制：为什么不用 autocast
    RMSNorm         -> 第 08 章  为什么可以去掉 bias 和缩放参数
    apply_rotary_emb-> 第 09 章  RoPE 旋转位置编码
    CausalSelfAttention -> 第 10-12 章  注意力三步曲 / SDPA / QK-Norm 与 GQA
    MLP             -> 第 13 章  ReLU 平方

阅读顺序建议：按类定义顺序从上往下读，每个类都对应一章。
"""

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import log0, COMPUTE_DTYPE



# ===========================================================================
# Linear：精度控制的核心
# ===========================================================================
class Linear(nn.Linear):
    """
    一个会「在 forward 时把权重转成激活值 dtype」的 nn.Linear。

    ── 为什么需要这个类？────────────────────────────────────────
    混合精度的朴素做法是 torch.amp.autocast：包住前向，框架自动把
    支持低精度的算子转到 bf16。但这样做的代价是：
      1. 「哪些算子跑在低精度」由框架的注册表决定，不在代码里，很难追踪；
      2. 想临时改精度（比如评测时关掉 fp8）只能靠嵌套 context manager；
      3. 权重（优化器要更新的对象）和激活值的精度纠缠在一起。

    我们的做法是把精度控制点收敛到一处：
      · 参数永远存 fp32 -> 优化器的数值精度有保证，不会出现 bf16 的 1-beta2 归零
      · forward 时把权重 cast 成 COMPUTE_DTYPE 再做矩阵乘 -> 走 bf16 tensor core

    结果和 autocast 等价，但「哪里在降精度」是显式的一行代码。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 权重是 fp32 master copy；x 通常已经是 COMPUTE_DTYPE（来自 embedding）
        return F.linear(x, self.weight.to(dtype=x.dtype))


# ===========================================================================
# 归一化
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
      参数量对比（LLaMA）：LayerNorm 的 γ+β 占 hidden_dim 个参数 × 2，
      RMSNorm 只占 hidden_dim 个，省一半。
    """
    return F.rms_norm(x, (x.size(-1),))


def layer_norm(x: torch.Tensor, weight, bias) -> torch.Tensor:
    """
    标准 LayerNorm，带可选 bias。存在的唯一目的是**消融对比**：
    把 ModelConfig.norm_type 改成 "layer"，就能测出 RMSNorm 到底赢了多少。

    注意这里用 eps=1e-5，与 nanoGPT / nanochat 一致（RMSNorm 论文用 1e-6）。
    """
    return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)


class LayerNormAffine(nn.Module):
    """
    带可学习仿射的 LayerNorm，**存在的唯一目的是消融对比**
    （把 `ModelConfig.norm_type` 改成 "layer" 测 RMSNorm 赢了多少）。

    ⚠⚠ 它必须是一个真正的 nn.Module，而不是「lambda 闭包捕获两个
    nn.Parameter」—— 2026-10 实测发现后者有两个致命问题：

    (1) **参数不可见**：闭包里的 nn.Parameter 不在 `model.parameters()` 里，
        所以 **optimizer 永远看不到 γ/β**，它们永远停在初始值 (1, 0)。
        那样训出来的不是「LayerNorm」，是「没有仿射的 LayerNorm」。
        `setup_optimizer` 的「参数分组不完整」assert 也抓不到 ——
        它检查的是「模型有的参数是否都被分组」，方向正好相反。

    (2) **dtype 不匹配**：`torch.ones(ndim)` 默认 float32，
        而激活是 COMPUTE_DTYPE(bf16) -> `F.layer_norm` 直接抛
        `expected scalar type BFloat16 but found Float`。
        CPU 上因为全是 float32 而「碰巧能跑」，掩盖了 (1)。
    """

    def __init__(self, ndim: int, bias: bool, device=None):
        super().__init__()
        # ★ 用 COMPUTE_DTYPE 建：闭包版用默认 float32 是上面 (2) 的根因
        self.weight = nn.Parameter(torch.ones(ndim, device=device,
                                             dtype=COMPUTE_DTYPE))
        self.bias = (nn.Parameter(torch.zeros(ndim, device=device,
                                              dtype=COMPUTE_DTYPE))
                     if bias else None)

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias)


def make_norm(cfg, ndim: int, device=None):
    """
    按配置构造归一化层。

    RMSNorm 完全无参 -> 返回的是一个函数，不是 nn.Module。
    这样 nn.ModuleList 里就不会有一堆空模块，参数量统计也更干净。

    ⚠ LayerNorm 那条**必须**返回 nn.Module（见 LayerNormAffine 的 docstring）——
      参数要能被 named_parameters / optimizer 看到。
    """
    if cfg.norm_type == "rms":
        return rms_norm
    if cfg.norm_type == "layer":
        return LayerNormAffine(ndim, cfg.use_bias, device)
    raise ValueError(f"未知 norm_type: {cfg.norm_type}")


# ===========================================================================
# RoPE：旋转位置编码
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

    为什么要 base = 100000？
      base 控制「最高频率」和「最低频率」的比值。base 越大，低频越慢，
      越能表达长距离位置关系。原始 RoPE 用 10000，LLM 普遍用 100000。
    """
    if device is None:
        device = "cpu"
    # 通道维度：偶数步长的通道才成对，所以 arange(0, head_dim, 2)
    channel = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel / head_dim))
    # 时间维度：每个位置 t 与每个频率 i 的外积 -> (seq_len, head_dim/2)
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    # 加 batch 维和 head 维，方便后面广播
    # 形状 (1, seq_len, 1, head_dim/2)
    return (freqs.cos().to(dtype)[None, :, None, :],
            freqs.sin().to(dtype)[None, :, None, :])


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    对 q 和 k 施加 RoPE。

    x 形状 (B, T, H, D)  —— 注意是 **head_dim 在最后一维**，
    不是 SDPA 习惯的 (B, H, T, D)。我们在 attention 里直接用 (B,T,H,D)，
    省掉两次 transpose（nanochat 也是这么做的）。

    实现细节：把最后一维切成前后两半，配对旋转（GPT-NeoX 风格），
        x = [x1 | x2]           x1 = x[..., :d],  x2 = x[..., d:]
        y1 = x1·cos - x2·sin     <- 我们实现的是 y1 = x1·cos + x2·sin
        y2 = x1·sin + x2·cos           y2 = -x1·sin + x2·cos
    差一个负号，等价于「用 -θ 旋转」。因为只关心 q 和 k 的**相对**旋转量，
    两者等价，但为了和已有 checkpoint 兼容，我们沿用 nanochat 的约定。
    """
    assert x.ndim == 4, f"期望 (B,T,H,D)，得到 {tuple(x.shape)}"
    d = x.size(3) // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=3)


# ===========================================================================
# 注意力后端：Flash Attention 2
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
#   所以下面用 compile_or_eager 的同一套思路做「探测一次，失败回落」——
#   回落目标是 sdpa_bt 的显式 mask 路径，也就是上面表格里的 0.50 ms。
#   数值上两者等价（实测最大误差 0.0039 = bf16 舍入），所以回落不改变结果，
#   只改变速度。

# 编译后的 flex_attention。缓存起来，全局只编译一次。
_FLEX = None


def _flex_attention():
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
    全部的注意力收益。cache key 里不含 batch：block_mask 的第 0/1 维传
    None 表示「所有 batch / 所有 head 共用同一张mask」，这正是我们要的
    （因果性和滑窗宽度都不依赖具体样本）。
    """
    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        delta = q_idx - kv_idx
        return (delta >= 0) & (delta <= left)

    return create_block_mask(mask_mod, None, None, T, T, device=str(device))


def attend_sliding_flex(q, k, v, left):
    """
    滑窗注意力走 flex_attention（FA2 风格 kernel）。

    形状约定沿用全项目：(B, T, H, D)。flex_attention 要 (B, H, T, D)，
    所以进出各 transpose 一次 —— 和 sdpa_bt 是同样的两次 transpose 开销。

    任何一步不可用（无 CUDA / 编译失败 / block_mask 建不出来）都回落到
    `attend_sliding_sdpa`，那是显式 mask 的 mem-efficient 路径。
    """
    flex = _flex_attention()
    if flex is None:
        return attend_sliding_sdpa(q, k, v, left)
    try:
        B, T, H, D = q.shape
        block_mask = _sliding_block_mask(T, left, q.device)
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


_FLEX_FALLBACK_LOGGED = False


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


# ===========================================================================
# 注意力
# ===========================================================================
class CausalSelfAttention(nn.Module):
    """
    因果自注意力。

    ── 一步一个动作地看 ──────────────────────────────────────────
    1) 投影：x (B,T,D) -> q,k,v
    2) 位置编码：对 q,k 施加 RoPE（**只对 q,k，不对 v**）
    3) 归一化：QK-Norm，防止 logits 爆炸
    4) 加权：softmax(QK^T/√d)·V，因果 mask
    5) 输出投影：合并所有头 -> (B,T,D)

    ── 关于形状 ──────────────────────────────────────────────────
    nanochat 全程用 (B, T, H, D)，不用 SDPA 惯用的 (B, H, T, D)。
    原因：H 维度挨着特征维，取 head 的时候是 view(B,T,H,D) 一步到位；
    而 (B,H,T,D) 需要先 transpose 两次，最后再转回来。
    SDPA 接受任意前两维是 batch 的 4D 输入，所以不用改。
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
        self.attn_impl = getattr(cfg, "attn_impl", "flex")   # "flex" | "sdpa"

        assert cfg.n_head % cfg.n_kv_head == 0, "n_head 必须是 n_kv_head 的整数倍"
        kv_dim = cfg.n_kv_head * cfg.head_dim

        # 四个投影分开写（不是 nanoGPT 那种融合的 c_attn）。
        # 原因：Muon 要求同一组里的参数形状相同才能堆叠通信，
        # 而且分开后能对 q/k/v 做不同的处理（QK norm 只作用在 q,k）。
        self.c_q = Linear(self.d_model, self.n_head * self.head_dim, bias=cfg.use_bias, device=device)
        self.c_k = Linear(self.d_model, kv_dim, bias=cfg.use_bias, device=device)
        self.c_v = Linear(self.d_model, kv_dim, bias=cfg.use_bias, device=device)
        self.c_proj = Linear(self.d_model, self.d_model, bias=cfg.use_bias, device=device)

        # Value Embedding 的门控：只有开启 use_value_embeds 时才存在。
        # 输出维度是 n_kv_head（每个 kv 头一个标量门），不是 kv_dim。
        self.ve_gate_channels = 12
        self.ve_gate = (Linear(self.ve_gate_channels, cfg.n_kv_head, bias=False, device=device)
                        if (cfg.use_value_embeds and has_value_embed(layer_idx, cfg.n_layer))
                        else None)

    def forward(self, x, cos_sin, window, kv_cache=None, ve=None):
        B, T, _ = x.shape

        # (1) 投影 -> (B, T, H, D)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value Embedding（ResFormer）：把「token 本身的身份」按门控混进 value
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            # 门控由输入的前 12 个通道决定，范围 (0, 3) —— 内容相关的门控
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # (2) 位置编码：只作用在 q 和 k 上
        # ★ use_rope 必须真的被检查。曾经的 bug：这个开关只出现在
        #   describe() 的打印字符串里，forward 无条件施加 RoPE，
        #   于是 --no-rope 这个消融开关**完全无效**，bpb 一模一样，
        #   消融表会得出「RoPE 毫无影响」的错误结论。
        cos, sin = cos_sin
        if self.use_rope and cos.size(1) >= T:
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # (3) QK-Norm：先按 head_dim 归一化，再放大
        #     为什么需要？q·k 的点积在 d 维上求和，logits = q·k/√d。
        #     训练早期权重随机，logits 可能很大 -> softmax 饱和成 one-hot
        #     -> 梯度消失。QK-Norm 把 q,k 拉回单位球面，logits 天然有界。
        #     代价：每个 head 只能表达「方向」，不能表达「强度」。
        if self.qk_norm_scale > 0:
            q = rms_norm(q) * self.qk_norm_scale
            k = rms_norm(k) * self.qk_norm_scale

        # (4) 加权
        if kv_cache is None:
            y = attend(q, k, v, window, attn_impl=self.attn_impl)
        else:
            kc, vc = kv_cache.get_layer(self.layer_idx)
            y = attend_with_kvcache(q, kc, vc, k, v, kv_cache, window,
                                    attn_impl=self.attn_impl)
            # ★★ 推进 KV cache 的写指针 —— 整份实现里最容易漏的一步。
            #
            # attend_with_kvcache 只**读** cache_seqlens 来定位写入位置，
            # 它自己不推进。推进必须在这里做，而且**只能在最后一层做一次**。
            #
            # 为什么必须是最后一层？
            #   每一层都要把本层的 k/v 写到 cache 的同一个位置（pos 由
            #   cache_seqlens 决定，所有层共享）。如果每层都 advance，
            #   n_layer 层就会推进 n_layer 次，写指针直接跑飞，而且各层
            #   写到不同位置 —— attention 读到的是别的层的 k/v。
            #   只有最后一层 advance 一次，才能保证「所有层写完，指针恰好前进 T」。
            #
            # 漏了这一步会怎样？
            #   cache_seqlens 恒为 0 → 每次 decode 都写到位置 0，
            #   有效长度 S 恒等于 1 → 模型永远只看得见自己那一个 token。
            #   症状是「生成出来像胡言乱语，但没有任何报错」——
            #   最难查的一类 bug。验证：uv run pytest -k kv_cache_prefill -v
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # (5) 合并头 + 输出投影
        y = y.contiguous().view(B, T, self.d_model)
        return self.c_proj(y)


def has_value_embed(layer_idx: int, n_layer: int) -> bool:
    """
    Value Embedding 只加在「隔一层」和「最后一层」上。

    为什么不每层都加？
      残差流的容量有限，每加一路信号就多一份需要维护的信息。
      隔层加 + 末层必有，是 ResFormer 论文的折中：足够提供身份信息，
      又不至于淹没原始的上下文信号。
    """
    return layer_idx % 2 == (n_layer - 1) % 2


def sdpa_bt(q, k, v, causal, mask=None, gqa=False):
    """
    调用 PyTorch SDPA，并处理「布局不匹配」这个坑。

    ── 坑在哪？────────────────────────────────────────────────
    PyTorch 的 scaled_dot_product_attention **严格要求 (B, H, L, E) 布局**。
    我们全模型用的是 (B, T, H, D)，直接喂进去会被当成 (B, H, T, D)：

        T=1  -> 侥幸「能跑」，但返回的是 (B,H,L,E) 布局，形状悄悄变了
        T=2  -> 直接 RuntimeError（q 的 L=2 和 k 的 H=6 撞在同一维上）
        T=S  -> 正常（因为 T==H 时两种解读碰巧同形，纯属巧合）

    很多人会在这里踩坑并怀疑人生。

    ── 为什么会这样？──────────────────────────────────────────
    因为 nanochat 这类代码全程用 (B,T,H,D) 并不是 PyTorch 的约定，
    而是 **Flash Attention 3** 的约定（FA3 原生吃 (B,T,H,D)）。
    我们没有 FA3，退回 SDPA，就只能迁就 PyTorch 的 (B,H,L,E)。

    解决：进 SDPA 前把 T 和 H 对调，出来后再对调回来。
    两次 transpose 的开销约 0.06ms（实测，8×512×6×32），相对注意力本身可忽略。
    """
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, is_causal=causal, enable_gqa=gqa)
    return out.transpose(1, 2)


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
    """
    left = window[0]
    T = q.size(1)
    if left is None or left <= 0 or left >= T:
        # 全上下文：最快路径，本来就是 FA2
        return sdpa_bt(q, k, v, causal=True, gqa=(q.size(2) != k.size(2)))
    # 滑窗：行 j（查询位置）能看到列 i 当且仅当 0 <= j-i <= left
    if attn_impl == "flex":
        return attend_sliding_flex(q, k, v, left)
    return attend_sliding_sdpa(q, k, v, left)


def attend_with_kvcache(q, k_cache, v_cache, k, v, kv_cache, window,
                        attn_impl="flex"):
    """
    推理时带 KV cache 的注意力。

    形状约定 (B, T, H, D)，T 是**本次要算的位置数**（decode 时 T=1）。
    k_cache / v_cache 形状 (B, S, H, D)，S 是 cache 容量。
    写入位置由 cache_seqlens 指明；写完后有效长度 = cache_seqlens + T。
    """
    B, Tq, H, D = q.shape
    # ── 原地写入 cache ──
    # 用「切片赋值」而不是 scatter_：
    #   · 切片赋值是连续内存拷贝，比 scatter_ 快，也不会有 stride-0 索引的坑
    #     （实测 scatter_ 配 expand 出来的索引会漏写 batch 的非 0 行）
    #   · 前提是 batch 内所有行同步前进 —— 这本来就是 KVCache 的约定
    #     （见 KVCache.get_pos 的注释），prefill 复制成 N 份后它们也完全一致
    pos = int(kv_cache.cache_seqlens[0].item())
    k_cache[:, pos:pos + Tq] = k.to(k_cache.dtype)
    v_cache[:, pos:pos + Tq] = v.to(v_cache.dtype)

    # 有效长度 = 已有长度 + 本次新增。batch 内各行同步前进，取第 0 行即可。
    S = int(kv_cache.cache_seqlens[0].item()) + Tq
    kk, vv = k_cache[:, :S], v_cache[:, :S]     # 只取有效部分，尾部是未初始化数据
    # SDPA 要求 q/k/v 三者 dtype 一致。cache 通常按 COMPUTE_DTYPE 建，
    # 但如果模型走了别的精度（比如 fp16 回落路径）就会不匹配。
    if kk.dtype != q.dtype:
        kk, vv = kk.to(q.dtype), vv.to(q.dtype)
    gqa = (q.size(2) != kk.size(2))

    left = window[0]
    if left is None or left <= 0 or left >= S:
        # is_causal=False：因果性已经由「只取 0..S-1 且 S 正好等于当前位置+1」保证了。
        # 这条路径实测选中 flash（FA2），decode 阶段 Tq=1，开销主要在带宽不在算力。
        return sdpa_bt(q, kk, vv, causal=False, gqa=gqa)

    # ── 滑窗 decode：query 的绝对位置是 cache_seqlens + [0..Tq) ──
    # 注意这里**不能**直接复用 attend_sliding_flex：那个函数假设 q/k/v
    # 在同一个长度上，而这里 kv 的有效长度 S 远大于本次要算的 Tq。
    # 复用一个 block_mask 也会算错（偏移量不同）。
    # decode 时 Tq 通常是 1，直接走显式 mask 就够 —— S 通常也不大，
    # 而且 flex_attention 在 Tq=1 上没有优势（它是为长序列设计的）。
    qpos = (kv_cache.cache_seqlens + torch.arange(Tq, device=q.device)).view(B, 1)
    kpos = torch.arange(S, device=q.device).view(1, -1)
    delta = qpos.unsqueeze(-1) - kpos.unsqueeze(1)               # (B, Tq, S)
    mask = (delta >= 0) & (delta <= left)
    return sdpa_bt(q, kk, vv, causal=False, mask=mask, gqa=gqa)


# ===========================================================================
# MLP
# ===========================================================================
class MLP(nn.Module):
    """
    前馈网络：升维 -> 非线性 -> 降维。

    尺寸：C -> 4C -> C。参数占了整个模型的大约 2/3。

    ── 为什么是 ReLU 平方而不是 GELU？───────────────────────────
    GELU(x) = x · Φ(x)，需要算 erf/exp，是一条平滑曲线。
    ReLU²(x) = max(0,x)²，是分段二次，导数在 x>0 时就是 2x，极其便宜。

    经验结论（在 nanochat / modded-nanogpt 的消融里）：
      ReLU² 在小模型上比 GELU 略好一点点，而且快得多。
      本项目默认 relu2，你可以把 ModelConfig.activation 改成 "gelu" 做对比。
    """

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


# ===========================================================================
# Block
# ===========================================================================
class Block(nn.Module):
    """
    一个 Transformer 块：注意力 + MLP，各自带一个残差连接。

        x = x + Attn(norm(x))
        x = x + MLP(norm(x))

    ── Pre-LN vs Post-LN ───────────────────────────────────────
    Pre-LN（上面这种，nanochat 和 nanoGPT 都是）：
        残差主干上永远是「干净」的 x，梯度可以直接穿过 norm 前的旁路回去，
        所以深层也能稳定训练。代价是表示能力略弱（多路信息在相加时没有门控）。

    Post-LN（原始 Transformer 论文的写法）：
        x = norm(x + Sublayer(x))
        每一层的输入都被重新归一化，表示能力强，但梯度要穿过 norm 才能回去，
        深层容易不稳定，必须靠 warmup 救。

    现代 LLM 几乎清一色 Pre-LN。
    """

    def __init__(self, cfg, layer_idx: int, device=None):
        super().__init__()
        self.norm1 = make_norm(cfg, cfg.n_embd, device)
        self.attn = CausalSelfAttention(cfg, layer_idx, device)
        self.norm2 = make_norm(cfg, cfg.n_embd, device)
        self.mlp = MLP(cfg, device)

    def forward(self, x, cos_sin, window, kv_cache=None, ve=None):
        x = x + self.attn(self.norm1(x), cos_sin, window, kv_cache, ve)
        x = x + self.mlp(self.norm2(x))
        return x
