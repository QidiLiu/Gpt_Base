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

import torch
import torch.nn as nn
import torch.nn.functional as F



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


def make_norm(cfg, ndim: int, device=None):
    """
    按配置构造归一化层。

    RMSNorm 完全无参 -> 返回的是一个函数，不是 nn.Module。
    这样 nn.ModuleList 里就不会有一堆空模块，参数量统计也更干净。
    """
    if cfg.norm_type == "rms":
        return rms_norm
    if cfg.norm_type == "layer":
        w = nn.Parameter(torch.ones(ndim, device=device))
        b = nn.Parameter(torch.zeros(ndim, device=device)) if cfg.use_bias else None
        return lambda x: layer_norm(x, w, b)
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
            y = attend(q, k, v, window)
        else:
            kc, vc = kv_cache.get_layer(self.layer_idx)
            y = attend_with_kvcache(q, kc, vc, k, v, kv_cache, window)
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


def attend(q, k, v, window):
    """
    训练时的注意力。

    window: (left, right) 元组
      (T, 0)  = 全上下文
      (W, 0)  = 滑动窗口，只看最近 W 个
      (0, 0)  = 只看自己

    ── 为什么要区分两条路径？────────────────────────────────────
    全上下文时我们走 SDPA 的 is_causal=True，它会选中最快的
    flash attention kernel，完全不物化 (T,T) 的注意力矩阵。

    但 SDPA **不支持滑动窗口**。要用滑窗，只能自己构造 (T,T) 的 bool mask，
    这会强制它退到 memory-efficient kernel，并且要物化 mask。
    nanochat 因此在有滑窗时依赖 Flash Attention 3（FA3 原生支持 window_size）。
    本项目没有 FA3，所以走「显式 mask」这条路，代价是变慢。
    这不是 bug，是硬件现实 —— 教程卷3 会让你亲眼看到这个性能悬崖。
    """
    gqa = (q.size(2) != k.size(2))
    left = window[0]
    T = q.size(1)
    if left is None or left <= 0 or left >= T:
        # 全上下文：最快路径
        return sdpa_bt(q, k, v, causal=True, gqa=gqa)
    # 滑窗：构造 mask。行 j（查询位置）能看到列 i 当且仅当 0 <= j-i <= left
    idx = torch.arange(T, device=q.device)
    delta = idx[:, None] - idx[None, :]          # (T, T)，delta[j,i] = j - i
    mask = (delta >= 0) & (delta <= left)         # 因果 ∧ 窗口
    return sdpa_bt(q, k, v, causal=False, mask=mask, gqa=gqa)


def attend_with_kvcache(q, k_cache, v_cache, k, v, kv_cache, window):
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
        # is_causal=False：因果性已经由「只取 0..S-1 且 S 正好等于当前位置+1」保证了
        return sdpa_bt(q, kk, vv, causal=False, gqa=gqa)

    # 滑窗：query 绝对位置 = cache_seqlens + [0..Tq)，key 位置 = 0..S-1
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
