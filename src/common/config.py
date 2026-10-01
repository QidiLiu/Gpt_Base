"""
配置：单一旋钮（--depth）+ 分层开关。

这个文件是整个项目的「设计说明书」。两件事：

  (1) 【单一旋钮】只给你一个 depth，其余全自动推导。
      这是 nanochat 最核心的设计思想：用户不该需要理解 40 个超参之间的
      相互依赖关系，只该说「我要小一点 / 大一点」。
      推导过程（scaling law）在 resolve() 里，教程卷5 会逐行推导。

  (2) 【分层开关】架构上的每个「可选设计」都暴露成一个开关，
      默认值是 nanochat 的生产配置，但你可以单独关掉来做消融实验。
      每个开关在教程卷2/卷3 都有对应章节，告诉你它为什么存在。
"""

import math
from dataclasses import dataclass, field, asdict


# ===========================================================================
# 1) 模型结构配置
# ===========================================================================
@dataclass
class ModelConfig:
    # ---- 几何形状（由 depth 推导，但也可以直接指定）----
    n_layer: int = 6              # 深度（唯一的「复杂度旋钮」）
    n_embd: int = 384             # 模型维度 = 残差流宽度
    n_head: int = 6               # query 头数
    n_kv_head: int = 6            # kv 头数，< n_head 即为 GQA
    sequence_len: int = 1024      # 上下文长度 T
    vocab_size: int = 16384       # 由 tokenizer 决定
    head_dim: int = 64            # n_embd // n_head

    # ---- 归一化（消融：改成 "layer" 即可对比 LayerNorm）----
    norm_type: str = "rms"        # "rms" | "layer"

    # ---- 位置编码（消融：use_rope=False 即退回可学习位置嵌入）----
    use_rope: bool = True
    rope_base: float = 100000.0

    # ---- 注意力（消融：qk_norm_scale=0.0 即关掉 QK Norm）----
    qk_norm_scale: float = 1.2    # 0.0 = 不做 QK norm
    window_pattern: str = "L"     # "L" 全上下文 | "SSL" 滑窗平铺（滑窗需要 FA3）

    # ---- 滑窗用哪条实现路径（消融开关，见 model/layers.py:attend）----
    # "flex"  用 torch.nn.attention.flex_attention —— 编译出的是 FA2 风格的
    #         kernel，原生支持 block_mask 滑窗。本卡实测滑窗 0.50ms → 0.18ms。
    # "sdpa"  原始路径：物化 (T,T) 的 bool mask，SDPA 退回 mem-efficient 后端。
    #         **保留它不是为了性能，是为了教程卷3 第 20 章**：那一章整章教的
    #         就是「滑窗的性能悬崖」，两条路径都在才能把 2.8× 的差距测出来。
    #         全换成 flex 之后悬崖就没了，第 20 章失去对比对象。
    # 注意：全上下文（window >= T）永远走 is_causal=True 的 SDPA flash 快路径，
    # 不受这个开关影响 —— 那条路径本来就是 FA2，flex 在那里没有收益。
    attn_impl: str = "flex"

    # ---- MLP（消融：可选 "gelu"）----
    activation: str = "relu2"     # "relu2" | "gelu"

    # ---- 输出头 ----
    tie_embeddings: bool = False  # True = wte 与 lm_head 共享权重（nanoGPT 的做法）
    use_bias: bool = False        # Linear / Norm 里要不要 bias
    logit_softcap: float = 15.0   # 0.0 = 不做 softcap

    # ---- 残差流 trick（高级，默认全关；每项对应教程卷3 一章）----
    use_resid_lambdas: bool = False   # 逐层残差缩放
    use_x0_lambdas: bool = False       # 逐层回混初始嵌入
    use_value_embeds: bool = False     # Value Embeddings (ResFormer)
    use_smear: bool = False            # 混入前一 token 的嵌入
    use_backout: bool = False          # 末层前减去中层残差

    # 词表按 64 对齐（DDP 张量对齐 / tensor core 效率），forward 时再切回来
    pad_vocab_size_to: int = 64

    def window_sizes(self) -> list[tuple[int, int]]:
        """
        把 window_pattern 字符串展开成「每一层的窗口大小」。

        "L"  = 全上下文      -> (sequence_len, 0)
        "S"  = 四分之一上下文 -> (ceil(sequence_len/4 对齐到 128), 0)

        最后一层强制 L：最靠近输出的层需要看全局。
        """
        pattern = self.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"非法 window_pattern: {self.window_pattern}"
        long_w = self.sequence_len
        short_w = -(-long_w // 4 // 128) * 128  # 向上取整到 128 的倍数
        table = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [table[pattern[i % len(pattern)]] for i in range(self.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes


def build_model_config(depth: int, aspect_ratio: int, head_dim: int,
                       sequence_len: int, vocab_size: int, **overrides) -> ModelConfig:
    """
    【单一旋钮】从 depth 推导出完整形状。

        base_dim  = depth × aspect_ratio        模型宽度与深度成正比
        n_embd    = 向上取整到 head_dim 的倍数   保证 n_embd % head_dim == 0
        n_head    = n_embd / head_dim

    为什么宽度要和深度成正比（aspect_ratio 固定）？
      因为我们扫的是「等比例放大」的模型族，不是单独调宽度或深度。
      等比例缩放是 scaling law 成立的前提：只有形状一致，
      推导出的 batch size / LR / weight decay 才能跨 depth 迁移。
    """
    base_dim = depth * aspect_ratio
    n_embd = math.ceil(base_dim / head_dim) * head_dim
    cfg = ModelConfig(
        n_layer=depth,
        n_embd=n_embd,
        n_head=n_embd // head_dim,
        n_kv_head=n_embd // head_dim,   # 默认不做 GQA
        head_dim=head_dim,
        sequence_len=sequence_len,
        vocab_size=vocab_size,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ===========================================================================
# 2) 优化器配置
# ===========================================================================
@dataclass
class MuonConfig:
    """
    Muon（Momentum Orthogonalized by Newton-schulz）。

    flavor:
      "simple"   最简版：5 步 Newton-Schulz 正交化。教学默认，代码短。
      "advanced" 完整版：Polar Express 正交化 + MuonEq 行均衡
                  + Muon+ 重归一化 + NorMuon 方差缩减 + 谨慎权重衰减。
                  这是 nanochat 的生产配置，代码长 3 倍。
    教程卷4 会先把 simple 手写一遍，再讲 advanced 每一步在修什么。
    """
    flavor: str = "simple"
    lr: float = 0.02                # 矩阵参数学习率
    momentum: float = 0.95          # Nesterov 动量
    ns_steps: int = 5               # 正交化迭代步数
    weight_decay: float = 0.0       # 矩阵参数的权重衰减
    # advanced 模式下逐项可关（用于消融）
    use_polar_express: bool = True
    use_muon_eq: bool = True
    use_muon_plus: bool = True
    use_nor_muon: bool = True
    use_cautious_wd: bool = True


@dataclass
class AdamWConfig:
    """嵌入 / lm_head / 标量参数用 AdamW。数值稳定的部分交给它。"""
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    weight_decay_embedding: float = 0.001
    weight_decay_unembedding: float = 0.01
    betas_embedding: tuple = (0.8, 0.995)
    betas_unembedding: tuple = (0.8, 0.96)


@dataclass
class OptimConfig:
    muon: MuonConfig = field(default_factory=MuonConfig)
    adamw: AdamWConfig = field(default_factory=AdamWConfig)
    # LR 随模型维度的缩放：η ∝ 1/√(d_model/768)。768 是调参时用的参考宽度。
    dmodel_lr_ref: int = 768


# ===========================================================================
# 3) 训练配置
# ===========================================================================
@dataclass
class TrainConfig:
    # ---- batch ----
    device_batch_size: int = 24     # 每卡 micro-batch。OOM 就往下调 16/8/4/2/1
    total_batch_size: int = 65536   # 以 token 计的全局 batch。-1 = 自动推导
    # ---- 训练时长（三选一，优先级从上到下）----
    num_iterations: int = -1        # 直接指定步数（-1 = 不用）
    target_flops: float = -1.0      # 指定总 FLOPs，反推步数（-1 = 不用）
    target_param_data_ratio: float = 12.0   # 数据量 / 参数量（Chinchilla=20，nanochat 实测 12）
    # ---- 调度 ----
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65    # 最后 65% 的步数用来降 LR
    final_lr_frac: float = 0.05     # 末尾 LR 降到峰值的 5%
    weight_decay_final_frac: float = 0.0   # 末尾 WD 衰减到这个比例（乘数）
    # ---- 评测与存档 ----
    eval_every: int = 100           # 每 N 步算一次 val bpb（-1 = 关）
    eval_tokens: int = 1 * 524288   # 每次 val 用多少 token
    sample_every: int = 200         # 每 N 步采样一次（-1 = 关）
    save_every: int = -1            # 每 N 步存档（-1 = 只在最后）
    log_every: int = 10
    # ---- 其它 ----
    grad_clip: float = 0.0          # >0 开启梯度裁剪（nanochat 默认不裁）
    seed: int = 42
    model_tag: str | None = None    # 存档目录名，None = 自动用 f"d{depth}"


@dataclass
class RunConfig:
    """一次运行的完整配置。"""
    mode: str
    aspect_ratio: int
    head_dim: int
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    vocab_size: int
    data_shards: int = 1

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "aspect_ratio": self.aspect_ratio,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "model": asdict(self.model),
            "optim_muon": asdict(self.optim.muon),
            "optim_adamw": asdict(self.optim.adamw),
            "train": asdict(self.train),
        }


# ===========================================================================
# 4) 四档预设
# ===========================================================================
# 每一档都刻意选了不同的 aspect_ratio / head_dim，
# 这样你能同时看到「深度怎么影响宽度」和「头数是独立旋钮」。
#
#   档位      depth aspect head_dim  n_embd n_head  用途
#   debug       2     16      32       32     1     单步调试，什么都极小
#   smoke       4     32      32      128     4     2-3 分钟全流程，每章验证用
#   ablation    6     64      64      384     6     消融实验专用（见下）
#   full       24     32      64      768    12     消融后的最佳组合，只训一次
#
# ── 为什么有 ablation 和 full 两档「正式」规模？──────────────────────
#   两者回答的是**不同问题**，所以超参数的取法正好相反：
#
#   ablation (d6)  回答「某个 trick 值多少钱」。
#       基线必须保持**中性**：Muon simple + 5 个 trick 全关。
#       一旦基线本身已经开了 trick，测出来的是「trick A 相对 trick A+B」，
#       单项贡献就被稀释了。规模也要够大到架构差异能穿透初始化噪声 ——
#       debug 档（20 步）实测五组 bpb 完全一致，什么都测不出来。
#
#   full (d24)  回答「最终模型能有多好」。
#       所以它开满 nanochat 的生产配置：5 个残差流 trick 全开 + Muon advanced。
#       只训一次，不做对照，所以不需要中性。
#
# ── full 档的实测数字（RTX 4060 Ti 16GB / d24 / seq1024）────────────
#   参数量       195M（trick 全关）→ 346M（全开，value_embeds 单独 +151M）
#   MFU          23.7%（fwd+bwd，不含优化器）
#
#   ★★ device_batch_size 的实测阶梯 ★★
#   **必须多步测量**。单步测量会严重低估稳态显存需求：优化器状态、
#   梯度累积的中间张量、cudnn workspace 都在第 2 步才达到峰值。
#   下面是连跑 3-4 个完整 step 的结果（每档都是真实训练循环，
#   包含 grad_accum 次 micro-step + 一次 opt.step）：
#
#     dbs   accum  峰值reserved  tok/s   MFU    总耗时   稳定性
#       2     512      4.83 GiB  13,220  21.1%   46.2 h   ✅
#       4     256      8.85 GiB  14,854  23.7%   40.9 h   ✅
#       8     128     12.77 GiB  14,765  23.6%   41.2 h   ✅ ← 默认
#      12      85    14.68 GiB  13,898    —       —      ❌ 第 2 步 CUDA driver error
#      16      —      18.77 GiB      780    —       —      ❌ 超物理显存，溢写 host
#
#   （总步数 2088、目标 token 数 2.19B 在 dbs=4/8 下相同，所以总耗时
#     也几乎相同 —— 这也是为什么 4 和 8 的 tok/s 读数只差 0.6%。）
#
#   **三个反直觉的结论**：
#
#   (1) 加大 batch 对吞吐毫无收益。dbs 从 4 加到 8 是 14,854 → 14,765，
#       严格来说**略降**。也就是说 **GPU 在 dbs=4 时已经吃饱**——
#       MFU 稳定在 23.7% 是这张卡的真实上限，不是 batch 不够大。
#       单步测量曾得出「dbs=8 快 1.44×」的结论，那是错的：那次把
#       opt.step() 的耗时摊到了单步上，而真实的时间分布是
#       fwd 33% / bwd 67% / 优化器 0.6% —— 瓶颈在 fwd+bwd，随 dbs 线性增长。
#
#   (2) dbs=12 会崩，而且**不是 OOM**。它在第 1 步跑完（14.68 GiB），
#       第 2 步抛 `CUDA driver error: device not ready` —— 驱动在显存
#       分配上撞到边界后无法恢复。单步测量时它看着能用（92% 占用），
#       正是最危险的「看着没事」状态。独立进程重试 3 次，每次都复现。
#
#   (3) 所以 4 和 8 的取舍不是速度，是**余量**：吞吐一样，
#       8 用 12.77 GiB（80%），4 用 8.85 GiB（55%）。
#       默认给 8，因为 grad_accum 只有一半（128 vs 256），dataloader 的
#       Python 开销和每步的 CPU 侧开销都摊得更薄，而且 3.2 GiB 余量
#       实测够 SFT 和 eval_sft 用。要更宽裕就 --device-batch-size 4。
#
#   ⚠ 改 device_batch_size 时必须检查 total_batch_size 的整除性（见下方注释）。

PRESETS = {
    "debug": dict(
        tag="d2", depth=2,
        aspect_ratio=16, head_dim=32, sequence_len=128, vocab_size=8192,
        device_batch_size=4, total_batch_size=2048, target_param_data_ratio=1.0,
        num_iterations=20, eval_every=10, sample_every=10, log_every=1,
        data_shards=1,
    ),
    "smoke": dict(
        tag="d4", depth=4,
        aspect_ratio=32, head_dim=32, sequence_len=512, vocab_size=8192,
        device_batch_size=8, total_batch_size=16384, target_param_data_ratio=8.0,
        num_iterations=-1, eval_every=100, sample_every=200, log_every=20,
        data_shards=1,
    ),
    # ── 消融专用：d6，基线保持中性（Muon simple + trick 全关）──────
    "ablation": dict(
        tag="d6", depth=6,
        aspect_ratio=64, head_dim=64, sequence_len=1024, vocab_size=16384,
        # total_batch_size(65536) = 16 × 1024 × 4，所以 device_batch_size
        # 必须是 sequence_len 的倍数关系的因子（否则 resolve_scaling 会 assert）
        device_batch_size=16, total_batch_size=65536, target_param_data_ratio=12.0,
        num_iterations=-1, eval_every=200, sample_every=400, log_every=20,
        data_shards=4,
    ),
    # ── 消融后的最佳组合：d24，只训一次 ────────────────────────────
    # 形状对齐 nanochat 的 d24：24 层 × 768 维 × 12 头，182M scaling 参数。
    # aspect_ratio 必须从 ablation 档的 64 改成 32 —— 沿用 64 会得到 1536 维
    # / 705M 参数，光权重+梯度+Muon 动量就吃掉 10.5 GiB，本卡放不下激活。
    "full": dict(
        tag="d24", depth=24,
        aspect_ratio=32, head_dim=64, sequence_len=1024, vocab_size=16384,
        # total_batch_size 来自推导
        #   B_ref × (target_tokens / D_ref)^0.383 = 2^19 × (2.19B/330M)^0.383
        # 并向上取整到 2 的幂。必须被 device_batch_size×sequence_len 整除。
        # total_batch_size 显式钉死，不留 -1：preset 应该是可复现的快照，
        # 而不是 resolve_scaling 公式的输出。
        device_batch_size=8, total_batch_size=1048576,
        # ⚠ 改 device_batch_size 时必须检查 total_batch_size 的整除性 ——
        #   resolve_scaling 会 assert（total_batch % (dbs × T) == 0）。
        #   dbs=8 -> tokens_per_micro = 8192 = 2^13，而 2^20 = 2^13 × 128，
        #   所以 total_batch_size 不用动（grad_accum = 128）。
        #   注意 dbs=12 时 tokens_per_micro = 12288 = 2^12 × 3，
        #   **任何 2 的幂都不满足** —— 那时 total_batch 只能是 12288 的倍数。
        target_param_data_ratio=12.0,   # → 2,189,426,688 tokens / 2088 步
        num_iterations=-1,
        eval_every=200, sample_every=400, log_every=50,
        # 100+ 小时的 run 必须定期存档。save_every=-1（默认）意味着崩在
        # 第 100 小时就只剩磁盘上的垃圾。
        save_every=200,
        # 2.19B tokens / 每个 shard 约 35M 有效 token → 需要约 64 个 shard
        # 才够训一轮不重复。4 个 shard 会重复约 10 轮，val bpb 偏乐观。
        data_shards=64,
        # ↓ 消融结论：nanochat 生产配置。改这几个开关就能换配置，
        #   不必动上面的形状部分。
        muon_flavor="advanced",
        use_resid_lambdas=True, use_x0_lambdas=True, use_value_embeds=True,
        use_smear=True, use_backout=True,
    ),
}


def default_tag(mode: str) -> str:
    """
    档位 -> 存档目录名。

    单一事实来源。script/_common.sh 里也有一份 TAG 映射（bash 读不到
    Python），两者由 tests/test_presets.py 的 test_common_sh_matches_presets
    做交叉校验 —— 这正是本项目的纪律：可证伪的声明就用可证伪的方式守住。
    """
    assert mode in PRESETS, f"mode 必须是 {list(PRESETS)} 之一，得到 {mode}"
    return str(PRESETS[mode]["tag"])


def default_depth(mode: str) -> int:
    """档位 -> 唯一旋钮 depth 的默认值。"""
    assert mode in PRESETS, f"mode 必须是 {list(PRESETS)} 之一，得到 {mode}"
    return int(PRESETS[mode]["depth"])


def make_run_config(mode: str, depth: int | None = None,
                    vocab_size: int | None = None, **overrides) -> RunConfig:
    """
    组装一份完整配置。

    vocab_size 传 None 时用预设里的期望值（脚本会传 tokenizer 的真实值，
    两者不一致会直接 assert 报错，防止训出来的模型和 tokenizer 对不上）。

    ── 分发机制：preset 的键和调用方的 overrides 走**同一条**路径 ──
      之前只有 overrides 走分发循环，preset 里多写的键会被**静默忽略** ——
      于是「往 preset 里加一个 use_smear=True 却什么都没发生」是完全可能的，
      而且没有任何报错。加 muon_flavor= 之前没人发现，因为那时候 preset
      里的键刚好都显式传给了 ModelConfig / TrainConfig 构造函数。
      现在统一成 settings = {**preset, **overrides} 再一起分发。
    """
    assert mode in PRESETS, f"mode 必须是 {list(PRESETS)} 之一，得到 {mode}"
    p = dict(PRESETS[mode])
    expected_vocab = p.pop("vocab_size")
    shards = p.pop("data_shards")
    p.pop("tag")
    p.pop("depth")          # 已经用来推导 n_layer 了，不参与下面的分发
    aspect = p.pop("aspect_ratio")     # 只用来推宽度，不是 ModelConfig 的字段

    depth = depth if depth is not None else default_depth(mode)
    vocab_size = vocab_size if vocab_size is not None else expected_vocab

    model = build_model_config(
        depth=depth, aspect_ratio=aspect, head_dim=p["head_dim"],
        sequence_len=p["sequence_len"], vocab_size=vocab_size,
    )
    train = TrainConfig()
    optim = OptimConfig()

    for k, v in {**p, **overrides}.items():
        if hasattr(model, k):
            setattr(model, k, v)
        elif hasattr(train, k):
            setattr(train, k, v)
        elif k.startswith("muon_") and hasattr(optim.muon, k[len("muon_"):]):
            setattr(optim.muon, k[len("muon_"):], v)
        elif k.startswith("adamw_") and hasattr(optim.adamw, k[len("adamw_"):]):
            setattr(optim.adamw, k[len("adamw_"):], v)
        else:
            raise ValueError(f"未知的配置项: {k}")

    return RunConfig(
        mode=mode, aspect_ratio=aspect, head_dim=p["head_dim"],
        model=model, optim=optim, train=train,
        vocab_size=vocab_size, data_shards=shards,
    )


# ===========================================================================
# 5) 【单一旋钮】的推导 —— 教程卷5 核心章节
# ===========================================================================
def scaling_params(model: ModelConfig) -> int:
    """
    用于 scaling law 的「参数量」。

    用哪个口径？各家论文不统一：
      Kaplan et al.   只算非嵌入参数
      Chinchilla      算全部参数
    nanochat 的实验结论（见 nanochat/dev/LOG.md, Jan 27 2026）：
      「transformer 矩阵 + lm_head」这个组合能得到最干净的 scaling law。
    所以我们用这个口径。注意它**不含** wte 嵌入表和 value_embeds
    （那些是查找表，不是矩阵乘）。
    """
    matrices = 0
    for block in range(model.n_layer):
        d, kv = model.n_embd, model.n_kv_head * model.head_dim
        matrices += model.n_head * model.head_dim * d   # c_q
        matrices += kv * d                              # c_k
        matrices += kv * d                              # c_v
        matrices += d * d                              # c_proj
        matrices += d * 4 * d                          # mlp c_fc
        matrices += 4 * d * d                          # mlp c_proj
    lm_head = model.n_embd * model.vocab_size
    return matrices + lm_head


def resolve_scaling(run: RunConfig, log=print) -> dict:
    """
    把 depth 推出的所有超参算出来。这是「单一旋钮」的全部魔法。

    输入：depth（经 ModelConfig 变成参数量 P）
    输出：训练多少 token、batch 多大、LR 缩放多少、weight decay 缩放多少

    ── 步骤 1：训练时长 ────────────────────────────────────────────
        目标 token 数 = 数据参数比 × P
        数据参数比 = 12（nanochat 在 ClimbMix 上实测的最优值，
                       注意比 Chinchilla 论文的 20 更激进：
                       现代数据质量更高，不需要那么多重复 token）

    ── 步骤 2：batch size ──────────────────────────────────────────
        Power Lines 论文（arXiv 2505.13738）给出 B_opt ∝ D^0.383
        参考点：d12 模型的最优 batch 约 2^19 = 524,288 token
        （d12 是 nanochat 调参的主战场，再往上靠外推）
        算出后 clamp 到 2 的幂，因为 2 的幂在 GPU 上对齐最友好

    ── 步骤 3：学习率缩放 ──────────────────────────────────────────
        AdamW 的标准结论：η ∝ √(B / B_ref)
        Muon 没有公认结论，nanochat 直接沿用同一规则（注释里明说是假设）

    ── 步骤 4：weight decay 缩放 ──────────────────────────────────
        T_epoch 框架（arXiv 2405.13698）：保持 T_epoch = B/(η·λ·D) 不变
        代入步骤 3 的 η ∝ √(B/B_ref)，解出：
            λ = λ_ref · √(B/B_ref) · (D_ref/D)
        警告：这两篇论文研究的是 AdamW，我们在拿 AdamW 的理论套 Muon，
              属于「盲信」，代码注释里也这么写。这是诚实标注的不确定性。
    """
    m, t = run.model, run.train

    # 参考系固定在 d12 —— 所有缩放规则都相对于它
    d12 = build_model_config(12, run.aspect_ratio, run.head_dim,
                             m.sequence_len, m.vocab_size)
    P = scaling_params(m)
    D_ref = run.train.target_param_data_ratio * scaling_params(d12)
    B_ref = 2 ** 19

    target_tokens = int(t.target_param_data_ratio * P)

    total_batch = t.total_batch_size
    if total_batch == -1:
        predicted = B_ref * (target_tokens / D_ref) ** 0.383
        total_batch = 2 ** round(math.log2(predicted))
        log(f"  自动 batch size: B ∝ D^0.383 -> {total_batch:,} tokens")
    assert total_batch > 0

    batch_lr_scale = 1.0
    if abs(total_batch / B_ref - 1.0) > 1e-9:
        batch_lr_scale = (total_batch / B_ref) ** 0.5
        log(f"  LR 缩放: η ∝ √(B/B_ref) = {batch_lr_scale:.4f}")

    weight_decay_scaled = run.optim.muon.weight_decay * math.sqrt(total_batch / B_ref) * (D_ref / target_tokens)

    # grad accum = 全局 batch / 单卡 micro-batch
    tokens_per_micro = t.device_batch_size * m.sequence_len
    assert total_batch % tokens_per_micro == 0, (
        f"total_batch_size({total_batch}) 必须被 device_batch_size×sequence_len"
        f"({tokens_per_micro}) 整除。把 device_batch_size 调小或 seq 调短。"
    )
    grad_accum = total_batch // tokens_per_micro

    if t.num_iterations > 0:
        num_iterations = t.num_iterations
    elif t.target_flops > 0:
        flops_per_token = estimate_flops_per_token(m)
        num_iterations = round(t.target_flops / (flops_per_token * total_batch))
    else:
        num_iterations = max(1, target_tokens // total_batch)

    return {
        "scaling_params": P,
        "target_tokens": target_tokens,
        "total_batch_size": total_batch,
        "grad_accum_steps": grad_accum,
        "num_iterations": num_iterations,
        "batch_lr_scale": batch_lr_scale,
        "weight_decay_scaled": weight_decay_scaled,
        "D_ref": D_ref, "B_ref": B_ref,
    }


def estimate_flops_per_token(m: ModelConfig) -> int:
    """
    每个 token 的 FLOPs（前向 + 反向）。

    推导（沿用 PaLM 论文附录 B）：
      · 每个参与矩阵乘的参数，前向 2 FLOPs，反向 4 FLOPs，共 6
      · 注意力内部的 QK^T 和 AV 两步，各自 2·h·q·T 每层，共 12·h·q·T

    与 Chinchilla 的精确公式差约 1%，差别在于：
      · Chinchilla 把 embedding 查表算作 FLOPs（这里不算，它只是查表）
      · Chinchilla 把 softmax 的 exp/求和算作 FLOPs（这里不算，可忽略）
    """
    h, q = m.n_head, m.head_dim
    attn = 0
    for window, _ in m.window_sizes():
        eff = m.sequence_len if window < 0 else min(window, m.sequence_len)
        attn += 12 * h * q * eff
    return 6 * scaling_params(m) + attn
