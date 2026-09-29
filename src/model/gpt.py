"""
GPT 主干：把 layers.py 的零件组装起来，加上 nanochat 的残差流 trick。

▓▓ 这一章要你敲的部分 ▓▓
    GPT.__init__            5 个 trick 的开关（卷3）
    GPT.init_weights        初始化策略 + ★ meta device 的坑（卷3 第 16 章）
    GPT.forward             组装（卷3）
    GPT._apply_smear        Smear（卷3 第 19 章）
    build_model             meta device 三步法（卷3 第 16 章）
    sample_from_logits      采样顺序（卷7）

────────────────────────────────────────────────────────────────
卷3 的知识地图
────────────────────────────────────────────────────────────────
    ch16  meta device 三步建模型  -> build_model + init_weights 里的 RoPE 重算
    ch17  resid_lambdas / x0     -> __init__ 的两个 Parameter + forward 里的缩放
    ch18  Value Embeddings       -> __init__ 的 ModuleDict + attention 的 ve_gate
    ch19  Smear / Backout        -> _apply_smear + forward 里的残差扣除
    ch20  滑动窗口注意力          -> window_sizes（只在 config.py 里，这里只消费）
"""

import json
import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import COMPUTE_DTYPE, log0
from model.layers import (
    Linear, rms_norm, precompute_rope, Block, has_value_embed,
)


class GPT(nn.Module):
    def __init__(self, cfg, device=None):
        """
        ⚠ 大坑：__init__ 可能在 meta device 上执行（见 tutorial/卷3 第 16 章）。
        所以这个函数里只能算形状和 dtype，**不能有任何真实数据操作**。
        真正的初始化全部放到 init_weights() 里。
        """
        super().__init__()
        self.config = cfg
        self.window_sizes = cfg.window_sizes()      # 🔶 已给（卷3 第 20 章）

        # 词表按 64 对齐。为什么要对齐？
        #   1) DDP 里梯度是按第一个维度切分的，对齐后各 rank 切出来一样大；
        #   2) GEMM 的 tile 通常是 8/64 的倍数，对齐能吃到更高吞吐。
        # 这是纯优化，不影响模型语义 —— forward 里会把 logits 切回真实词表大小。
        padded = ((cfg.vocab_size + cfg.pad_vocab_size_to - 1) // cfg.pad_vocab_size_to) * cfg.pad_vocab_size_to
        self.padded_vocab_size = padded
        if padded != cfg.vocab_size:
            log0(f"  词表 {cfg.vocab_size} -> {padded}（对齐到 64，纯性能优化）")

        # ❗ 待实现：self.transformer = nn.ModuleDict({...})
        #   · "wte": nn.Embedding(padded, n_embd)
        #   · "h":   nn.ModuleList([Block(cfg, i, device) for i in range(n_layer)])
        # 以及 self.lm_head = Linear(n_embd, padded, bias=False)
        raise NotImplementedError(
            "待实现：GPT.__init__ 的主体 ——\n"
            "  self.transformer = nn.ModuleDict({\n"
            "      'wte': nn.Embedding(padded, cfg.n_embd, device=device),\n"
            "      'h':   nn.ModuleList([Block(cfg, i, device) for i in range(cfg.n_layer)]),\n"
            "  })\n"
            "  self.lm_head = Linear(cfg.n_embd, padded, bias=False, device=device)\n"
            "参考实现：git show solution:src/model/gpt.py")

        # ---- 以下都是残差流 trick，默认全部关闭（卷3 第 17-19 章）----
        # ❗ 待实现：5 个开关，各自只在对应的 cfg 标志为 True 时创建
        #
        # resid_lambdas[i]：第 i 层入口处把残差流整体乘一个可学标量
        #   为什么是「逐层」而不是一个全局标量？
        #   因为不同深度的「信息累积压力」不同：浅层需要更强的缩放来
        #   避免多次累加后爆炸，深层需要更弱来避免过拟合噪声。
        #
        # x0_lambdas[i]：第 i 层入口处把「初始嵌入」按比例加回来
        #   动机：深层的信息被反复非线性变换，可能「忘了」原文。
        #   加一条从初始嵌入的直连，让模型随时能取回原始信号。
        #
        # smear_gate / smear_lambda：把前一个 token 的嵌入混进当前 token
        # backout_lambda：最后 norm 之前减去中层残差
        # value_embeds：给隔层 + 末层提供额外的 token 身份信息
        #
        # 注意它们的类型：
        #   逐层标量 -> nn.Parameter(torch.ones/zeros(n_layer))
        #   value_embeds -> nn.ModuleDict({str(i): nn.Embedding(padded, kv_dim)})
        #     只放 has_value_embed(i, n_layer) 为 True 的层（隔层 + 末层）
        raise NotImplementedError(
            "待实现：GPT.__init__ 的 5 个 trick 开关 ——\n"
            "  if cfg.use_resid_lambdas: self.resid_lambdas = nn.Parameter(torch.ones(n_layer, device=device))\n"
            "  if cfg.use_x0_lambdas:     self.x0_lambdas     = nn.Parameter(torch.zeros(n_layer, device=device))\n"
            "  if cfg.use_smear:\n"
            "      self.smear_gate = Linear(24, 1, bias=False, device=device)\n"
            "      self.smear_lambda = nn.Parameter(torch.zeros(1, device=device))\n"
            "  if cfg.use_backout:  self.backout_lambda = nn.Parameter(0.2*torch.ones(1, device=device))\n"
            "  if cfg.use_value_embeds:\n"
            "      kv_dim = cfg.n_kv_head * cfg.head_dim\n"
            "      self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded, kv_dim, device=device)\n"
            "                                        for i in range(cfg.n_layer) if has_value_embed(i, cfg.n_layer)})\n"
            "参考实现：git show solution:src/model/gpt.py")

        # RoPE 表。超算 10 倍长度：它很小（seq×head_dim/2×2），
        # 多算一点省得以后要动态增长的麻烦。真的不够时 forward 会 assert。
        self.rotary_seq_len = cfg.sequence_len * 10
        cos, sin = precompute_rope(self.rotary_seq_len, cfg.head_dim,
                                   base=cfg.rope_base, device=device,
                                   dtype=COMPUTE_DTYPE)
        # persistent=False：这个 buffer 是常量，不该进 checkpoint
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # =======================================================================
    # ❗ 初始化：单独一个函数，因为要在 to_empty(device) 之后调用
    # =======================================================================
    @torch.no_grad()
    def init_weights(self):
        """
        一次把所有参数初始化完。集中在一个函数里是为了「初始化逻辑」可审计。

        ── 为什么不用 PyTorch 默认的初始化？─────────────────────
        因为默认初始化（Kaiming uniform）是给 CNN 设计的，对 transformer 不合适：
          · 残差流在深层会累积放大，logits 尺度会炸
          · 每个参数都用同样的初始尺度，没有区分「输入侧」和「输出侧」

        我们的策略（沿用 modded-nanogpt / nanochat）：
          · 嵌入 std=0.8       —— 嵌入是残差流的源头，尺度要接近 1
          · lm_head std=0.001  —— 输出侧几乎从零开始，训练初期等于「均匀预测」
          · 投影层 c_proj 全零  —— 每个 block 初始是恒等映射，残差流干净
          · 其余用 Uniform 而非 Normal —— 避免离群值

        ★★ 本文件最重要的一个 bug 在这里 ★★
        __init__ 里算的 cos/sin 是在 **meta device** 上创建的（只有形状）。
        to_empty() 只给它们分配「未初始化的垃圾内存」，不填任何值 ——
        实测会直接变成 NaN，整个模型 loss 立刻是 nan。
        所以凡是「在 __init__ 里创建的、依赖数据的量」，
        都必须在 init_weights 里用真实设备重算一遍。

        ── 你要写的 ──
        1) torch.manual_seed(可复现)
        2) 嵌入/反嵌入：normal_(std=0.8) / normal_(std=0.001)
        3) 每个 block 的 6 个 Linear：
             c_q/c_k/c_v -> uniform_(-s, s)
             mlp.c_fc    -> uniform_(-s*0.4, s*0.4)     0.4 倍
             c_proj（两个）-> zeros_
           其中 s = sqrt(3) * n_embd ** -0.5
           提示：为什么乘 sqrt(3)？Uniform(-s,s) 的标准差是 s/sqrt(3)，
                乘 sqrt(3) 才能得到和 std=s 的 Normal 一样的尺度。
        4) 5 个 trick 的参数分别初始化：
             resid_lambdas -> 逐层递减 1.15 -> 1.05（浅层强、深层弱）
             x0_lambdas    -> 逐层递减 0.20 -> 0.05（浅层更依赖原文）
             smear_lambda  -> zeros_（初始关闭）
             smear_gate    -> uniform_(0, 0.02)
             backout_lambda-> constant_(0.2)
             value_embeds  -> uniform_(-s, s)
             ve_gate       -> uniform_(0, 0.02)
        5) tie_embeddings 时让 wte 和 lm_head 共享权重
        6) ★ 重算 RoPE 表（cos/sin）到真实设备
        """
        raise NotImplementedError(
            "待实现：init_weights —— 见 docstring 的六步\n"
            "\n"
            "  ★ 最容易漏的一步：重算 RoPE\n"
            "    cos, sin = precompute_rope(self.rotary_seq_len, cfg.head_dim,\n"
            "                                 base=cfg.rope_base, device=self.get_device(),\n"
            "                                 dtype=COMPUTE_DTYPE)\n"
            "    self.cos, self.sin = cos, sin\n"
            "  漏了这步 -> cos/sin 是 meta device 的垃圾内存 -> loss 立刻 nan\n"
            "  验证：uv run pytest -k uniform -v（初始 loss 应 ≈ ln(vocab_size)）\n"
            "参考实现：git show solution:src/model/gpt.py")

    # =======================================================================
    # 📖 只读：参数统计（纯诊断代码，理解即可，不用敲）
    # =======================================================================
    def num_matmul_params(self) -> int:
        """
        参与「与 token 流做矩阵乘」的参数量。

        统计方式：凡是本项目自定义的 Linear 实例都算；
        nn.Embedding（查表）和裸标量参数不算 —— 它们没有矩阵乘的 FLOPs。
        这个口径与 common/config.scaling_params 保持一致。
        """
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel()
        return n

    def param_breakdown(self) -> dict:
        """分组的参数量明细。打印出来帮助建立「哪部分占大头」的直觉。"""
        out = {}
        out["wte (token嵌入,查表)"] = self.transformer.wte.weight.numel()
        if hasattr(self, "value_embeds"):
            out["value_embeds"] = sum(p.numel() for p in self.value_embeds.parameters())
        out["lm_head"] = self.lm_head.weight.numel()
        out["transformer矩阵"] = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = 0
        for name in ("resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda"):
            if hasattr(self, name):
                scalars += getattr(self, name).numel()
        if hasattr(self, "smear_gate"):
            scalars += self.smear_gate.weight.numel()
        if scalars:
            out["逐层标量"] = scalars
        out["合计"] = sum(p.numel() for p in self.parameters())
        return out

    def estimate_flops_per_token(self) -> int:
        from common.config import estimate_flops_per_token
        return estimate_flops_per_token(self.config)

    def kv_bytes_per_token(self) -> int:
        """推理时每 token 的 KV cache 占用。GQA 的好处在这里体现。"""
        return (self.config.n_layer * 2 * self.config.n_kv_head
                * self.config.head_dim * COMPUTE_DTYPE.itemsize)

    def get_device(self):
        return self.transformer.wte.weight.device

    def describe(self) -> str:
        """把模型结构打印成一张表。只读代码，理解即可。"""
        cfg = self.config
        lines = [
            "── 模型结构 ──────────────────────────────",
            f"  深度 / 宽度   : {cfg.n_layer} 层 × {cfg.n_embd} 维"
            f"（aspect_ratio={cfg.n_embd // cfg.n_layer}）",
            f"  注意力头       : {cfg.n_head} query / {cfg.n_kv_head} kv"
            f" × head_dim {cfg.head_dim}",
            f"  上下文长度     : {cfg.sequence_len}",
            f"  词表           : {cfg.vocab_size}（padding 后 {self.padded_vocab_size}）",
            f"  位置编码       : {'RoPE base=' + str(int(cfg.rope_base)) if cfg.use_rope else '可学习位置嵌入'}",
            f"  归一化         : {cfg.norm_type}",
            f"  QK-Norm 缩放   : {cfg.qk_norm_scale if cfg.qk_norm_scale > 0 else '关闭'}",
            f"  激活           : {cfg.activation}",
            f"  滑窗模式       : {cfg.window_pattern} -> {self.window_sizes}",
            f"  权重绑定       : {cfg.tie_embeddings}",
            f"  logit softcap  : {cfg.logit_softcap if cfg.logit_softcap > 0 else '关闭'}",
        ]
        tricks = [n for n, on in [
            ("resid_lambdas", cfg.use_resid_lambdas), ("x0_lambdas", cfg.use_x0_lambdas),
            ("value_embeds", cfg.use_value_embeds), ("smear", cfg.use_smear),
            ("backout", cfg.use_backout)] if on]
        lines.append(f"  残差流 trick   : {', '.join(tricks) if tricks else '全部关闭（基础架构）'}")
        lines.append("  参数量明细：")
        for k, v in self.param_breakdown().items():
            lines.append(f"    {k:26s} {v:>12,}")
        mflop = self.estimate_flops_per_token()
        lines.append(f"    {'FLOPs / token':26s} {mflop:>12,}  ({mflop/1e6:.2f} M)")
        lines.append(f"    {'KV cache / token':26s} {self.kv_bytes_per_token():>12,} bytes")
        lines.append("──────────────────────────────────────────")
        return "\n".join(lines)

    # =======================================================================
    # ❗ forward：组装（卷3）
    # =======================================================================
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        """
        idx      (B, T)  int64，token id
        targets  (B, T)  int64，-1 表示「这个位置不参与 loss」
        返回：有 targets -> loss；无 targets -> logits (B, T, vocab_size)

        ── 六个步骤 ──────────────────────────────────────────
        1) 查表得到初始嵌入，然后**立刻归一化**
           为什么？固定住残差流的起点尺度。否则后面的层数一多，
           尺度会随机漂移，初始化完全不可控。

        2) Smear（如果有）：把前一个 token 的嵌入混进来 -> _apply_smear

        3) 逐层前向。每层入口处按开关做缩放：
             x = resid_lambdas[i] * x
             x = x + x0_lambdas[i] * x0            （x0 是第 1 步的初始嵌入）
           然后取 value_embeds[str(i)](idx)（如果有这一层）
           再 block(x, ...)

        4) Backout（如果有）：在最后一层之前 cache 住中层残差，
           最后减去 backout_lambda * x_backout

        5) 最后再归一化，然后 lm_head

        6) 切掉 padding 部分 -> 转 fp32 -> logit softcap
             softcap: cap * tanh(logits / cap)
             为什么需要？训练久了某些 token 的 logit 会涨到 30+，
             指数函数在那个区间已经饱和，梯度趋近 0，
             且 fp32 也开始损失精度。tanh softcap 相当于给 logits 加软上限。

        ── 两个必须做对的细节 ──────────────────────────────
        (a) reduction='none' 时 F.cross_entropy 返回扁平的 (B*T,)，
            要 reshape 回 (B, T) —— 评测和 RL 都依赖这个形状。
        (b) reduction='mean' 且整批 target 都被 ignore（都是 -1）时，
            F.cross_entropy 算出 0/0 = NaN。必须降级成 0 并告警，
            否则 NaN 会污染全部参数且不可恢复。
            （这在 SFT 里真的发生过：truncate 策略把监督信号全切掉了，
              512 长度下 23% 的样本会踩中。见 tutorial/卷1 第 04 章）
        """
        raise NotImplementedError(
            "待实现：GPT.forward —— 见 docstring 的六步\n"
            "  关键提示：\n"
            "  · T0 = 0 if kv_cache is None else kv_cache.get_pos()\n"
            "    cos_sin = (self.cos[:, T0:T0+T], self.sin[:, T0:T0+T])\n"
            "  · x = rms_norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))\n"
            "  · hasattr(self, 'smear_lambda') 时调 _apply_smear\n"
            "  · 循环里 hasattr(self,'resid_lambdas') / hasattr(self,'x0_lambdas') / value_embeds\n"
            "  · backout_layer = n_layer // 2，在该层后 cache x\n"
            "  · logits = self.lm_head(rms_norm(x))[..., :cfg.vocab_size].float()\n"
            "  · if cfg.logit_softcap > 0: logits = cap * tanh(logits / cap)\n"
            "  · loss = F.cross_entropy(..., reduction=loss_reduction)\n"
            "    reduction=='none' -> loss.view(B, T)\n"
            "    reduction=='mean' 且非有限 -> 降级成 0 并 log0 告警\n"
            "验证：uv run pytest -k \"uniform or loss_reduction or ignore or causal\" -v")

    # =======================================================================
    # ❗ Smear（卷3 第 19 章）
    # =======================================================================
    def _apply_smear(self, x, kv_cache):
        """
        Smear：x[i] = x[i] + lambda * sigmoid(gate) * x[i-1]

        ── 动机 ────────────────────────────────────────────
        只从 token 身份学到的嵌入里，没有任何「上一个词是什么」的信息。
        注意力理论上能自己学出来，但在浅层很难。
        把前一个 token 的嵌入直接加过来，等于免费给了模型一个 bigram 线索。

        ── 你要写的：两条路径 ────────────────────────────────
        (1) kv_cache is None（训练 / 朴素生成）
            整条序列都在手上，直接切片：
              gate = smear_lambda * sigmoid(smear_gate(x[:, 1:, :24]))
              return cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            提示：位置 0 没有「前一个 token」，所以原样保留。

        (2) kv_cache 不为 None（推理）
            单个 token 时要靠 cache 记住上一步的嵌入：
              prev = kv_cache.prev_embedding
              kv_cache.prev_embedding = x[:, -1:, :]     ← 存下当前，供下一步用
              T == 1 且 prev 不为 None -> x = x + gate * prev
              T > 1（prefill）      -> 和训练一样的切片逻辑

        为什么要存 prev_embedding？
            decode 时每次只喂一个 token，模型看不到前一个 token 的嵌入
            （它的 k/v 在 cache 里，但**嵌入**不在）。Smear 需要它。
        """
        raise NotImplementedError(
            "待实现：_apply_smear —— 见 docstring 的两条路径\n"
            "  注意 prefill (T>1) 和 decode (T==1) 要分开处理\n"
            "参考实现：git show solution:src/model/gpt.py")

    # =======================================================================
    # ❗ 采样（卷7（只读））
    # =======================================================================
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        朴素自回归：每生成一个 token 就把整条序列重新前向一次。
        O(L^2) 的重复计算。存在的意义是让你看清「KV cache 到底省了什么」
        —— inference/engine.py 会做同样的事但复用缓存。
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

    # =======================================================================
    # ❗ sample_from_logits（卷7）
    # =======================================================================
def sample_from_logits(logits, rng, temperature=1.0, top_k=None):
    """
    从 logits 里采一个 token。

    ── 采样顺序很讲究 ──────────────────────────────────────
        1) temperature == 0  -> 直接 argmax（贪心，完全确定）
        2) top_k 裁剪        -> 只保留概率最高的 k 个，其余置 -inf
        3) 除以 temperature
        4) softmax
        5) multinomial

    为什么 top_k 要在除温度之前？
      温度是对 logits 做缩放，它不改变排序。先除温度再 top_k，
      被裁掉的位置依然是 -inf，结果其实一样 ——
      但先 top_k 可以保证「只有 k 个数参与后续计算」，
      在 vocab 很大时是实打实的性能差异。

    ── 你要写的 ──
    ⚠ 一个真实的 bug：`logits.topk(k, dim=-1)` 返回的是 **namedtuple**，
      要用 `.values` / `.indices` 访问。
      曾经写成 `v, _ = logits.topk(...)` 然后用 `v.indices`，
      结果 v 是 Tensor，`.indices` 报 TypeError。

    验证：uv run pytest -k "sampling or topk" -v
    """
    raise NotImplementedError(
        "待实现：sample_from_logits ——\n"
        "  if temperature == 0.0: return logits.argmax(dim=-1, keepdim=True)\n"
        "  if top_k is not None and top_k > 0:\n"
        "      top = logits.topk(min(top_k, logits.size(-1)), dim=-1)   # namedtuple!\n"
        "      logits = torch.full_like(logits, float('-inf'))\n"
        "      logits.scatter_(-1, top.indices, top.values)\n"
        "  probs = F.softmax(logits / temperature, dim=-1)\n"
        "  return torch.multinomial(probs, num_samples=1, generator=rng)")


# ===========================================================================
# ❗ meta device 三步法（卷3 第 16 章）
# ===========================================================================
def build_model(cfg, device="cuda") -> GPT:
    """
    【meta device 三步法】建模型。

    为什么不用普通的 GPT(cfg) 然后 .to(device)？
      因为普通初始化会在**源设备**（CPU）上分配并写入所有随机数，
      再整体搬到 GPU。对 16 GB 的模型，这意味着：
        · CPU 上要先生成 ~16 GB 的随机数（慢）
        · 再走 PCIe 搬 16 GB（更慢）
        · 峰值内存 = CPU 16 GB + GPU 16 GB

    三步法完全跳过这两步：
        1) with torch.device("meta"): 只算形状，不分配任何内存，不产生随机数
        2) model.to_empty(device)  : 直接在目标设备上分配（内容是垃圾数据）
        3) model.init_weights()    : 在 GPU 上原地初始化

    峰值内存从 (CPU 16G + GPU 16G) 降到 (GPU 16G)，而且快得多。

    ── 你要写的 ──
    提示：torch.device("meta") 可以用作 context manager，
    在里面创建的任何张量都只在 meta 上有个「形状壳」。

    ⚠ 三步法的代价：init_weights 必须负责重算**所有**
      「在 __init__ 里创建的、依赖数据的量」。本项目里就是 cos/sin。
      漏了 -> NaN。这是 meta device 最常见的坑。
    """
    raise NotImplementedError(
        "待实现：build_model ——\n"
        "  with torch.device('meta'):\n"
        "      model = GPT(cfg)\n"
        "  model.to_empty(device=device)\n"
        "  model.init_weights()\n"
        "  return model\n"
        "验证：uv run pytest -k uniform -v（loss 应 ≈ ln(vocab_size)，不是 nan）")
