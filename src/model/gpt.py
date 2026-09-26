"""
GPT 主干：把 layers.py 的零件组装起来，加上 nanochat 的残差流 trick。

这个文件对应教程卷3 的全部内容。重点：
  · 用 meta device 三步建模型（避免随机初始化浪费显存带宽）
  · 5 个残差流 trick：resid_lambdas / x0_lambdas / value_embeds / smear / backout
  · forward 的两条路径：训练返回 loss，推理返回 logits
  · logit softcap
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
        ⚠ 大坑：__init__ 可能在 meta device 上执行（见 tutorial/卷3 第一章）。
        所以这个函数里只能算形状和 dtype，**不能有任何真实数据操作**。
        真正的初始化全部放到 init_weights() 里。
        """
        super().__init__()
        self.config = cfg
        self.window_sizes = cfg.window_sizes()

        # 词表按 64 对齐。为什么要对齐？
        #   1) DDP 里梯度是按第一个维度切分的，对齐后各 rank 切出来一样大；
        #   2) GEMM 的 tile 通常是 8/64 的倍数，对齐能吃到更高吞吐。
        # 这是纯优化，不影响模型语义 —— forward 里会把 logits 切回真实词表大小。
        padded = ((cfg.vocab_size + cfg.pad_vocab_size_to - 1) // cfg.pad_vocab_size_to) * cfg.pad_vocab_size_to
        self.padded_vocab_size = padded
        if padded != cfg.vocab_size:
            log0(f"  词表 {cfg.vocab_size} -> {padded}（对齐到 64，纯性能优化）")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded, cfg.n_embd, device=device),
            "h": nn.ModuleList([Block(cfg, i, device) for i in range(cfg.n_layer)]),
        })
        self.lm_head = Linear(cfg.n_embd, padded, bias=False, device=device)

        # ── 以下都是残差流 trick，默认全部关闭 ──────────────────
        # resid_lambdas[i]：第 i 层入口处把残差流整体乘一个可学标量
        if cfg.use_resid_lambdas:
            self.resid_lambdas = nn.Parameter(torch.ones(cfg.n_layer, device=device))
        # x0_lambdas[i]：第 i 层入口处把「初始嵌入」按比例加回来
        if cfg.use_x0_lambdas:
            self.x0_lambdas = nn.Parameter(torch.zeros(cfg.n_layer, device=device))
        # Smear：把前一个 token 的嵌入按门控混进当前 token
        if cfg.use_smear:
            self.smear_gate = Linear(24, 1, bias=False, device=device)
            self.smear_lambda = nn.Parameter(torch.zeros(1, device=device))
        # Backout：最后 norm 之前，减去中层残差以抹掉低层特征
        if cfg.use_backout:
            self.backout_lambda = nn.Parameter(0.2 * torch.ones(1, device=device))
        # Value Embeddings：给隔层 + 末层的 attention 额外提供 token 身份信息
        if cfg.use_value_embeds:
            kv_dim = cfg.n_kv_head * cfg.head_dim
            self.value_embeds = nn.ModuleDict({
                str(i): nn.Embedding(padded, kv_dim, device=device)
                for i in range(cfg.n_layer) if has_value_embed(i, cfg.n_layer)
            })

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
    # 初始化：单独一个函数，因为要在 to_empty(device) 之后调用
    # =======================================================================
    @torch.no_grad()
    def init_weights(self):
        """
        一次把所有参数初始化完。集中在一个函数里，是为了「初始化逻辑」可审计。

        ── 为什么不用 PyTorch 默认的初始化？─────────────────────
        因为默认初始化（Kaiming uniform）是给 CNN 设计的，对 transformer 不合适：
          · 残差流在深层会累积放大，logits 尺度会炸
          · 每个参数都用同样的初始尺度，没有区分「输入侧」和「输出侧」

        我们的策略（沿用 modded-nanogpt / nanochat）：
          · 嵌入 std=0.8       —— 嵌入是残差流的源头，尺度要接近 1
          · lm_head std=0.001  —— 输出侧几乎从零开始，训练初期等于「均匀预测」
          · 投影层 c_proj 全零  —— 每个 block 初始是恒等映射，残差流干净
          · 其余用 Uniform 而非 Normal —— 避免离群值
        """
        cfg = self.config
        torch.manual_seed(cfg.n_layer * 1000 + cfg.n_embd)  # 形状相同的模型给同一种子

        # 嵌入与反嵌入
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Uniform(-s, s) 的标准差是 s/√3，所以要乘 √3 才能和 std=s 的 Normal 一样
        s = math.sqrt(3) * cfg.n_embd ** -0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)      # 输出投影 = 0 -> 恒等
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4 倍
            torch.nn.init.zeros_(block.mlp.c_proj.weight)      # 输出投影 = 0 -> 恒等

        if cfg.use_resid_lambdas:
            # 深层残差缩放小一点：避免信息在深层过度累积
            for i in range(cfg.n_layer):
                self.resid_lambdas.data[i] = 1.15 - 0.10 * i / max(cfg.n_layer - 1, 1)
        if cfg.use_x0_lambdas:
            # 浅层更依赖原始嵌入，深层更少
            for i in range(cfg.n_layer):
                self.x0_lambdas.data[i] = 0.20 - 0.15 * i / max(cfg.n_layer - 1, 1)
        if cfg.use_smear:
            torch.nn.init.zeros_(self.smear_lambda)      # 初始关闭：门控 sigmoid(0)=0.5 但 lambda=0
            torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        if cfg.use_backout:
            torch.nn.init.constant_(self.backout_lambda, 0.2)
        if cfg.use_value_embeds:
            for ve in self.value_embeds.values():
                torch.nn.init.uniform_(ve.weight, -s, s)
            for block in self.transformer.h:
                if block.attn.ve_gate is not None:
                    torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        if cfg.tie_embeddings:
            # 权重绑定：wte 和 lm_head 共享同一份参数
            self.transformer.wte.weight = self.lm_head.weight

        # ── 必须在这里重算 RoPE 表 ─────────────────────────────
        # __init__ 里算的 cos/sin 是在 **meta device** 上创建的（只有形状）。
        # to_empty() 只给它们分配「未初始化的垃圾内存」，不填任何值 ——
        # 实测会直接变成 NaN，整个模型 loss 立刻是 nan。
        # 这就是 meta device 三步法最容易踩的坑：
        #   凡是「在 __init__ 里创建的、依赖数据的量」，都必须在 init_weights 里重算。
        cos, sin = precompute_rope(
            self.rotary_seq_len, cfg.head_dim, base=cfg.rope_base,
            device=self.get_device(), dtype=COMPUTE_DTYPE)
        self.cos, self.sin = cos, sin

    # =======================================================================
    # 参数统计
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

    # =======================================================================
    # forward
    # =======================================================================
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        """
        idx      (B, T)  int64，token id
        targets  (B, T)  int64，-1 表示「这个位置不参与 loss」
        返回：有 targets -> loss；无 targets -> logits (B, T, vocab_size)
        """
        cfg = self.config
        B, T = idx.shape
        assert T <= self.cos.size(1), (
            f"序列长度超过 RoPE 表容量：{T} > {self.cos.size(1)}"
        )

        # KV cache 场景下要从「当前位置」开始取 RoPE
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T])

        # (1) 查表得到初始嵌入
        x = self.transformer.wte(idx)
        x = rms_norm(x.to(COMPUTE_DTYPE))   # 嵌入后立刻归一化，固定住残差流的起点尺度

        # (2) Smear：把前一个 token 的嵌入混进来
        if hasattr(self, "smear_lambda"):
            x = self._apply_smear(x, kv_cache)

        # (3) 逐层前向
        x0 = x
        backout_layer = cfg.n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            if hasattr(self, "resid_lambdas"):
                x = self.resid_lambdas[i] * x
            if hasattr(self, "x0_lambdas"):
                x = x + self.x0_lambdas[i] * x0
            ve = (self.value_embeds[str(i)](idx).to(x.dtype)
                  if hasattr(self, "value_embeds") and str(i) in self.value_embeds else None)
            x = block(x, cos_sin, self.window_sizes[i], kv_cache, ve)
            if i == backout_layer:
                x_backout = x

        # (4) Backout：减掉中层残差
        if x_backout is not None and hasattr(self, "backout_lambda"):
            x = x - self.backout_lambda.to(x.dtype) * x_backout

        x = rms_norm(x)

        # (5) 输出头
        logits = self.lm_head(x)                      # (B, T, padded_vocab)
        logits = logits[..., :cfg.vocab_size]          # 切掉 padding 部分
        logits = logits.float()                        # 后面要算 CE，fp32 更稳

        # (6) Logit softcap：平滑地把 logits 压到 [-15, 15]
        #     为什么需要？训练久了某些 token 的 logit 会涨到 30+，
        #     指数函数在那个区间已经饱和，梯度趋近 0，且 fp32 也开始损失精度。
        #     tanh softcap 相当于给 logits 加了个软上限。
        if cfg.logit_softcap > 0:
            cap = cfg.logit_softcap
            logits = cap * torch.tanh(logits / cap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction)
            # reduction='none' 时 F.cross_entropy 返回扁平的 (B*T,)，
            # 这里 reshape 回 (B, T) 让调用方（评测、RL）能按位置处理。
            if loss_reduction == "none":
                loss = loss.view(B, T)
            elif loss_reduction == "mean" and not torch.isfinite(loss):
                # 整批 target 都被 ignore（都是 -1）时，F.cross_entropy 算的是 0/0 = NaN。
                # 这通常意味着上游的 mask 逻辑出了 bug（例如截断把监督信号全切掉了）。
                # 这里降级成 0 而不是让 NaN 污染全部权重 —— 后者不可恢复。
                log0("  [警告] 本批没有有效的监督 token（targets 全为 -1），loss 记为 0。"
                     "通常是截断策略把 assistant 部分切掉了，检查 truncate 参数。")
                loss = logits.sum() * 0.0
            return loss
        return logits

    def _apply_smear(self, x, kv_cache):
        """
        Smear：x[i] = x[i] + lambda * sigmoid(gate) * x[i-1]

        动机：只从 token 身份学到的嵌入里没有任何「上一个词是什么」的信息。
        注意力理论上能自己学出来，但在浅层很难。把前一个 token 的嵌入
        直接加过来，等于免费给了模型一个 bigram 线索。
        """
        B, T, _ = x.shape
        Tq = x.size(1)
        if kv_cache is None:
            assert T > 1, "训练时序列长度必须 > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            return torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        # 推理：单个 token 时要靠 cache 记住上一步的嵌入
        prev = kv_cache.prev_embedding
        kv_cache.prev_embedding = x[:, -1:, :]
        if Tq > 1:  # prefill，和训练一致
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            return torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        if prev is not None:  # decode
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
            return x + gate * prev
        return x

    # =======================================================================
    # 推理（朴素版；带 KV cache 的高效版在 inference/engine.py）
    # =======================================================================
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

    def get_device(self):
        return self.transformer.wte.weight.device

    def describe(self) -> str:
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
            f"  滑窗模式       : {cfg.window_pattern}"
            f" -> {self.window_sizes}",
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


def sample_from_logits(logits, rng, temperature=1.0, top_k=None):
    """
    从 logits 里采一个 token。

    顺序很讲究：**先 top-k 裁剪，再除温度，再 softmax**。
    如果先除温度再裁剪，被裁掉的 token 仍然参与了温度缩放，
    边界位置会有微小的数值差异。教程卷7 会详细讲这个顺序。
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        # 注意：topk 返回的是 namedtuple，要用 .values / .indices 访问，
        # 不能解包成 (v, _) 然后访问 v.indices —— 那是 Tensor 不是 namedtuple
        top = logits.topk(k, dim=-1)
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, top.indices, top.values)   # 只保留 top-k 的原始值
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


def build_model(cfg, device="cuda") -> GPT:
    """
    【meta device 三步法】建模型。

    为什么不用普通的 GPT(cfg) 然后 .to(device)？
      因为普通初始化会在**源设备**（CPU）上分配并写入所有随机数，
      再整体搬到 GPU。对 16 GB 的模型，这意味着：
        · CPU 上要先生成 ~16 GB 的随机数（慢）
        · 再走 PCIe 搬 16 GB（更慢）
        · 峰值内存 = CPU 16 GB + GPU 16 GB

    meta device 三步法完全跳过这两步：
        1) with torch.device("meta"): 只算形状，不分配任何内存，不产生随机数
        2) model.to_empty(device)  : 直接在目标设备上分配（内容是垃圾数据）
        3) model.init_weights()    : 在 GPU 上原地初始化

    峰值内存从 (CPU 16G + GPU 16G) 降到 (GPU 16G)，而且快得多。
    """
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    return model
