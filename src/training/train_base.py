"""
预训练入口。

跑法（推荐用 shell 脚本，它会串好前置步骤）：
    bash script/train_base.sh smoke
    uv run python -m training.train_base --mode smoke

这个文件对应教程卷5：训练循环、梯度累积、三条调度曲线、MFU、checkpoint。
"""

import os
import gc
import json
import time
import math
import argparse
from dataclasses import asdict

import torch

# torch.compile 的 inductor 在没有 C 编译器时会打一串很吵的警告。
# 本项目对此做了优雅降级（见 optim/muon.py:compile_or_eager），所以直接静音。
import logging
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

from common import (
    autodetect_device_type, compute_init, compute_cleanup, log0, logger,
    get_runs_dir, get_peak_flops, synchronize, get_max_memory, human_time,
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON,
)
from common.config import make_run_config, resolve_scaling
from data.tokenizer import get_tokenizer, get_token_bytes
from data.dataloader import make_dataloader
from model.gpt import build_model
from optim.muon import setup_optimizer
from evaluation.metrics import evaluate_bpb
from common.checkpoint import save_checkpoint, load_checkpoint, find_latest


# ===========================================================================
# 命令行
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="预训练 base 模型")
    p.add_argument("--mode", default="smoke", choices=["debug", "smoke", "full"],
                   help="档位：debug=单步调试 / smoke=2-3分钟 / full=20-25分钟")
    p.add_argument("--depth", type=int, default=None, help="覆盖档位默认深度（唯一旋钮）")
    # 消融开关：直接覆盖 ModelConfig 的字段
    for flag, typ, help_ in [
        ("norm-type", str, "rms|layer"),
        ("activation", str, "relu2|gelu"),
        ("window-pattern", str, "L|SSL|SSSL"),
    ]:
        p.add_argument(f"--{flag}", type=typ, default=None, help=help_)
    for flag, help_ in [
        ("no-rope", "关掉 RoPE（退回无位置编码）"),
        ("no-qk-norm", "关掉 QK Norm"),
        ("tie-embeddings", "打开权重绑定"),
        ("no-softcap", "关掉 logit softcap"),
        ("muon-advanced", "Muon 用 advanced 版（Polar Express + MuonEq + Muon+ + NorMuon）"),
        ("use-resid-lambdas", "开启 resid_lambdas"),
        ("use-x0-lambdas", "开启 x0_lambdas"),
        ("use-value-embeds", "开启 Value Embeddings"),
        ("use-smear", "开启 Smear"),
        ("use-backout", "开启 Backout"),
        ("all-tricks", "开启全部 5 个残差流 trick"),
    ]:
        p.add_argument(f"--{flag}", action="store_true", help=help_)
    p.add_argument("--num-iterations", type=int, default=None, help="覆盖步数")
    p.add_argument("--device-batch-size", type=int, default=None, help="覆盖 micro-batch")
    p.add_argument("--target-param-data-ratio", type=float, default=None,
                   help="覆盖数据参数比（12=nanochat默认，8=欠训练更快）")
    p.add_argument("--model-tag", type=str, default=None, help="存档目录名")
    p.add_argument("--save-every", type=int, default=None, help="每 N 步存档")
    p.add_argument("--no-resume", action="store_true",
                   help="不要从已有 checkpoint 续训，从头开始。"
                        "做消融实验时必须加：否则换个 tag 重跑同名实验时，"
                        "auto-resume 会把上次的存档捡起来，0 步就跑完了，"
                        "两组 bpb 会一模一样。")
    return p.parse_args()


def apply_overrides(cfg, args) -> None:
    """把命令行上的消融开关写进 ModelConfig。"""
    m = cfg.model
    if args.norm_type:       m.norm_type = args.norm_type
    if args.activation:      m.activation = args.activation
    if args.window_pattern:  m.window_pattern = args.window_pattern
    if args.no_rope:         m.use_rope = False
    if args.no_qk_norm:      m.qk_norm_scale = 0.0
    if args.tie_embeddings:  m.tie_embeddings = True
    if args.no_softcap:      m.logit_softcap = 0.0
    if args.all_tricks:
        m.use_resid_lambdas = m.use_x0_lambdas = True
        m.use_value_embeds = m.use_smear = m.use_backout = True
    for name in ("resid_lambdas", "x0_lambdas", "value_embeds", "smear", "backout"):
        if getattr(args, f"use_{name}"):
            setattr(m, f"use_{name}", True)
    if args.muon_advanced:
        cfg.optim.muon.flavor = "advanced"
    if args.num_iterations is not None:     cfg.train.num_iterations = args.num_iterations
    if args.device_batch_size is not None:  cfg.train.device_batch_size = args.device_batch_size
    if args.target_param_data_ratio is not None:
        cfg.train.target_param_data_ratio = args.target_param_data_ratio
    if args.save_every is not None:         cfg.train.save_every = args.save_every
    if args.model_tag:                      cfg.train.model_tag = args.model_tag


# ===========================================================================
# 三条调度曲线（教程卷5 第 31 章）
# ===========================================================================
def make_schedulers(train_cfg, num_iterations: int, weight_decay_final: float):
    """
    返回三个调度函数：

    lr_mult(step)          学习率乘数
    muon_momentum(step)    Muon 的 Nesterov 动量
    weight_decay(step)     Muon 的权重衰减

    ── 形状：warmup -> 平台 -> warmdown ────────────────────────
    学习率曲线（llama / nanochat 都用这个三段式）：

        1.0 ┤      ╭──────────────╮
            │     ╱                ╲
        0.05┤────╯                  ╰────
            └────┬──────────────────────┬────
               warmup              warmdown
               (40 步)              (65% of iters)

    为什么要 warmup？
      训练最开始参数接近随机（我们的初始化里 c_proj 全零、lm_head 极小），
      此时梯度方向噪声极大，AdamW 的一阶/二阶矩还没建立。
      直接用大学习率会把随机结构烧掉再也长不回来。
      40 步（几百个 batch）足够让矩估计稳定。

    为什么要 warmdown 而不是余弦衰减到 0？
      实测（nanochat/dev/LOG.md）三段式比余弦好。
      关键差别：余弦在前 80% 的时间里 LR 一直在下降，
      相当于大部分训练都没跑在最佳 LR 上；
      三段式有 35% 的时间满速跑。
    """

    warmup = train_cfg.warmup_steps
    warmdown = round(train_cfg.warmup_steps * 0) + round(train_cfg.warmdown_ratio * num_iterations)
    final_frac = train_cfg.final_lr_frac

    def lr_mult(step: int) -> float:
        if step < warmup:
            # 线性从 0 爬到 1。用 (step+1)/warmup 保证第一步不为 0，
            # 否则第一步的更新量是 0，等于白跑
            return (step + 1) / warmup
        if step <= num_iterations - warmdown:
            return 1.0
        progress = (num_iterations - step) / max(warmdown, 1)
        return progress + (1 - progress) * final_frac

    def muon_momentum(step: int) -> float:
        # 动量也在 warmup：早期动量缓冲里的噪声应该被压制
        if step < 400:
            frac = step / 400
            return (1 - frac) * 0.85 + frac * 0.97
        if step >= num_iterations - warmdown:
            # 训练末期降动量：让最后几步更接近纯 SGD，更「稳」
            progress = (step - (num_iterations - warmdown)) / max(warmdown, 1)
            return 0.97 * (1 - progress) + 0.90 * progress
        return 0.97

    def weight_decay(step: int) -> float:
        # 余弦衰减到 0（乘以 weight_decay_final 作为终点比例）
        cos = 0.5 * (1 + math.cos(math.pi * min(step, num_iterations) / num_iterations))
        return cos * weight_decay_final

    return lr_mult, muon_momentum, weight_decay


# ===========================================================================
# 训练
# ===========================================================================
def main():
    args = parse_args()
    device_type = autodetect_device_type()
    ddp, rank, local_rank, world_size, device = compute_init(device_type)
    sync = synchronize(device_type)
    log0(f"设备 {device} | COMPUTE_DTYPE {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    # ---- 1) 配置：单一旋钮推导 ----
    tokenizer = get_tokenizer()
    cfg = make_run_config(args.mode, depth=args.depth, vocab_size=tokenizer.get_vocab_size())
    apply_overrides(cfg, args)
    log0(f"\n【配置】{args.mode} 档")
    log0("── 单一旋钮推导 " + "─" * 45)
    sc = resolve_scaling(cfg, log=log0)
    num_iterations = sc["num_iterations"]
    log0(f"  深度 depth      = {cfg.model.n_layer}（唯一旋钮）")
    log0(f"  宽度 n_embd     = {cfg.model.n_embd}  头数 {cfg.model.n_head}×{cfg.model.head_dim}")
    log0(f"  scaling 参数量  = {sc['scaling_params']:,}")
    log0(f"  训练 token 数   = {sc['target_tokens']:,}")
    log0(f"  全局 batch      = {sc['total_batch_size']:,} tokens"
         f" = {sc['grad_accum_steps']} × {cfg.train.device_batch_size}×{cfg.model.sequence_len}")
    log0(f"  训练步数        = {num_iterations:,}")

    # ---- 2) 模型（meta device 三步法）----
    log0("── 模型 " + "─" * 58)
    model = build_model(cfg.model, device=device)
    log0(model.describe())

    tag = cfg.train.model_tag or f"d{cfg.model.n_layer}"
    ckpt_dir = os.path.join(get_runs_dir(), "base_checkpoints", tag)
    start_step = 0
    # 续训要恢复的三样东西。默认给空/初始值，由下面的续训块覆盖。
    resume_dl_state = None
    best_bpb = float("inf")
    history = []
    if os.path.isdir(ckpt_dir) and not args.no_resume:
        last = find_latest(ckpt_dir)
        if last is not None:
            log0(f"发现已有存档 step={last}，自动续训"
                 f"（想从头跑请加 --no-resume）")
            model.load_state_dict(load_checkpoint(ckpt_dir, last, device, "model"))
            # ★ 光恢复权重是不够的。meta 里另外三样也必须读回来，
            #   否则「续训」名不副实：
            #     dataloader_state  数据流位置。不恢复 -> 从头重放已训过的数据，
            #                       而 make_dataloader 的 resume_state 参数
            #                       （dataloader.py 里认真实现了 pq_idx/rg_idx/epoch）
            #                       就永远没有调用方。
            #     best_val_bpb      历史最好指标。不恢复 -> 重置为 inf，
            #                       「历史最好」只统计续训之后，还会把错误值写回新存档。
            #     history           训练曲线。不恢复 -> 曲线图只剩续训之后那一段。
            # 注意：优化器状态是刻意不入 checkpoint 的（见 optim/muon.py 的说明），
            #       续训会丢动量、头几十步有抖动，那是本项目有意的简化，不在这里处理。
            meta = load_checkpoint(ckpt_dir, last, "cpu", "meta")
            resume_dl_state = meta.get("dataloader_state")
            # best_val_bpb 存盘时若还是 inf 会被写成 JSON null（见
            # checkpoint._json_safe），这里把 None 还原成 inf。
            _bpb = meta.get("best_val_bpb")
            best_bpb = float("inf") if _bpb is None else float(_bpb)
            history = meta.get("history", [])
            log0(f"  已恢复：dataloader_state={resume_dl_state} | "
                 f"best_val_bpb={best_bpb} | history {len(history)} 条")
            start_step = last + 1
    elif args.no_resume:
        log0("--no-resume：从头开始训练")

    # ---- 3) 优化器 ----
    log0("── 优化器 " + "─" * 56)
    log0(f"  Muon flavor = {cfg.optim.muon.flavor}"
         + ("（Polar Express + MuonEq + Muon+ + NorMuon + 谨慎WD）" if cfg.optim.muon.flavor == "advanced" else "（5 步 Newton-Schulz）"))
    optimizer = setup_optimizer(model, cfg.optim, batch_lr_scale=sc["batch_lr_scale"])
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    # ---- 4) 数据 ----
    log0("── 数据 " + "─" * 58)
    # resume_state 传进去，dataloader 才会从存档里的位置继续，
    # 而不是把前 start_step 步的数据重喂一遍。
    train_loader = make_dataloader(tokenizer, cfg.train.device_batch_size,
                                   cfg.model.sequence_len, "train", device,
                                   resume_state=resume_dl_state)
    token_bytes = get_token_bytes(device)
    x, y, dl_state = next(train_loader)
    log0(f"  device_batch_size={cfg.train.device_batch_size}  seq_len={cfg.model.sequence_len}")
    log0(f"  首个 batch: {tuple(x.shape)}  BOS={tokenizer.get_bos_token_id()}")

    # ---- 5) 调度器 ----
    lr_mult, mom_sched, wd_sched = make_schedulers(cfg.train, num_iterations, sc["weight_decay_scaled"])
    flops_per_token = model.estimate_flops_per_token()
    peak_flops = get_peak_flops(torch.cuda.get_device_name(0)) if device_type == "cuda" else float("inf")
    if peak_flops == float("inf"):
        log0("  未登记 GPU，MFU 不显示")
    else:
        log0(f"  GPU 峰值算力（fp32累加）= {peak_flops/1e12:.1f} TFLOPS")
    log0(f"  预计总 FLOPs = {flops_per_token * sc['total_batch_size'] * num_iterations:.3e}")

    # ---- 6) 训练循环 ----
    model.train()
    # best_bpb 与 history 已在上面的续训块里初始化（续训时会从 meta 恢复）
    ema, smooth, total_time = 0.9, 0.0, 0.0
    t_start = time.time()
    log0("── 训练 " + "─" * 58)

    for step in range(start_step, num_iterations + 1):
        last_step = step == num_iterations
        sync()
        t0 = time.time()

        # ---- 6.1 梯度累积 ----
        # 每个 micro-step 独立前向反向，最后一次才 step。
        # loss 要除以累积步数：backward 是「梯度累加」，
        # 不除的话有效学习率会放大 grad_accum 倍。
        loss_val = 0.0
        for micro in range(sc["grad_accum_steps"]):
            loss = model(x, y)
            loss_val = loss.detach()
            (loss / sc["grad_accum_steps"]).backward()
            # 立刻预取下一批：这次 H2D 与 GPU 上的反向重叠，不浪费时间
            x, y, dl_state = next(train_loader)

        # ---- 6.2 更新超参 ----
        lm = lr_mult(step)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lm
            if g["kind"] == "muon":
                g["momentum"] = mom_sched(step)
                g["weight_decay"] = wd_sched(step)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        sync()
        dt = time.time() - t0
        if step > 10:
            total_time += dt

        # ---- 6.3 日志 ----
        lf = loss_val.item()
        smooth = ema * smooth + (1 - ema) * lf
        debiased = smooth / (1 - ema ** (step + 1))
        tok_per_sec = int(sc["total_batch_size"] / dt)
        flops_per_sec = flops_per_token * sc["total_batch_size"] / dt
        mfu = 100 * flops_per_sec / peak_flops if peak_flops != float("inf") else 0.0

        if step % cfg.train.log_every == 0 or step <= 3 or last_step:
            done = step - 10
            eta = (num_iterations - step) * total_time / done / 60 if done > 0 else 0
            log0(f"step {step:5d}/{num_iterations} ({100*step/num_iterations:5.1f}%)"
                 f" | loss {debiased:7.4f} | lrm {lm:.3f}"
                 f" | dt {dt*1000:6.1f}ms | tok/s {tok_per_sec:,}"
                 f" | mfu {mfu:5.1f}% | {total_time/60:5.1f}m eta {eta:.1f}m")
        history.append(dict(step=step, train_loss=debiased, lrm=lm, dt=dt,
                            tok_per_sec=tok_per_sec, mfu=mfu,
                            tokens=step * sc["total_batch_size"]))

        # ---- 6.4 验证 ----
        if cfg.train.eval_every > 0 and (last_step or step % cfg.train.eval_every == 0):
            val_loader = make_dataloader(tokenizer, cfg.train.device_batch_size,
                                         cfg.model.sequence_len, "val", device)
            n_batches = max(1, cfg.train.eval_tokens //
                            (cfg.train.device_batch_size * cfg.model.sequence_len))
            bpb = evaluate_bpb(model, val_loader, token_bytes, n_batches)
            best_bpb = min(best_bpb, bpb)
            log0(f"  >>> step {step}: val_bpb = {bpb:.4f}  (历史最好 {best_bpb:.4f})")
            history.append(dict(step=step, val_bpb=bpb))
            model.train()

        # ---- 6.5 采样 ----
        if cfg.train.sample_every > 0 and (last_step or step % cfg.train.sample_every == 0):
            model.eval()
            for pi, prompt in enumerate(["The capital of France is",
                                         "In 2026, the price of compute"]):
                ids = tokenizer.encode(prompt, prepend="<|bos|>")
                # seed 随 prompt 变，否则同一 step 的多个 prompt 会因
                # 随机数相同而给出完全一样的续写，看着像 bug 其实不是
                out = list(model.generate(ids, max_tokens=16, temperature=0.7,
                                          seed=step * 100 + pi))
                log0(f"  [{prompt!r}]\n    -> {tokenizer.decode(out)!r}")
            model.train()

        # ---- 6.6 存档 ----
        if last_step or (cfg.train.save_every > 0 and step % cfg.train.save_every == 0):
            save_checkpoint(ckpt_dir, step, model.state_dict(), {
                "model_config": asdict(cfg.model),
                "run_config": cfg.to_dict(),
                "step": step, "best_val_bpb": best_bpb,
                "dataloader_state": dl_state,
                "history": history,
            })
            # 存档成功的提示由 save_checkpoint 自己打印，这里不再重复

    # ---- 7) 收尾 ----
    sync()
    peak_mem = get_max_memory(device_type)() / 1024 ** 2
    total_flops = flops_per_token * sc["total_batch_size"] * num_iterations
    log0("\n── 完成 " + "─" * 58)
    log0(f"  总耗时（不含前 10 步）: {human_time(total_time)}")
    log0(f"  全程墙钟              : {human_time(time.time() - t_start)}")
    log0(f"  峰值显存              : {peak_mem:.0f} MiB")
    log0(f"  总 FLOPs              : {total_flops:.4e}")
    if total_time > 0 and peak_flops != float("inf"):
        log0(f"  平均 MFU              : {100*total_flops/total_time/peak_flops:.2f}%")
    if best_bpb < float("inf"):
        log0(f"  最好 val_bpb          : {best_bpb:.4f}")

    write_history(ckpt_dir, history)
    compute_cleanup()


def write_history(ckpt_dir, history):
    """把训练历史存成 CSV + 画曲线。matplotlib 失败不影响训练结果。"""
    import csv
    os.makedirs(ckpt_dir, exist_ok=True)
    keys = sorted({k for h in history for k in h})
    path = os.path.join(ckpt_dir, "history.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)
    log0(f"  训练历史 -> {path}")
    try:
        plot_history(history, os.path.join(ckpt_dir, "curves.png"))
    except Exception as e:
        log0(f"  （画图跳过：{type(e).__name__}）")


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [h["step"] for h in history if "train_loss" in h]
    tl = [h["train_loss"] for h in history if "train_loss" in h]
    vs = [h["step"] for h in history if "val_bpb" in h]
    vb = [h["val_bpb"] for h in history if "val_bpb" in h]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, tl); axes[0].set_title("train loss (nats/token)")
    axes[0].set_xlabel("step"); axes[0].grid(alpha=.3)
    if vs:
        axes[1].plot(vs, vb, "o-", color="crimson"); axes[1].set_title("val bpb (bits/byte)")
        axes[1].set_xlabel("step"); axes[1].grid(alpha=.3)
    msteps = [h["step"] for h in history if "mfu" in h]
    mfu = [h["mfu"] for h in history if "mfu" in h]
    axes[2].plot(msteps, mfu, color="seagreen"); axes[2].set_title("MFU (%)")
    axes[2].set_xlabel("step"); axes[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    log0(f"  训练曲线 -> {path}")


if __name__ == "__main__":
    main()
