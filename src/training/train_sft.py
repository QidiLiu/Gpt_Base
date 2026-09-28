"""
SFT（监督微调）：把预训练模型变成会对话的模型。

对应教程卷7「SFT 与 loss masking」。

核心只有一句话：**loss masking**。
    预训练时模型学「无条件地接着写下一个 token」
    SFT 只想让它学「assistant 该说什么」
    所以只有 assistant 产生的 token 参与 loss，其余全部 mask 掉

其余部分（超参继承、warm-start 优化器、验证集配比）都是工程细节，
但每一条都有它的理由，代码注释里都写了。
"""

import os
import time
import argparse

import torch

from common import (
    autodetect_device_type,
    compute_init,
    compute_cleanup,
    log0,
    get_runs_dir,
    synchronize,
    human_time,
    save_checkpoint,
    load_latest,
)
from data.tokenizer import get_tokenizer
from data.tasks import default_sft_mixture, default_sft_validation
from optim.muon import setup_optimizer


def parse_args():
    p = argparse.ArgumentParser(description="SFT")
    p.add_argument("--mode", default="smoke", choices=["debug", "smoke", "full"])
    p.add_argument("--base-tag", default=None, help="基座模型 tag（默认 d4/d6）")
    p.add_argument("--num-iterations", type=int, default=None)
    p.add_argument("--device-batch-size", type=int, default=None)
    p.add_argument("--mmlu-epochs", type=int, default=3)
    p.add_argument("--gsm8k-epochs", type=int, default=4)
    p.add_argument("--smoltalk-shards", type=int, default=1,
                   help="只用 SmolTalk 的前几个分片（1 片≈224MB/40K 条，省时间）")
    p.add_argument("--init-lr-frac", type=float, default=0.8,
                   help="起始 LR 占基座 LR 的比例")
    p.add_argument("--warmdown-ratio", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def make_sft_loader(task, tokenizer, device_batch_size, seq_len, device,
                    pad_id, seed=0):
    """
    SFT 的 dataloader：和预训练完全不同。

    三个关键差异：
      1) **定长 + 右侧 padding**（预训练是 best-fit 无 padding）
         一条对话长短差异巨大，必须补齐到 seq_len。
         padding 用 <|bos|>，并且 mask=0，模型学不到它。
      2) **保留 mask**（预训练是全 1）
         这是 SFT 的全部秘密。
      3) **按序列而不是按 token 计 loss**
         预训练每个 token 一份；SFT 只对 mask=1 的 token 计。
    """
    g = torch.Generator().manual_seed(seed)

    def gen():
        n = len(task)
        idx = 0
        skipped = 0
        while True:
            ex = task[idx % n]
            idx += 1
            # truncate="left"：保留对话结尾（最近一轮 assistant 回复）。
            # 若用默认的 "right"，长对话的 assistant 回复会被整段切掉，
            # 导致 mask 全 0 -> 交叉熵 0/0 = NaN。实测 23% 的样本会踩中。
            ids, mask = tokenizer.render_conversation(ex, max_tokens=seq_len,
                                                      truncate="left")
            if len(ids) < 2:
                skipped += 1
                continue
            # 防御：即使 truncate="left"，仍要确认至少有一个有效 target
            if not any(mask[1:]):
                skipped += 1
                if skipped % 200 == 1:
                    log0(f"  （已跳过 {skipped} 条无有效监督信号的对话）")
                continue
            T = seq_len
            inp = torch.full((T,), pad_id, dtype=torch.long)
            tgt = torch.full((T,), -1, dtype=torch.long)     # -1 = 不计 loss
            m = torch.zeros(T, dtype=torch.long)
            L = len(ids)
            inp[:L] = torch.tensor(ids, dtype=torch.long)
            m[:L] = torch.tensor(mask, dtype=torch.long)
            # targets 是 inputs 右移一位；最后一个位置没有 target
            tgt[:L - 1] = torch.tensor(ids[1:L], dtype=torch.long)
            tgt[:L - 1] = tgt[:L - 1].where(m[:L - 1] == 1, -1)
            yield inp, tgt, m

    it = gen()
    while True:
        xs, ts = [], []
        for _ in range(device_batch_size):
            x, t, _ = next(it)
            xs.append(x); ts.append(t)
        x = torch.stack(xs).to(device, non_blocking=True)
        t = torch.stack(ts).to(device, non_blocking=True)
        yield x, t


def main():
    args = parse_args()
    device_type = autodetect_device_type()
    ddp, rank, local_rank, world_size, device = compute_init(device_type)
    sync = synchronize(device_type)
    tokenizer = get_tokenizer()
    pad_id = tokenizer.get_bos_token_id()

    # ---- 1) 载入基座 ----
    base_tag = args.base_tag or {"debug": "d2", "smoke": "d4", "full": "d6"}[args.mode]
    log0(f"载入基座模型 {base_tag} ...")
    model, tok, base_meta = load_latest(get_runs_dir(), base_tag, device)
    mcfg = model.config
    log0(f"  基座: d{mcfg.n_layer} d{mcfg.n_embd} h{mcfg.n_head} T{mcfg.sequence_len} V{mcfg.vocab_size}")

    # ---- 2) 继承超参 ----
    # 为什么继承？SFT 和预训练是同一个模型的不同阶段，
    # 换 batch size / LR 相当于换了一次「冷启动」，会浪费基座的训练成果。
    base_cfg = base_meta.get("run_config", {})
    dbs = args.device_batch_size or base_cfg.get("train", {}).get("device_batch_size", 8)
    dbs = max(2, dbs // 2)          # SFT 序列更「难」一点，batch 减半
    seq_len = mcfg.sequence_len
    log0(f"  继承并调整: device_batch_size={dbs} (基座的 {base_cfg.get('train',{}).get('device_batch_size','?')} 的一半), seq_len={seq_len}")

    # ---- 3) 数据 ----
    log0("── 数据 " + "─" * 58)
    train_task = default_sft_mixture(args.mmlu_epochs, args.gsm8k_epochs,
                                    smoltalk_shards=args.smoltalk_shards)
    val_task = default_sft_validation()
    log0(f"  训练混合: {len(train_task):,} 条 (SmolTalk×1, MMLU×{args.mmlu_epochs}, GSM8K×{args.gsm8k_epochs})")
    log0(f"  验证混合: {len(val_task):,} 条")
    loader = make_sft_loader(train_task, tokenizer, dbs, seq_len, device, pad_id)

    # ---- 4) 优化器 ----
    # weight_decay=0：预训练末尾已经把 WD 衰减到 0 了，SFT 继续用 0。
    # init-lr-frac：不要从 0 开始（那等于重新 warmup），
    #   也不要用满 LR（会破坏基座学到的表征），用 80%。
    from common.config import AdamWConfig, MuonConfig, OptimConfig
    optim_cfg = OptimConfig(
        muon=MuonConfig(flavor="simple", lr=0.02, weight_decay=0.0),
        adamw=AdamWConfig(embedding_lr=0.05, unembedding_lr=0.005),
    )
    optimizer = setup_optimizer(model, optim_cfg, batch_lr_scale=1.0)
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"] * args.init_lr_frac
    log0(f"  LR 起始比例 {args.init_lr_frac}，warmdown 比例 {args.warmdown_ratio}")

    # ---- 5) 步数 ----
    steps_per_epoch = max(1, len(train_task) // (dbs * seq_len // 16))
    num_iterations = args.num_iterations or min(steps_per_epoch, 2000)
    log0(f"  训练步数 {num_iterations}（一轮约 {steps_per_epoch} 步）")

    # ---- 6) 训练 ----
    ckpt_dir = os.path.join(get_runs_dir(), "sft_checkpoints", base_tag)
    model.train()
    ema, smooth = 0.9, 0.0
    history = []
    t0 = time.time()
    log0("── 训练 " + "─" * 58)

    for step in range(1, num_iterations + 1):
        sync(); ts = time.time()
        x, t = next(loader)
        loss = model(x, t)
        loss.backward()
        lrm = 1.0
        if step > num_iterations * (1 - args.warmdown_ratio):
            lrm = (num_iterations - step) / max(num_iterations * args.warmdown_ratio, 1)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm
        optimizer.step()
        model.zero_grad(set_to_none=True)
        sync(); dt = time.time() - ts

        lf = loss.item()
        smooth = ema * smooth + (1 - ema) * lf
        debiased = smooth / (1 - ema ** step)
        if step % args.log_every == 0 or step == 1:
            done = step - 1
            eta = (num_iterations - step) * dt * (num_iterations / done - 1) if done else 0
            log0(f"step {step:5d}/{num_iterations} ({100*step/num_iterations:5.1f}%)"
                 f" | loss {debiased:7.4f} | lrm {lrm:.3f} | dt {dt*1000:6.1f}ms"
                 f" | {(time.time()-t0)/60:4.1f}m")
        history.append(dict(step=step, sft_loss=debiased, lrm=lrm, dt=dt))

        if args.eval_every > 0 and step % args.eval_every == 0:
            log0(f"  >>> step {step} 抽样对话：")
            from inference.engine import Engine
            eng = Engine(model.eval(), tokenizer)
            for q in ["What is the capital of France?", "What is 12 * 7?"]:
                ids = tokenizer.encode(f"<|user_start|>{q}<|user_end|><|assistant_start|>",
                                       prepend="<|bos|>")
                out, _ = eng.generate_batch(ids, num_samples=1, max_tokens=32,
                                            temperature=0.0, use_tools=True)
                log0(f"    用户: {q}")
                log0(f"    助手: {tokenizer.decode(out[0][len(ids):])!r}")
            model.train()

    save_checkpoint(ckpt_dir, num_iterations, model.state_dict(), {
        "model_config": base_meta["model_config"],
        "base_tag": base_tag, "step": num_iterations, "history": history,
    })
    log0(f"\n  SFT 完成，耗时 {human_time(time.time()-t0)} -> {ckpt_dir}")
    compute_cleanup()


if __name__ == "__main__":
    main()
