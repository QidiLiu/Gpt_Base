"""
评测基座模型：bpb + 多选题准确率 + 采样。

跑法：
    uv run python -m training.eval_base --mode smoke
    bash script/eval_base.sh smoke
"""

import os
import json
import argparse


from common import (
    autodetect_device_type, compute_init, log0, get_runs_dir, load_latest,
)
from data.tokenizer import get_token_bytes
from data.dataloader import make_dataloader
from data.tasks import MMLU, ARC
from evaluation.metrics import evaluate_bpb, evaluate_multiple_choice
from common.config import default_tag


def parse_args():
    p = argparse.ArgumentParser(description="评测基座模型")
    p.add_argument("--mode", default="smoke",
                   choices=["debug", "smoke", "ablation", "full"])
    p.add_argument("--tag", default=None)
    p.add_argument("--eval-tokens", type=int, default=2 * 524288,
                   help="val bpb 用多少 token 统计")
    p.add_argument("--mc-examples", type=int, default=300,
                   help="每个多选任务评多少题")
    p.add_argument("--skip-download", action="store_true",
                   help="跳过任务数据集下载（只测 bpb）")
    return p.parse_args()


def main():
    args = parse_args()
    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    tag = args.tag or default_tag(args.mode)

    log0(f"载入 {tag} ...")
    model, tokenizer, meta = load_latest(get_runs_dir(), tag, device)
    model.eval()
    cfg = model.config
    log0(f"  d{cfg.n_layer} d{cfg.n_embd} h{cfg.n_head} T={cfg.sequence_len} V={cfg.vocab_size}")
    log0(model.describe())

    results = {"tag": tag, "step": meta.get("step"),
               "best_val_bpb_during_training": meta.get("best_val_bpb")}

    # ---- 1) val bpb ----
    log0("\n── 1) 验证集 bits-per-byte " + "─" * 36)
    log0("  bpb 与词表大小无关，是唯一能跨模型比较的语言建模指标。")
    log0("  参考量级：随机 ≈ 8，英文自然文本 ≈ 1.0-1.5，训好的小模型 ≈ 0.9-1.3")
    token_bytes = get_token_bytes(device)
    val_loader = make_dataloader(tokenizer, 8, cfg.sequence_len, "val", device)
    n_batches = max(1, args.eval_tokens // (8 * cfg.sequence_len))
    results["val_bpb"] = evaluate_bpb(model, val_loader, token_bytes, n_batches)

    # ---- 2) 多选题 ----
    if not args.skip_download:
        log0("\n── 2) 多选题 loglikelihood 准确率 " + "─" * 28)
        log0("  注意：基座模型没做过 SFT，它并不知道「要回答一个字母」。")
        log0("  所以这里的数字反映的是「语言理解能力」，不是「答题格式能力」。")
        log0("  SFT 之后再测才是有意义的对比 —— 那正是 eval_sft.sh 干的事。")
        for name, task in [("MMLU test", MMLU("all", "test")),
                           ("ARC-Easy", ARC("ARC-Easy", "test")),
                           ("ARC-Challenge", ARC("ARC-Challenge", "test"))]:
            items = [{"question": task[i]["question"], "choices": task[i]["choices"],
                      "gold": task[i]["gold"]} for i in range(min(len(task), args.mc_examples))]
            r = evaluate_multiple_choice(model, tokenizer, items, device,
                                         max_examples=args.mc_examples)
            log0(f"  {name:16s} acc = {r['accuracy']*100:5.1f}%  (n={r['n']})")
            results[name.lower().replace(" ", "_").replace("-", "_")] = r["accuracy"]

    # ---- 3) 采样 ----
    log0("\n── 3) 固定 prompt 采样 " + "─" * 44)
    from inference.engine import Engine
    eng = Engine(model, tokenizer)
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "In 2026, the price of compute",
        "def fibonacci(n):",
    ]
    for q in prompts:
        ids = tokenizer.encode(q, prepend="<|bos|>")
        out, _ = eng.generate_batch(ids, num_samples=1, max_tokens=32,
                                    temperature=0.0, use_tools=False)
        log0(f"  {q!r}\n    -> {tokenizer.decode(out[0][len(ids):])!r}")

    # ---- 4) 与 nanochat 的可比性说明 ----
    log0("\n── 4) 数字能不能和 nanochat 比？ " + "─" * 32)
    log0("  不能。原因有三：")
    log0("   (a) 模型规模差一个数量级（ablation 档 d6 = 17M，full 档 d24 = 182M scaling 参数）")
    log0("   (b) 总训练 FLOPs 差约两个数量级")
    log0("   (c) 数据集不同（本项目 ClimbMix 子集 vs nanochat 同源但分片数不同）")
    log0("  bpb 的价值在于「同一模型开/关某个 trick 后的相对差异」，而不是绝对值。")
    log0("  那些差异才 0.01-0.05 量级 —— 这正是教程要你亲手测出来的东西。")

    out_path = os.path.join(get_runs_dir(), "eval_base.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log0(f"\n  结果已写入 {out_path}")


if __name__ == "__main__":
    main()
