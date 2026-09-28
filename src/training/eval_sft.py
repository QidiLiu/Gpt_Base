"""
评测 SFT 模型：多选准确率 + GSM8K pass@1 + 多轮对话展示。

与 eval_base 的关键区别：
    eval_base 测的是「语言理解能力」（基座没学过对话格式）
    eval_sft  测的是「按格式正确回答」的能力 —— 这才是 SFT 的目标
"""

import os
import json
import argparse

import torch

from common import (
    autodetect_device_type, compute_init, log0, get_runs_dir, COMPUTE_DTYPE,
)
from data.tokenizer import get_tokenizer
from data.tasks import MMLU, ARC, GSM8K
from evaluation.metrics import compute_pass_at_k
from inference.engine import Engine, KVCache
from common.checkpoint import load_checkpoint, find_latest
from model.gpt import build_model
from common.config import ModelConfig


def parse_args():
    p = argparse.ArgumentParser(description="评测 SFT 模型")
    p.add_argument("--mode", default="smoke", choices=["debug", "smoke", "full"])
    p.add_argument("--tag", default=None)
    p.add_argument("--mc-examples", type=int, default=300)
    p.add_argument("--gsm8k-examples", type=int, default=40)
    p.add_argument("--num-samples", type=int, default=4, help="GSM8K 每题采样几条算 pass@1")
    p.add_argument("--show-chat", type=int, default=4, help="展示几段多轮对话")
    return p.parse_args()


def load_sft(mode, tag, device):
    base_tag = tag or {"debug": "d2", "smoke": "d4", "full": "d6"}[mode]
    runs = get_runs_dir()
    sft_dir = os.path.join(runs, "sft_checkpoints", base_tag)
    if not (os.path.isdir(sft_dir) and find_latest(sft_dir) is not None):
        log0(f"没有 SFT 存档（{sft_dir}）。先跑：bash script/train_sft.sh {mode}")
        raise SystemExit(1)
    step = find_latest(sft_dir)
    meta = load_checkpoint(sft_dir, step, "cpu", "meta")
    fields = set(ModelConfig.__dataclass_fields__)
    cfg = ModelConfig(**{k: v for k, v in meta["model_config"].items() if k in fields})
    model = build_model(cfg, device=device)
    model.load_state_dict(load_checkpoint(sft_dir, step, device, "model"))
    model.eval()
    log0(f"载入 SFT 模型 {base_tag} (step {step})")
    return model, get_tokenizer()


def eval_mc(model, tokenizer, name, task, n, device):
    """多选题：直接让模型生成一个字母（这正是 SFT 教会它的格式）。"""
    mcfg = model.config
    m_nkv, m_hd, m_nl = mcfg.n_kv_head, mcfg.head_dim, mcfg.n_layer
    correct = 0
    for i in range(min(len(task), n)):
        ex = task[i]
        prompt = (f"<|user_start|>{ex['messages'][0]['content']}<|user_end|>"
                  f"<|assistant_start|>")
        ids = tokenizer.encode(prompt, prepend="<|bos|>")
        # 只喂最后一个位置：RoPE 会按 cache_seqlens 偏移到正确位置，
        # 所以等价于「已经处理了前 len(ids)-1 个 token」
        kv = KVCache(1, m_nkv, m_hd, m_nl, len(ids), device, dtype=COMPUTE_DTYPE)
        kv.cache_seqlens.fill_(len(ids) - 1)
        with torch.no_grad():
            logits = model(torch.tensor([ids[-1:]], device=device), kv_cache=kv)
        # 只在 A/B/C/D 四个 token 里取 argmax。
        # 不这么做的话，SFT 步数很少的模型会 argmax 到 <|python_start|>
        # 之类的特殊 token 上，得出 0% 这种没有信息量的数字。
        row = logits[0, -1]
        best, best_logit = 0, float("-inf")
        for i, L in enumerate("ABCD"):
            tid = tokenizer.encode(L)   # 字母单独成 token
            if row[tid] > best_logit:
                best, best_logit = i, row[tid]
        correct += int(best == ex["gold"])
    acc = correct / max(1, min(len(task), n))
    log0(f"  {name:16s} acc = {acc*100:5.1f}%  (n={min(len(task), n)})")
    return acc


def main():
    args = parse_args()
    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer = load_sft(args.mode, args.tag, device)
    engine = Engine(model, tokenizer)
    results = {}

    # ---- 1) 多选 ----
    log0("\n── 1) 多选题（生成式，SFT 后才有意义） " + "─" * 22)
    for name, task in [("MMLU test", MMLU("all", "test")),
                       ("ARC-Easy", ARC("ARC-Easy", "test")),
                       ("ARC-Challenge", ARC("ARC-Challenge", "test"))]:
        results[name] = eval_mc(model, tokenizer, name, task, args.mc_examples, device)

    # ---- 2) GSM8K pass@1 ----
    log0("\n── 2) GSM8K pass@1（要模型真的解出来） " + "─" * 24)
    log0("  这是唯一能验证「工具调用 + 推理」是否真的学会的指标。")
    log0("  小模型在这个任务上通常接近 0，别气馁 —— 看格式对不对就行。")
    task = GSM8K("main", "test")
    n = min(len(task), args.gsm8k_examples)
    shown = 0
    per_example = []
    for i in range(n):
        ex = task[i]
        prompt = (f"<|user_start|>{ex['messages'][0]['content']}<|user_end|>"
                  f"<|assistant_start|>")
        ids = tokenizer.encode(prompt, prepend="<|bos|>")
        outs, _ = engine.generate_batch(ids, num_samples=args.num_samples,
                                        max_tokens=192, temperature=0.8,
                                        top_k=40, seed=i, use_tools=True)
        # ★ 用满 num_samples 条，而不是 outs[:1]。
        #   pass@1 的无偏估计式化简后正好是 c/n（n 个样本里的正确比例），
        #   所以「用 n 条估 pass@1」和「只看第 1 条」是同一个量，
        #   但前者方差低得多（n=4 时标准差约为 n=1 的一半）。
        #   之前 outs[:1] 既浪费了 3/4 的采样，又拿到了更抖的估计。
        oks = [task.evaluate(ex, tokenizer.decode(o[len(ids):])) for o in outs]
        per_example.append(oks)
        if shown < 2:
            shown += 1
            log0(f"  题目: {ex['messages'][0]['content'][:90]}")
            log0(f"  正确答案: {ex['gold_answer']} | "
                 f"{sum(oks)}/{len(oks)} 条判对")
            log0(f"  模型输出: {tokenizer.decode(outs[0][len(ids):])[:220]!r}")
            log0("")
    pass1 = sum(compute_pass_at_k(oks, k=1) for oks in per_example) / max(n, 1)
    results["gsm8k_pass@1"] = pass1
    log0(f"  GSM8K pass@1 = {pass1*100:5.1f}%  "
         f"(n={n} 题 × {args.num_samples} 采样)")

    # ---- 3) 多轮对话展示 ----
    log0("\n── 3) 多轮对话 " + "─" * 52)
    convs = [
        [("What is the capital of France?", None),
         ("And of Japan?", None)],
        [("What is 12 * 7?", None)],
        [("Tell me a joke.", None), ("Make it funnier.", None)],
    ]
    for conv in convs[:args.show_chat]:
        history = []
        for user_text, _ in conv:
            S = tokenizer.encode_special
            ids = [tokenizer.get_bos_token_id()]
            for u, a in history:
                ids += [S("<|user_start|>")] + tokenizer.encode(u) + [S("<|user_end|>")]
                ids += [S("<|assistant_start|>")] + tokenizer.encode(a) + [S("<|assistant_end|>")]
            ids += [S("<|user_start|>")] + tokenizer.encode(user_text) + [S("<|user_end|>"),
                                                                          S("<|assistant_start|>")]
            outs, _ = engine.generate_batch(ids, num_samples=1, max_tokens=96,
                                            temperature=0.7, top_k=40, seed=0)
            reply = tokenizer.decode(outs[0][len(ids):])
            history.append((user_text, reply))
            log0(f"  \033[36m你 > \033[0m{user_text}")
            log0(f"  \033[90m模型 > \033[0m{reply}")
        log0("")

    out_path = os.path.join(get_runs_dir(), "eval_sft.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log0(f"  结果已写入 {out_path}")


if __name__ == "__main__":
    main()
