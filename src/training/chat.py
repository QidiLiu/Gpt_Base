"""
和基座模型聊天（SFT 后）。

跑法：
    bash script/chat.sh smoke              # 交互式
    bash script/chat.sh smoke -p "你好"      # 单次提问

支持的选项：
    --temperature / -t   采样温度，0 = 贪心（确定性）
    --top-k / -k         top-k 截断
    --max-tokens / -m    最长生成多少 token
    --no-tools           关闭计算器工具
    --seed               随机种子
"""

import argparse
import sys

from common import autodetect_device_type, compute_init, log0, get_runs_dir
from inference.engine import Engine
from common.checkpoint import load_checkpoint, find_latest
from model.gpt import build_model
from data.tokenizer import get_tokenizer
from common.config import ModelConfig


def parse_args():
    p = argparse.ArgumentParser(description="和模型聊天")
    p.add_argument("--mode", default="smoke", choices=["debug", "smoke", "full"])
    p.add_argument("--tag", default=None, help="基座 tag（决定用哪个 SFT 模型）")
    p.add_argument("-p", "--prompt", default=None, help="单次提问（不给则进入交互模式）")
    p.add_argument("-t", "--temperature", type=float, default=0.7)
    p.add_argument("-k", "--top-k", type=int, default=40)
    p.add_argument("-m", "--max-tokens", type=int, default=256)
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--system", default=None, help="可选的 system 提示")
    return p.parse_args()


def load_sft_model(mode, tag, device):
    """优先加载 SFT 后的模型；没跑过 SFT 就回落到基座。"""
    import os
    base_tag = tag or {"debug": "d2", "smoke": "d4", "full": "d6"}[mode]
    runs = get_runs_dir()
    sft_dir = os.path.join(runs, "sft_checkpoints", base_tag)
    tokenizer = get_tokenizer()

    if os.path.isdir(sft_dir) and find_latest(sft_dir) is not None:
        step = find_latest(sft_dir)
        meta = load_checkpoint(sft_dir, step, "cpu", "meta")
        log0(f"载入 SFT 模型 {base_tag} (step {step})")
    else:
        log0(f"未找到 SFT 模型，回落到基座 {base_tag}"
             f"（先跑 bash script/train_sft.sh {mode}）")
        from common.checkpoint import load_latest
        model, tokenizer, meta = load_latest(runs, base_tag, device)
        return model, tokenizer, base_tag

    fields = set(ModelConfig.__dataclass_fields__)
    cfg = ModelConfig(**{k: v for k, v in meta["model_config"].items() if k in fields})
    model = build_model(cfg, device=device)
    model.load_state_dict(load_checkpoint(sft_dir, step, device, "model"))
    return model, tokenizer, base_tag


def render_history(tokenizer, system, history):
    """把 (user, assistant) 历史拼成一段带特殊 token 的 prompt。"""
    S = tokenizer.encode_special
    ids = [tokenizer.get_bos_token_id()]
    if system:
        ids += [S("<|user_start|>"), S("<|assistant_start|>")]
        ids += tokenizer.encode(system)
        ids += [S("<|assistant_end|>")]
    for user, assistant in history:
        ids += [S("<|user_start|>")] + tokenizer.encode(user) + [S("<|user_end|>")]
        ids += [S("<|assistant_start|>")] + tokenizer.encode(assistant) + [S("<|assistant_end|>")]
    ids += [S("<|user_start|>")]
    return ids


def main():
    args = parse_args()
    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, tag = load_sft_model(args.mode, args.tag, device)
    model.eval()
    engine = Engine(model, tokenizer)

    log0("")
    log0(f"  模型: {tag} | 温度 {args.temperature} | top-k {args.top_k} | "
         f"工具 {'关' if args.no_tools else '开'}")
    log0("  命令: /reset 清空历史, /exit 退出")
    log0("")

    history = []

    def ask(user_text):
        ids = render_history(tokenizer, args.system, history)
        out, _ = engine.generate_batch(
            ids, num_samples=1, max_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
            seed=args.seed, use_tools=not args.no_tools)
        reply = tokenizer.decode(out[0][len(ids):])
        history.append((user_text, reply))
        return reply

    # ---- 单次模式 ----
    if args.prompt:
        print(ask(args.prompt))
        return

    # ---- 交互模式 ----
    while True:
        try:
            user_text = input("\033[36m你 > \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            break
        if user_text == "/reset":
            history.clear()
            log0("  （历史已清空）")
            continue
        print(f"\033[90m模型 > \033[0m{ask(user_text)}")


if __name__ == "__main__":
    main()
