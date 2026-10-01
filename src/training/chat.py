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

from common import autodetect_device_type, compute_init, log0, get_runs_dir
from inference.engine import Engine, render_chat_prompt
from common.checkpoint import load_checkpoint, find_latest
from model.gpt import build_model
from data.tokenizer import get_tokenizer
from common.config import ModelConfig, default_tag


def parse_args():
    p = argparse.ArgumentParser(description="和模型聊天")
    p.add_argument("--mode", default="smoke",
                   choices=["debug", "smoke", "ablation", "full"])
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
    base_tag = tag or default_tag(mode)
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


def render_history(tokenizer, system, history, user_text=None):
    """
    把 (user, assistant) 历史 + 本轮 user_text 拼成一段 prompt。

    ── ★ user_text 必须传进来 ──────────────────────────────────
    曾经的 bug：`ask(user_text)` 调用的是 `render_history(tok, system, history)`
    —— **根本没把 user_text 传下去**，它只在生成完回复后进了
    history.append。于是模型收到的 prompt 是
        <|bos|><|user_start|><|user_end|><|assistant_start|>
    一个**空的 user 轮**，模型在凭空回答，`chat.sh` 完全不可用。
    现在 user_text 是第 4 个参数，缺省 None 时只渲染历史前缀。

    ── ★ 特殊 token 必须用 encode_special 逐个取 id ──────────────
    不能写成 f"<|user_start|>{u}<|user_end|>" 再整个 encode()：
    encode() 走 tiktoken 的 encode_ordinary，它**按定义忽略** special
    tokens，会把 "<|user_start|>" 切成 12 个普通 token。
    详见 inference/engine.py:render_chat_prompt 的完整说明。

    ── ★ 最后一轮交给 render_chat_prompt ───────────────────────
    训练时（render_conversation）产出的序列是
        BOS <|user_start|> u <|user_end|> <|assistant_start|> a <|assistant_end|>
    推理 prompt 必须停在 <|assistant_start|>。
    为了让 chat 永远不会和训练格式漂移，本轮 (system + user_text)
    直接复用 render_chat_prompt —— 它与 render_conversation 的输出
    逐 token 对齐，由 tests 锁定。
    """
    S = tokenizer.encode_special
    ids = [tokenizer.get_bos_token_id()]
    for i, (user, assistant) in enumerate(history):
        # system 合并进**第一条** user 消息（render_conversation 就是这么做的，
        # 实测：system + "\n\n" + 第一条 user 合成一个 user 轮）。
        text = f"{system}\n\n{user}" if (system and i == 0) else user
        ids += [S("<|user_start|>")] + tokenizer.encode(text) + [S("<|user_end|>")]
        ids += [S("<|assistant_start|>")] + tokenizer.encode(assistant) + [S("<|assistant_end|>")]
    if user_text is None:
        # 只要历史前缀：收在 <|user_start|>，调用方自己接内容
        return ids + [S("<|user_start|>")]
    # 本轮（含 system，若历史为空则 system 归这一轮）。
    # [1:] 去掉 render_chat_prompt 自带的 BOS —— 本函数已经加过了。
    tail_system = None if history else system
    return ids + render_chat_prompt(tokenizer, user_text, tail_system)[1:]


def ask(engine, tokenizer, history, user_text, *, system=None,
        max_tokens=256, temperature=0.7, top_k=40, seed=42, use_tools=True):
    """
    问一轮，把 (user_text, reply) 追加进 history，返回 reply。

    ── 为什么要提成模块级函数 ──────────────────────────────────
    之前它是 main() 里的闭包，**无法被测试调用**。于是
    「ask 没有把 user_text 传给 render_history」这个 bug 零测试覆盖，
    一直活到今天 —— 模型收到的是空 user 轮，chat.sh 完全不可用。
    提成模块级之后，tests/test_engine.py 可以直接调它，
    并断言真正送进 Engine 的 ids 里含有用户的问题。
    """
    ids = render_history(tokenizer, system, history, user_text)
    out, _ = engine.generate_batch(
        ids, num_samples=1, max_tokens=max_tokens,
        temperature=temperature, top_k=top_k,
        seed=seed, use_tools=use_tools)
    reply = tokenizer.decode(out[0][len(ids):])
    history.append((user_text, reply))
    return reply


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

    def _ask(user_text):
        return ask(engine, tokenizer, history, user_text,
                   system=args.system, max_tokens=args.max_tokens,
                   temperature=args.temperature, top_k=args.top_k,
                   seed=args.seed, use_tools=not args.no_tools)

    # ---- 单次模式 ----
    if args.prompt:
        print(_ask(args.prompt))
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
        print(f"\033[90m模型 > \033[0m{_ask(user_text)}")


if __name__ == "__main__":
    main()
