"""
评测指标。

三个指标，各回答一个不同的问题：

  bpb     这个模型建模语言的能力有多强？（与词表无关，可跨模型比较）
  acc     它能不能做对选择题？（用 loglikelihood 评分，不靠模型自己生成）
  pass@k  它能不能解对题？（要模型真的生成出来）

教程卷7 会逐个拆。
"""

import math
import torch
import torch.nn.functional as F

from common import log0


# ===========================================================================
# 1) bits per byte
# ===========================================================================
@torch.no_grad()
def evaluate_bpb(model, loader, token_bytes, max_batches: int) -> float:
    """
    在验证集上算 bpb（bits per byte，每字节消耗多少比特）。

    ── 为什么不直接用 loss？────────────────────────────────────
    loss 的单位是「每个 token 的 nats」。但不同模型的词表大小不同，
    一个 token 覆盖的字节数也不同，所以 loss **不可比**。
    8192 词表的模型和一个 50257 词表的模型，同样的 loss 意味着完全不同的
    建模能力。

    bpb 换算到「原始字节」这个与分词方式无关的单位，就可以公平比较了。

        bpb = (每 token 的 nats) / ln(2) / (每 token 的字节数)
            = total_nats / total_tokens / ln(2) / (total_bytes / total_tokens)
            = total_nats / ln(2) / total_bytes

    最后这个形式最干净：只需要累加「总 nats」和「总字节数」两个数。

    参考量级：随机字节约 8 bpb，自然英语约 1.0-1.5，训练良好的小模型约 0.8-1.0。
    """
    # ★ 顺序：先记住原状态，再切 eval()。反过来写的话 eval() 已经把
    #   training 置成 False，was_training 恒为 False，下面的还原成了死代码。
    was_training = model.training
    model.eval()
    total_nats, total_bytes = 0.0, 0.0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        # dataloader 产出 (x, y, state)；这里只关心前两个
        x, y = batch[0], batch[1]
        x, y = x.to(model.get_device()), y.to(model.get_device())
        # 逐 token 的 loss，reduction='none' 才能拿到每个位置的值
        loss_vec = model(x, y, loss_reduction="none")          # (B, T)
        valid = (y >= 0)                                        # -1 是不计 loss 的位置
        total_nats += loss_vec[valid].sum().item()
        total_bytes += token_bytes[y[valid]].sum().item()

    if was_training:
        model.train()

    bpb = total_nats / (math.log(2) * max(total_bytes, 1e-9))
    log0(f"  bpb: 总 nats={total_nats:,.0f} / 总字节={total_bytes:,.0f} -> {bpb:.4f}")
    return bpb


# ===========================================================================
# 2) 多选题：loglikelihood 评分
# ===========================================================================
@torch.no_grad()
def score_choices(model, tokenizer, question: str, choices: list[str],
                  device="cuda") -> list[float]:
    """
    给一道选择题的每个选项打分，返回各自的「平均每 token 对数似然」。

    ── 关键洞察：选择题不用模型「生成」答案 ─────────────────────
    最朴素的想法是「让模型生成 'A' 还是 'B'」，然后比较两个 token 的概率。
    但这样只用了 1 个 token 的信息，信噪比很低。

    更好的做法（lm-eval-harness / DCLM CORE 的标准做法）：
    对每个选项，把「问题 + 选项」完整拼起来，
    算模型给**选项部分**的平均对数似然，取最高的那个选项为预测。

    为什么取平均而不是求和？
      求和会让长选项天然吃亏（log 概率是负的，越长越低）。
      取平均相当于「按 token 归一化」，消除了长度偏置。
      代价是牺牲了「整句概率」这个更正确的似然 —— 实践上平均更稳。

    对小模型还有一个细节：必须保持各选项的 **token 边界一致**。
    比如选项是 "A" 和 "B"，那么 prompt 里必须正好是 "答案=A" 这种形式，
    不能一个带空格一个不带，否则 token 化结果不同，分数不可比。
    """
    scores = []
    bos = tokenizer.get_bos_token_id()
    prompt = f"Multiple Choice question: {question}\n"
    for letter, choice in zip("ABCDEFGH", choices):
        prompt += f"- {choice}={letter}\n"
    prompt += "\nRespond only with the letter of the correct answer."

    base_ids = tokenizer.encode(prompt, prepend=bos)
    for letter in "ABCDEFGH"[:len(choices)]:
        full = prompt + letter
        full_ids = tokenizer.encode(full, prepend=bos)
        # 「选项 token」的起点 = 完整序列比 prompt 多出来的部分
        # 注意不能假设多 1 个 token：字母前面可能触发不同的合并
        start = len(base_ids)
        # 保险起见用「最长公共前缀」定位，见 common_prefix_len
        start = common_prefix_len(base_ids, full_ids)
        if start >= len(full_ids):
            scores.append(float("-inf"))
            continue
        ids = torch.tensor([full_ids], device=device)
        logits = model(ids)                                    # (1, T, V)
        # 预测位置 i 的 token 是 full_ids[i+1]
        targets = ids[:, 1:]
        logprobs = F.log_softmax(logits[:, :-1].float(), dim=-1)
        picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # (T-1,)
        scores.append(picked[start - 1:].mean().item())
    return scores


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, items: list[dict],
                             device="cuda", max_examples: int = 500) -> dict:
    """
    在一批选择题上评测。items 每项形如
        {"question": str, "choices": [str,...], "gold": int}
    """
    # ★ 顺序：先记住原状态，再切 eval()。反过来写的话 eval() 已经把
    #   training 置成 False，was_training 恒为 False，下面的还原成了死代码。
    was_training = model.training
    model.eval()
    n = min(len(items), max_examples)
    correct = 0
    for it in items[:n]:
        scores = score_choices(model, tokenizer, it["question"],
                               it["choices"], device)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == it["gold"])
    if was_training:
        model.train()
    return {"accuracy": correct / max(n, 1), "n": n}


# ===========================================================================
# 3) pass@k
# ===========================================================================
@torch.no_grad()
def compute_pass_at_k(outcomes: list[list[bool]], k: int) -> float:
    """
    无偏估计的 pass@k（来自 Codex 论文的推导）。

    给一道题生成 n 个样本，其中 c 个正确。pass@k = 至少有一个对的概率。
    直接算 c/n 是有偏的（n 小的时候方差极大），正确的无偏估计是：

        pass@k = 1 - C(n-c, k) / C(n, k)

    C(a,b) = 0 当 a < b（样本不够挑，必然有一个对的）。

    这个公式的价值在于：**可以用 n > k 的采样来更稳地估 pass@k**。
    """
    from math import comb
    n = len(outcomes)
    if n == 0:
        return 0.0
    c = sum(outcomes[:k]) if k <= n else 0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
