"""
对照实现：nanoGPT 式的 **naive 拼接** dataloader。

这是 tutorial/卷1 第 05 章「★ 消融实验」用的对照实现。

────────────────────────────────────────────────────────────────
它和 make_dataloader 的唯一区别
────────────────────────────────────────────────────────────────
    best-fit : 每行以 BOS 开头，文档边界用 BOS 显式标记，0% 跨文档
    naive    : 所有 token 串成一条长流，随机切 T 长窗口，
               窗口会横跨文档边界（cross-contamination）

────────────────────────────────────────────────────────────────
怎么衡量「污染」
────────────────────────────────────────────────────────────────
`contamination_rate` 统计每行**第一个 BOS 之前**有多少个 token。
    · best-fit：恒为 0（每行一定从 BOS 开始）
    · naive   ：> 0 表示这一行是从某篇文档的**中间**切进来的，
                那些 token 的上下文是「残缺的」——
                模型看到的是一个没有开头的句子，却要预测它的下一个词。

用 `contaminated_fraction` 得到「有多少比例的行是从文档中间开始的」。
这两个数字就是 bpb 差异的根源。
"""

import argparse
import random

import pyarrow.parquet as pq
import torch

from data.dataset import list_parquet_files


def build_token_pool(tokenizer, split="train", pool_tokens=4_000_000,
                     n_row_groups=6):
    """
    把若干个 row group 的所有文档串成一条 token 长流。

    真实训练里这一步是流式的（不可能全读进内存）。
    这里为了做对照实验，先收集固定数量的 token 就够了。
    """
    pool = []
    bos = tokenizer.get_bos_token_id()
    for path in list_parquet_files(split):
        pf = pq.ParquetFile(path)
        for rg_idx in range(min(pf.num_row_groups, n_row_groups)):
            for text in pf.read_row_group(rg_idx).column("text").to_pylist():
                pool.extend(tokenizer.encode(text, prepend=bos))
                if len(pool) >= pool_tokens:
                    return pool
    return pool


def naive_dataloader(tokenizer, batch_size, seq_len, pool, device="cpu", seed=0):
    """
    产出 (x, y)，形状与 make_dataloader 相同，但窗口会跨文档。

    yields
        x (B, T) 随机切出的窗口
        y (B, T) 右移一位
    """
    rng = random.Random(seed)
    while True:
        ix = [rng.randrange(0, len(pool) - seq_len - 1) for _ in range(batch_size)]
        x = torch.stack([torch.tensor(pool[i:i + seq_len], dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(pool[i + 1:i + 1 + seq_len], dtype=torch.long)
                         for i in ix])
        yield x.to(device), y.to(device)


# ===========================================================================
# 污染度量
# ===========================================================================
def contamination_rate(x, bos):
    """
    每行「第一个 BOS 之前」的 token 数。

    返回 0 表示该行从 BOS 开始（干净）。
    返回 >0 表示该行从某篇文档中间切进来（有污染）。
    返回 -1 表示该行一个 BOS 都没有（整行都在一篇文档内部，也算污染）。
    """
    out = []
    for row in x:
        idx = (row == bos).nonzero()
        out.append(int(idx[0]) if len(idx) else -1)
    return out


def contaminated_fraction(x, bos):
    """有多少比例的行是从文档中间开始的（不是从 BOS 开始）。"""
    rates = contamination_rate(x, bos)
    return sum(1 for r in rates if r != 0) / len(rates)


# ===========================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="naive 拼接对照实现（第 05 章消融实验）")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--pool-tokens", type=int, default=4_000_000)
    p.add_argument("--n-batches", type=int, default=16)
    args = p.parse_args()

    from data.tokenizer import get_tokenizer
    tok = get_tokenizer()
    bos = tok.get_bos_token_id()

    print("=== 构建 token 长流（naive 拼接的前提）===")
    pool = build_token_pool(tok, "train", args.pool_tokens)
    print(f"  池子 {len(pool):,} token  （{len(pool)/tok.get_vocab_size():.0f} 倍词表大小）")

    print("\n=== naive 拼接 vs best-fit 的污染率 ===")
    print(f"{'策略':<12} {'污染行占比':>10} {'首BOS前token数(前12行)':>24}")
    print("-" * 52)

    dl = naive_dataloader(tok, args.batch_size, args.seq_len, pool, device="cpu")
    rates_naive = []
    for _ in range(args.n_batches):
        x, _ = next(dl)
        rates_naive += contamination_rate(x, bos)
    n_naive = sum(1 for r in rates_naive if r != 0) / len(rates_naive) * 100
    print(f"{'naive':<12} {n_naive:9.1f}% {str(rates_naive[:12]):>24}")

    # best-fit 用真实的 dataloader 对比
    # 注意 make_dataloader 产出 3 个值 (x, y, state)，naive 版只产出 2 个
    from data.dataloader import make_dataloader
    dl2 = make_dataloader(tok, args.batch_size, args.seq_len, "train", device="cpu")
    rates_bestfit = []
    for _ in range(args.n_batches):
        batch = next(dl2)
        rates_bestfit += contamination_rate(batch[0], bos)
    n_best = sum(1 for r in rates_bestfit if r != 0) / len(rates_bestfit) * 100
    print(f"{'best-fit':<12} {n_best:9.1f}% {str(rates_bestfit[:12]):>24}")

    print("\n→ best-fit 恒为 0%（每行一定从 BOS 开始）")
    print("→ naive 的大多数行 > 0，意味着模型要从「没有开头的句子」开始预测")
    print("\n=== 一行 naive 窗口的开头长什么样（注意第一个 BOS 的位置）===")
    x, _ = next(naive_dataloader(tok, 1, args.seq_len, pool, device="cpu"))
    r = contamination_rate(x, bos)[0]
    n_show = min(args.seq_len, 40)
    print(f"  首个 BOS 位于位置 {r}")
    print("  " + tok.visualize(x[0, :n_show].tolist()))
