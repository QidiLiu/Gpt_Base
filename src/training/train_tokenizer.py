"""
训练 BPE tokenizer。

流程：
    遍历预训练 shard 的 row group -> 取出原始文本 -> 喂给 rustbpe
    -> 存 tokenizer.pkl -> 顺便算出 token_bytes.pt（bpb 指标要用）

⚠ 本文件虽是 `src/training/`（卷5 通常只读），但其中**两个函数属于卷1 手抄**：
    iter_parquet_text    卷1 第 02 章「用生成器喂文本」
    compute_token_bytes  卷1 第 06 章「算 token_bytes」
  所以它们在 main 分支上是骨架。其余函数（main / report_compression / iter_tiny_text）
  是纯工程，只读即可。

跑法：
    uv run python -m training.train_tokenizer --vocab-size 16384
    uv run python -m training.train_tokenizer --vocab-size 16384 --shards 4
"""

import argparse
import os
import time

import torch

from common import (
    get_tokenizer_dir, log0, human_time, autodetect_device_type,
    get_base_dir, COMPUTE_DTYPE,
)
from data.dataset import list_parquet_files, download_tiny_shakespeare, get_base_dir_shakespeare
from data.tokenizer import BPETokenizer


# ===========================================================================
# ❗ 卷1 第 02 章：惰性遍历 parquet
# ===========================================================================
def iter_parquet_text(split="train", limit_shards=None):
    """
    惰性遍历 parquet 里的原始文本，喂给 BPE 训练器。

    ── 为什么必须是生成器 ──────────────────────────────
      4 个 shard 约 10 亿字符，不可能全读进内存。
      BPE 训练器只需要「一个接一个的字符串」，生成器正好满足。
      内存占用只等于「BPE 自身的合并表」，与数据量无关。

    ── 你要写的 ──────────────────────────────────────
    1) paths = list_parquet_files(split)，再按 limit_shards 截断
    2) 对每个 path：pq.ParquetFile(path)
    3) 遍历 pf.num_row_groups，逐个 pf.read_row_group(rg_idx)
    4) 把该 row group 的 "text" 列转成 python list 并 yield 出去
       （用 yield from 即可，不要先收集成 list —— 那就退化成全量读入内存了）
    """
    raise NotImplementedError(
        "待实现：iter_parquet_text ——\n"
        "  import pyarrow.parquet as pq\n"
        "  paths = list_parquet_files(split)\n"
        "  if limit_shards: paths = paths[:limit_shards]\n"
        "  for path in paths:\n"
        "      pf = pq.ParquetFile(path)\n"
        "      for rg_idx in range(pf.num_row_groups):\n"
        "          rg = pf.read_row_group(rg_idx)\n"
        "          yield from rg.column('text').to_pylist()\n"
        "\n"
        "  注意：这里必须逐 row group 惰性产出。一旦先 .to_pylist() 整个文件，\n"
        "        10 亿字符会一次性进内存，OOM。\n"
        "  验证：uv run python -m training.train_tokenizer --vocab-size 8192 --shards 1\n"
        "参考实现：git show solution:src/training/train_tokenizer.py")


def iter_tiny_text():
    """玩具数据：tinyshakespeare 全文（约 1.1 MB）。🔶 已给"""
    path = download_tiny_shakespeare()
    with open(path, encoding="utf-8") as f:
        yield f.read()


# ===========================================================================
# ❗ 卷1 第 06 章：算 token_bytes
# ===========================================================================
def compute_token_bytes(tok: BPETokenizer, device="cpu") -> torch.Tensor:
    """
    对每个 token id 记录它对应多少字节。

    ── 为什么需要这张表（bpb）────────────────────────────
      普通的 loss 是「每 token 的 nats」，它依赖词表大小，没法跨模型比较。
      bpb 是「每原始字节多少 bits」，与词表无关，可以公平比较。
          bpb = (loss_nats / ln2) / (该 token 平均覆盖的字节数)
      所以需要「token id -> 字节数」这张表。

    ── 你要写的 ──────────────────────────────────────
    1) n = tok.get_vocab_size()，建一个长度 n 的 float32 张量（zeros）
    2) 逐个 token id 用 len(tok.decode_bytes(tid)) 填
    3) ★ 必须在 tokenizer **最终定下来之后**、用同一个 tok 对象算
       （训练完再算，中途词表会变）
    """
    raise NotImplementedError(
        "待实现：compute_token_bytes ——\n"
        "  n = tok.get_vocab_size()\n"
        "  arr = torch.zeros(n, dtype=torch.float32, device=device)\n"
        "  for tid in range(n):\n"
        "      try:    arr[tid] = len(tok.decode_bytes(tid))\n"
        "      except Exception: arr[tid] = 0.0   # 特殊 token 解不出字节，记 0\n"
        "  return arr\n"
        "\n"
        "  验证：uv run python scratch/bpb_demo.py\n"
        "参考实现：git show solution:src/training/train_tokenizer.py")


def report_compression(tok: BPETokenizer):
    """在一段样本文本上测压缩率与 chars-per-token。"""
    sample = (
        "The quick brown fox jumps over the lazy dog. "
        "In 2026, the price of compute dropped by roughly 10x per year. "
        "Attention is all you need, but so is a good tokenizer."
    )
    ids = tok.encode(sample)
    n_bytes, n_tokens = len(sample.encode("utf-8")), len(ids)
    log0("")
    log0("── 压缩率报告 ──────────────────────────────")
    log0(f"样本文本        : {n_bytes} 字节, {n_tokens} token")
    log0(f"chars / token  : {len(sample)/n_tokens:.3f}")
    log0(f"bytes / token  : {n_bytes/n_tokens:.3f}")
    log0(f"压缩率         : {n_bytes/n_tokens:.3f} bytes/token")
    log0(f"词表大小       : {tok.get_vocab_size():,}")
    log0("───────────────────────────────────────────")
    log0(f"token 预览     : {tok.visualize(ids, with_token_id=True)}")
    log0("")


def main():
    p = argparse.ArgumentParser(description="训练 BPE tokenizer")
    p.add_argument("--vocab-size", type=int, default=16384,
                   help="含特殊 token 的总词表大小")
    p.add_argument("--shards", type=int, default=2,
                   help="用几个 shard 的文本训练（-1 = 全部）")
    p.add_argument("--toy", action="store_true",
                   help="改用 tinyshakespeare 训练（秒级，用于调试）")
    args = p.parse_args()

    t0 = time.time()
    tokdir = get_tokenizer_dir()
    ckpt = os.path.join(tokdir, "tokenizer.pkl")
    if os.path.exists(ckpt):
        log0(f"tokenizer 已存在，跳过训练：{ckpt}")
    else:
        if args.toy:
            log0("用 tinyshakespeare 训练 tokenizer（toy 模式）")
            tok = BPETokenizer.train(iter_tiny_text(), args.vocab_size)
        else:
            tok = BPETokenizer.train(
                iter_parquet_text("train", None if args.shards < 0 else args.shards),
                args.vocab_size,
            )
        tok.save(tokdir)

        # token_bytes 必须用最终词表算，所以放在训练之后
        device = autodetect_device_type()
        tb = compute_token_bytes(tok, device)
        torch.save(tb.cpu(), os.path.join(tokdir, "token_bytes.pt"))
        log0(f"token_bytes 已保存 ({tb.numel():,} tokens)")
        log0(f"tokenizer 训练耗时 {human_time(time.time()-t0)}")

    # 无论是否训练都跑一次报告，方便随时检查当前 tokenizer 的质量
    report_compression(BPETokenizer.load())


if __name__ == "__main__":
    main()
