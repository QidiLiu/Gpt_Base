"""
训练 BPE tokenizer（卷1 的第 3 章会手抄这个文件）。

流程：
    遍历预训练 shard 的 row group -> 取出原始文本 -> 喂给 rustbpe
    -> 存 tokenizer.pkl -> 顺便算出 token_bytes.pt（bpb 指标要用）

跑法：
    uv run python -m training.train_tokenizer --vocab-size 16384
    uv run python -m training.train_tokenizer --vocab-size 16384 --shards 4
"""

import argparse
import os
import time

import torch

from common import get_tokenizer_dir, log0, human_time, autodetect_device_type
from data.dataset import list_parquet_files, download_tiny_shakespeare
from data.tokenizer import BPETokenizer


def iter_parquet_text(split="train", limit_shards=None):
    """
    惰性遍历 parquet 里的原始文本，喂给 BPE 训练器。

    为什么用生成器？
      4 个 shard 约 10 亿字符，不可能全读进内存。
      BPE 训练器只需要「一个接一个的字符串」，生成器正好满足。
      内存占用只等于「BPE 自身的合并表」，与数据量无关。
    """
    import pyarrow.parquet as pq

    paths = list_parquet_files(split)
    if limit_shards:
        paths = paths[:limit_shards]
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            yield from rg.column("text").to_pylist()


def iter_tiny_text():
    """玩具数据：tinyshakespeare 全文（约 1.1 MB）。"""
    path = download_tiny_shakespeare()
    with open(path, encoding="utf-8") as f:
        yield f.read()


def compute_token_bytes(tok: BPETokenizer, device="cpu") -> torch.Tensor:
    """
    对每个 token id 记录它对应多少字节。

    用途：bpb（bits per byte）。
      普通的 loss 是「每个 token 的 nats」，它依赖词表大小，没法跨模型比较。
      bpb 是「每个原始字节多少 bits」，与词表无关，可以公平比较。
      换算：bpb = (loss_nats / ln2) / (该 token 平均覆盖的字节数)
      所以需要这张表。
    """
    n = tok.get_vocab_size()
    arr = torch.zeros(n, dtype=torch.float32, device=device)
    for tid in range(n):
        try:
            arr[tid] = len(tok.decode_bytes(tid))
        except Exception:
            # 理论上不该有解码不了的 token；特殊 token 也不是 0 字节
            arr[tid] = 0.0
    return arr


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

    # ★ 复用缓存前**必须核对词表大小**。
    #   缓存目录只有一个槽位，不按 vocab_size 分桶 —— 谁先跑谁建好，
    #   之后所有档位都复用那一个。
    #   曾经的真实后果：某次 smoke 建了个 V=8192 的 tokenizer，之后
    #   ablation/full 传 --vocab-size 16384 被静默忽略，真的按 8192 训练，
    #   而日志只写「tokenizer 已存在，跳过训练」——
    #   lm_head 尺寸差一倍，而没有任何一行提示对不上。
    #   详见 doc/tutorial/37-全流程与排错.md。
    need_train = True
    if os.path.exists(ckpt):
        try:
            cached_vocab = BPETokenizer.load(tokdir).get_vocab_size()
        except Exception as e:
            # 文件在但读不出来（半个 pkl / 版本不兼容）—— 同样该重建
            log0(f"缓存的 tokenizer 读不出来（{type(e).__name__}: "
                 f"{str(e)[:60]}），重建")
            cached_vocab = None
        if cached_vocab == args.vocab_size:
            log0(f"tokenizer 已存在（V={cached_vocab}），跳过训练：{ckpt}")
            need_train = False
        elif cached_vocab is not None:
            log0(f"缓存的 tokenizer 是 V={cached_vocab}，"
                 f"与要求的 V={args.vocab_size} 不符 -> 重建")

    if need_train:
        if args.toy:
            log0("用 tinyshakespeare 训练 tokenizer（toy 模式）")
            tok = BPETokenizer.train(iter_tiny_text(), args.vocab_size)
        else:
            tok = BPETokenizer.train(
                iter_parquet_text("train", None if args.shards < 0 else args.shards),
                args.vocab_size,
            )
        tok.save(tokdir)

        # token_bytes 必须用最终词表算，所以放在训练之后。
        # 旧的那个和词表不匹配，所以无条件覆盖 —— 不做「已存在就跳过」。
        device = autodetect_device_type()
        tb = compute_token_bytes(tok, device)
        torch.save(tb.cpu(), os.path.join(tokdir, "token_bytes.pt"))
        log0(f"token_bytes 已保存 ({tb.numel():,} tokens)")
        log0(f"tokenizer 训练耗时 {human_time(time.time()-t0)}")

    # 无论是否训练都跑一次报告，方便随时检查当前 tokenizer 的质量。
    # 必须传 tokdir —— 不传会回落到 get_tokenizer_dir()，
    # 那样报告的是「别处那个」tokenizer 而不是刚处理过的这个。
    report_compression(BPETokenizer.load(tokdir))


if __name__ == "__main__":
    main()
