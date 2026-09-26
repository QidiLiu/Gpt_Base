"""
Dataloader：把一堆变长文档打包成定长的 (B, T) 训练 batch。

**卷 1 最有价值的一段代码。** 建议在 tutorial/卷1 第 05 章的
「概念」部分读完再动手。

▓▓ 这一章要你敲的部分 ▓▓
    iter_documents  —— 读 parquet
    make_dataloader —— ★ best-fit 装箱算法（本文件核心）

────────────────────────────────────────────────────────────────
背景：为什么要装箱
────────────────────────────────────────────────────────────────
ClimbMix 文档长度实测（scratch/packing_demo.py）：
    min=4  p25=279  中位数=628  p75=829  max=25055
而 GPU 只吃矩形 tensor：`(B, T)`，T 必须固定。

四种做法，代价递增：
  0) padding 到最长    -> 99% 算力在算 PAD，且模型学到推理时不存在的模式
  1) 截断到固定长度    -> 长文档 97% 被丢
  2) naive 拼接        -> 窗口跨文档边界，模型学「从上一篇文章中间接着写」
  3) BOS-aligned best-fit -> 本文件采用

best-fit 的四条规则：
  (1) 每行开头是 <|bos|>
  (2) 每篇文档前面也有 <|bos|>（靠 tokenizer 编码时 prepend 实现）
  (3) 装箱时**优先选「能完整放下的最长文档」**
  (4) 都放不下时，**裁剪最短的**填满

代价与收益（务必读完 tutorial 第 05 章的「为什么丢 35% 划算」）：
  · 利用率 100%（无 padding），但约 35% token 在裁剪时被丢弃
  · 换来的：0% 跨文档污染 + 显式的文档边界信号
  · 65% 的干净数据 > 90% 的脏数据
"""

import torch

from data.dataset import list_parquet_files
from common import get_dist_info


# ===========================================================================
# ❗ 要你敲的部分 1：读 parquet
# ===========================================================================
def iter_documents(split: str, tokenizer_batch_size: int = 128,
                   resume_state: dict | None = None):
    """
    无限循环地产出 (text_batch, state)。

        text_batch  一批原始字符串
        state       {"pq_idx": 第几个 parquet 文件,
                     "rg_idx":  第几个 row group,
                     "epoch":   第几轮}

    保存 state 就实现了「任意位置精确续训」。

    ── 你要想清楚的点 ──────────────────────────────
    (1) 为什么必须用生成器？
        1 个 shard 有 8.6 万篇文档约 2.5 亿字符，一次读进内存会爆。
        惰性读取让内存占用只等于「一个 row group」，与数据量无关。

    (2) 为什么按 row group 读，而不是按文件读？
        实测 1 个 shard = 84 个 row group，每个 1024 篇文档、约 3 MB。
        parquet 的 row group 是天然的独立分块单位。

    (3) 外层为什么是 `while True`？
        因为训练是多 epoch 的，数据要循环用。

    (4) 续训时怎么避免重复训练？
        提示：从 resume_state 的 rg_idx 出发，并且**往前跳 1 个**。
        多卡时还要考虑每个 rank 只读不同的 row group（本项目 world_size 恒为 1，
        但代码形态要留着 —— 教程卷 6 会讲）。

    (5) train 和 val 的 shard 怎么区分？
        提示：看 data/dataset.py 的 list_parquet_files，它已经处理了。
    """
    raise NotImplementedError(
        "待实现：iter_documents ——\n"
        "  1) import pyarrow.parquet as pq；paths = list_parquet_files(split)\n"
        "  2) 从 resume_state 读出 pq_idx / rg_idx / epoch\n"
        "  3) while True 外层循环，遍历 paths\n"
        "  4) 每行：pf = pq.ParquetFile(path)\n"
        "  5) 内层 while 遍历 row group：rg = pf.read_row_group(i)\n"
        "     batch = rg.column('text').to_pylist()\n"
        "     按 tokenizer_batch_size 切片 yield (切片, state)\n"
        "  6) 每轮结束时 epoch += 1\n"
        "参考实现：git show solution:src/data/dataloader.py")


# ===========================================================================
# ❗ 要你敲的部分 2：best-fit 装箱
# ===========================================================================
def make_dataloader(tokenizer, batch_size: int, seq_len: int, split: str,
                    device="cuda", resume_state: dict | None = None,
                    buffer_size: int = 1000, tokenizer_threads: int = 4,
                    tokenizer_batch_size: int = 128):
    """
    产出 (inputs, targets, state)：
        inputs, targets  形状 (batch_size, seq_len)，dtype int64，在 device 上
        state             打包位置，用于精确续训

    ── 三个关键设计（务必想清楚，每一个都有代价）──────────────

    (1) 为什么每行容量是 seq_len + 1 而不是 seq_len？
        因为要同时构造 inputs 和 targets 这一对：
            inputs  = row[0 : T]
            targets = row[1 : T+1]
        所以每行需要 T+1 个 token。

    (2) 为什么全程不 padding？
        padding token 会让模型在推理时遇到从未见过的 id 分布。
        宁可裁掉 token，也要 100% 利用率 + 无 padding。

    (3) 为什么先在 CPU 拼好再一次性搬 GPU？
        逐行 torch.tensor(...) 会产生几千次小 H2D 拷贝，
        每次几微秒的启动开销累加起来很可观。
        正确做法：预分配 pinned CPU buffer -> 一次 non_blocking 拷贝。
        而且 gpu_buffer 反复复用，生成过程中零显存分配。

    ── 装箱算法本身（★ 本章核心）──────────────────────
    对每一行，反复：
        A) 找出「能完整放进剩余空间的、最长的」那篇文档
        B) 如果找到了 -> 放进去
        C) 如果一篇都放不下 -> 裁剪**最短的**那篇填满剩余空间，行结束

    为什么 A 要「最长的」？
        长文档能放下的位置越来越少，先安排它们，剩下的空间才能被小文档高效利用。
        （tutorial 第 05 章的实验 C 手工构造：best-fit 裁 5 个 token，
          first-fit 裁 10 个，裁最长的裁 45 个。）

    为什么 C 要「最短的」？
        裁最短的，浪费最少。
        （反例：裁最长的会浪费 668% —— 同样见第 05 章实验 B。）

    ── 三个必须避开的坑 ──────────────────────────────

    坑 1：手动写 row[0] = bos
        文档在 tokenizer 编码时已经 prepend=bos 了，第一篇被放进来的文档
        自带 BOS，位置 0 自然就是 BOS。手动再写一次会得到两个 BOS。
        正确做法：pos 从 0 开始，不手动写。
        验证：x[0, :3] 应该是 [True, False, False]。

    坑 2：忘了 break
        规则 C 执行后行已满。while 循环必须 break，
        否则下一轮 remaining=0，`doc[:0]` 是空，行永远不满。

    坑 3：CPU buffer 形状算错
        cpu_buffer 要装下 2 × batch_size × seq_len（inputs + targets），
        然后用 .view() 切成两半。
    """
    raise NotImplementedError(
        "待实现：make_dataloader ——\n"
        "  1) row_capacity = seq_len + 1\n"
        "  2) batches = iter_documents(split, tokenizer_batch_size, resume_state)\n"
        "     bos = tokenizer.get_bos_token_id()\n"
        "     doc_buffer = []  （已 token 化、还没用掉的文档）\n"
        "  3) 定义 refill()：next(batches) 取一批文本 -> tokenizer.encode(text_batch, prepend=bos, num_threads=...)\n"
        "     逐篇 append 到 doc_buffer，并更新 pq_idx/rg_idx/epoch\n"
        "  4) 预分配缓冲区：\n"
        "       cpu_buffer = torch.empty(2*B*T, dtype=long, pin_memory=use_cuda)\n"
        "       gpu_buffer = torch.empty(2*B*T, dtype=long, device=device)\n"
        "       用 .view(B,T) 切出 cpu_inputs/cpu_targets/inputs/targets\n"
        "  5) while True:\n"
        "       for r in range(batch_size):\n"
        "         pos = 0\n"
        "         while pos < row_capacity:\n"
        "           while len(doc_buffer) < buffer_size: refill()\n"
        "           remaining = row_capacity - pos\n"
        "           A) 线性扫 doc_buffer 找 (best_idx, best_len)，条件 len<=remaining 且 >best_len\n"
        "           B) if best_idx>=0: pop 出来写入 row[pos:pos+best_len]; pos += best_len\n"
        "           C) else: 找最短的 pop 出来写入 row[pos:pos+remaining]; pos = row_capacity; break\n"
        "         cpu_inputs[r] = row[:-1];  cpu_targets[r] = row[1:]\n"
        "       state = {...}\n"
        "       gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)\n"
        "       yield inputs, targets, state\n"
        "\n"
        "  验证：\n"
        "    uv run python scratch/check_data.py\n"
        "    目标：targets == inputs 右移一位 / 每行以 BOS 开头 / 无 padding\n"
        "参考实现：git show solution:src/data/dataloader.py")
