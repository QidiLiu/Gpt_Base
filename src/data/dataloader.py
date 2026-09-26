"""
Dataloader：把一堆变长文档打包成定长的 (B, T) 训练 batch。

这是本项目最值得逐行读懂的一段代码。教程卷1 的核心章节会完整拆解。

核心问题：文档长度不一（实测 ClimbMix 中位数 2400 字符，max 11 万），
但 GPU 只吃矩形 tensor。怎么不浪费地拼成矩形？

naive 做法（nanoGPT 的做法）：
    把所有 token 首尾相连切成一条长流，再随机切 T 长度的窗口。
    问题：窗口会横跨文档边界，模型学到「从上一篇文章的中间接着写」。
          论文里叫 "cross-contamination"。

本模块的做法（nanochat 的 BOS-aligned best-fit）：
    1) 每行开头放一个 <|bos|>
    2) 依次把文档塞进这一行，**优先选「能完整放下的最长文档」**（best-fit）
    3) 都放不下时，裁剪**最短的**文档填满剩余空间
    结果：利用率 100%（无 padding），但约 35% token 在裁剪时被丢掉。

为什么愿意丢 35% 的 token 换「文档边界干净」？
    因为那 35% 如果留着，模型会花容量去学「跨文档续写」这种
    在真实推理时永远不会遇到的模式。丢掉的 token 换成更干净的训练信号，
    净收益为正。nanochat 的 leaderboard 实验支持这个取舍。
"""

import torch

from data.dataset import list_parquet_files
from common import get_dist_info


# ---------------------------------------------------------------------------
# 阶段一：产出「文档文本批次」
# ---------------------------------------------------------------------------
def iter_documents(split: str, tokenizer_batch_size: int = 128,
                   resume_state: dict | None = None):
    """
    无限循环地产出 (text_batch, state)，text_batch 是一批原始字符串。

    state = {"pq_idx": 第几个 parquet 文件, "rg_idx": 第几个 row group, "epoch": 第几轮}
    保存它就实现了「任意位置精确续训」。

    为什么按 row group 读、而不是一次读完整个 shard？
      一个 shard 有 86,016 篇文档约 2.5 亿字符，一次读进来内存爆掉。
      parquet 的 row group 是天然的分块单位（每个约 3 MB / 1024 篇），
      按需读取让内存占用与数据量无关。
    """
    import pyarrow.parquet as pq

    _, rank, _, world_size = get_dist_info()
    paths = list_parquet_files(split)
    assert paths, "没找到 parquet，先跑 script/train_base.sh"

    pq_idx = resume_state["pq_idx"] if resume_state else 0
    rg_start = resume_state["rg_idx"] if resume_state else None
    epoch = resume_state.get("epoch", 1) if resume_state else 1
    first_pass = True

    while True:  # 无限循环：训练是多 epoch 的
        pq_idx = resume_state["pq_idx"] if (first_pass and resume_state) else 0
        while pq_idx < len(paths):
            pf = pq.ParquetFile(paths[pq_idx])
            if first_pass and rg_start is not None and pq_idx == resume_state["pq_idx"]:
                # 续训：从上次位置 +1 个 row group 接着读，避免重复训练
                rg_idx = (rg_start // world_size + 1) * world_size + rank
                rg_start = None  # 只在第一次 pass 用
            else:
                rg_idx = rank  # 多卡时每张卡读不同的 row group
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size], {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
                rg_idx += world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


# ---------------------------------------------------------------------------
# 阶段二：打包成定长 batch
# ---------------------------------------------------------------------------
def make_dataloader(tokenizer, batch_size: int, seq_len: int, split: str,
                    device="cuda", resume_state: dict | None = None,
                    buffer_size: int = 1000, tokenizer_threads: int = 4,
                    tokenizer_batch_size: int = 128):
    """
    产出 (inputs, targets, state)：
        inputs, targets 形状都是 (batch_size, seq_len)，dtype int64，在 device 上
        state               打包位置，用于续训

    ── 三个关键设计 ─────────────────────────────────────────────

    (1) 为什么是 seq_len + 1？
        要构造 inputs/targets 这一对，必须有 T+1 个 token：
            inputs  = row[0 : T]
            targets = row[1 : T+1]
        所以每行的容量是 T+1，其中 row[0] 固定是 <|bos|>。

    (2) 为什么全程不 padding？
        padding token 会让模型在推理时遇到从未见过的 id 分布。
        宁可裁掉 35% 的 token，也要保证 100% 利用率 + 无 padding。

    (3) 为什么用「先建 CPU row_buffer，再一次性搬到 GPU」？
        逐行 torch.tensor(...) 会产生几千次小 H2D 拷贝，每次都有
        几微秒的启动开销，累加起来能吃掉可观的时间。
        正确做法是：CPU 侧拼好一整批 -> 一次 non_blocking H2D。
    """
    row_capacity = seq_len + 1
    batches = iter_documents(split, tokenizer_batch_size, resume_state)
    bos = tokenizer.get_bos_token_id()

    # doc_buffer 存「已经 token 化、还没用掉」的文档
    doc_buffer: list[list[int]] = []
    pq_idx = rg_idx = 0
    epoch = 1

    def refill():
        """从 parquet 再取一批文本，token 化后塞进 doc_buffer。"""
        nonlocal pq_idx, rg_idx, epoch
        text_batch, st = next(batches)
        pq_idx, rg_idx, epoch = st["pq_idx"], st["rg_idx"], st["epoch"]
        for tokens in tokenizer.encode(text_batch, prepend=bos, num_threads=tokenizer_threads):
            doc_buffer.append(tokens)

    use_cuda = (device == "cuda")
    # CPU 侧的暂存区：一次性装下整批，pin_memory 让 H2D 走 DMA
    cpu_buffer = torch.empty(2 * batch_size * seq_len, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * batch_size * seq_len, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:batch_size * seq_len].view(batch_size, seq_len)
    cpu_targets = cpu_buffer[batch_size * seq_len:].view(batch_size, seq_len)
    inputs = gpu_buffer[:batch_size * seq_len].view(batch_size, seq_len)
    targets = gpu_buffer[batch_size * seq_len:].view(batch_size, seq_len)

    # 拼行时用的工作区，大小是一整行
    row = torch.empty(row_capacity, dtype=torch.long)

    while True:
        for r in range(batch_size):
            # 从 0 开始。第一个被放进来的文档自带 <|bos|>（refill 里 prepend 的），
            # 所以位置 0 自然就是 BOS，不需要额外手动写。
            # 之后每篇文档的 BOS 就成了「文档分隔符」—— 模型学会看到 BOS 就
            # 知道「新文章开始了」，这正是我们要的。
            pos = 0

            while pos < row_capacity:
                # 保证 buffer 里有足够的候选文档
                while len(doc_buffer) < buffer_size:
                    refill()

                remaining = row_capacity - pos

                # 步骤 2：best-fit —— 找「能完整放下」的最长文档
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    n = len(doc)
                    if n <= remaining and n > best_len:
                        best_idx, best_len = i, n
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row[pos:pos + best_len] = torch.tensor(doc, dtype=torch.long)
                    pos += best_len

                # 步骤 3：没有文档能完整放下 -> 裁剪「最短的」那个填满
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row[pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos = row_capacity

            cpu_inputs[r] = row[:-1]
            cpu_targets[r] = row[1:]

        state = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        # 一次 H2D，之后每次 yield 复用同一块 GPU 显存（零分配）
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state
