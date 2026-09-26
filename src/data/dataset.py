"""
数据集：下载预训练 shard，以及 SFT/评测任务的 parquet。

▓▓ 这一章要你敲的部分 ▓▓
    download_shards  —— 训练/验证 shard 的划分约定
    load_hub_dataset —— 手动替代 HuggingFace datasets

────────────────────────────────────────────────────────────────
卷1 的知识地图
────────────────────────────────────────────────────────────────
    ch01  环境与全景图  -> 本文件的网络镜像设置
    ch02  为什么需要分词器 -> download_shards 的 -n 参数
    ch06  bits per byte  -> 最后一个 shard 当验证集的约定
    卷7  推理与 SFT      -> load_hub_dataset
"""

import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor

from common import (
    HF_ENDPOINT,
    GITHUB_RAW_MIRROR,
    get_data_dir,
    get_task_dir,
    log0,
    download_file,
)

# ===========================================================================
# 🔶 规格部分
# ===========================================================================
# 预训练数据：karpathy/climbmix-400b-shuffle
# 实测每个 shard 的结构（92 MB 压缩包）：
#   84 个 row group，86,016 篇文档，单列 string
#   每个 row group 1024 篇，约 3 MB 未压缩
#   文档长度 min=4 / 中位数=628 / max=25055 token
PRETRAIN_REPO = os.environ.get("GPT_PRETRAIN_REPO", "karpathy/climbmix-400b-shuffle")
PRETRAIN_MAX_SHARD = 6542
SHARD_FILENAME = "shard_{:05d}.parquet"

TINY_SHAKESPEARE_URL = (
    f"{GITHUB_RAW_MIRROR}/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def shard_url(index: int) -> str:
    """拼出第 index 个 shard 的下载地址。注意用 HF_ENDPOINT 而不是硬编码。"""
    return (f"{HF_ENDPOINT}/datasets/{PRETRAIN_REPO}/resolve/main/"
            + SHARD_FILENAME.format(index))


# ===========================================================================
# ❗ 要你敲的部分 1
# ===========================================================================
def download_shards(num_shards: int, workers: int = 4) -> str:
    """
    下载前 num_shards 个训练 shard，外加 1 个永远作为验证集的 shard。

    ── 你要想清楚的两个点 ──────────────────────────────

    (1) 为什么训练要用 num_shards + 1 个 shard 而不是 num_shards 个？
        提示：最后一个 shard（PRETRAIN_MAX_SHARD）是**验证集**，
        不参与训练。这样 val 才是模型从未见过的数据。
        为什么要「固定最后一个」而不是「随机切 0.5%」？
        因为训练是多 epoch 的循环，随机切的 val 会被反复看到。

    (2) 为什么要用 ThreadPoolExecutor 并发下载？
        单个 shard 92 MB，实测 hf-mirror 速度 5-8 MB/s，
        串行下 4 个 shard 要 2 分钟，并发只要 30 秒。
        注意用 pool.map 并转成 list 才能真正等完（不消费就是惰性的）。
    """
    raise NotImplementedError(
        "待实现：download_shards ——\n"
        "  1) data_dir = get_data_dir()\n"
        "  2) indices = list(range(num_shards)) + [PRETRAIN_MAX_SHARD]\n"
        "  3) ThreadPoolExecutor(max_workers=workers) + pool.map(\n"
        "        lambda i: download_file(shard_url(i),\n"
        "                               os.path.join(data_dir, SHARD_FILENAME.format(i)),\n"
        "                               desc=f'shard_{i:05d}'),\n"
        "        indices)\n"
        "  4) list(...) 消费掉 map 结果（否则不会真正执行）\n"
        "  5) return data_dir\n"
        "参考实现：git show solution:src/data/dataset.py")


def list_parquet_files(split: str = "train") -> list[str]:
    """
    列出某个 split 的 parquet 路径，按文件名排序。

    约定：最后一个 shard 是 val，其余全是 train。
    这个约定被 download_shards / dataloader / evaluate 三处共用。
    """
    data_dir = get_data_dir()
    files = sorted(f for f in os.listdir(data_dir)
                   if f.endswith(".parquet") and not f.endswith(".tmp"))
    paths = [os.path.join(data_dir, f) for f in files]
    assert paths, (
        f"{data_dir} 里没有 parquet。先跑 `bash script/train_base.sh smoke`，"
        "它会自动下载。"
    )
    # ❗ train 要排除最后一个，val 只要最后一个 —— 你来写这一行
    raise NotImplementedError(
        "待实现：一行 —— 提示：return paths[:-1] if split == 'train' else paths[-1:]")


# ===========================================================================
# 🔶 规格部分：玩具数据集
# ===========================================================================
def get_base_dir_shakespeare() -> str:
    from common import get_base_dir
    return os.path.join(get_base_dir(), "toy")


def download_tiny_shakespeare() -> str:
    """
    卷2 起步用的玩具数据：1.1 MB 纯文本。
    存在的意义是「秒级拿到数据」，让你先专心看模型，别被下载卡住。
    """
    d = get_base_dir_shakespeare()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "input.txt")
    if not os.path.exists(path):
        log0("下载 tinyshakespeare（约 1.1 MB）...")
        download_file(TINY_SHAKESPEARE_URL, path, desc="tinyshakespeare/input.txt")
    return path


# ===========================================================================
# ❗ 要你敲的部分 2
# ===========================================================================
def load_hub_dataset(repo_id: str, subset: str = "default", split: str = "train",
                     max_shards: int = None):
    """
    极简版 load_dataset，替代 HuggingFace datasets。

    流程：调 Hub 的 parquet 导出 API 列出分片 -> 下载 -> 用 pyarrow 读 -> concat。

    ── 你要想清楚的四个点 ──────────────────────────────

    (1) 为什么要自己写，而不用 datasets？
        完整版 datasets 有 30+ 个传递依赖。nanochat 把它整个删了
        （commit: "delete datasets dependency bye"）。我们只需要
        len / __getitem__ / shuffle 三个方法，几十行就够，
        还能顺便看清「打乱」到底怎么实现的。

    (2) 下载完成的标记怎么设计？
        提示：不要用「文件存在」判断（可能只下了一半）。
        要写一个 manifest.json **最后**写 —— 它的存在就代表下载完成。

    (3) 国内网络要绕哪两个坑？
        a) 必须显式带 User-Agent，否则镜像返回 403
           （Python 内置 urllib 的默认 UA 会被拒）
        b) 镜像返回的分片 URL 指向 huggingface.co，而那个域名本机不可达，
           **必须重写到 HF_ENDPOINT**

    (4) max_shards 是干什么的？
        SmolTalk 有 9 片共约 2 GB。SFT 只需要一小部分数据，
        smoke 档只取 1 片（224 MB）就够，省 4 分钟下载。
    """
    raise NotImplementedError(
        "待实现：load_hub_dataset ——\n"
        "  1) slug = repo_id.replace('/', '--'); d = os.path.join(get_task_dir(), slug, subset, split)\n"
        "     manifest = os.path.join(d, 'manifest.json')\n"
        "  2) if not os.path.exists(manifest):\n"
        "       os.makedirs(d, exist_ok=True)\n"
        "       requests.get(f'{HF_ENDPOINT}/api/datasets/{repo_id}/parquet/{subset}/{split}',\n"
        "                    headers={'User-Agent': 'python-requests/gpt-base'}).json()\n"
        "       逐个 download_file(u.replace('https://huggingface.co', HF_ENDPOINT), ...)\n"
        "       最后 json.dump(names, open(manifest,'w'))\n"
        "  3) 读回 names，若 max_shards 不为 None 则截断\n"
        "  4) pq.read_table 每个文件 -> pa.concat_tables -> HubDataset(table)\n"
        "  提示：HubDataset 类在 data/tasks.py 里\n"
        "参考实现：git show solution:src/data/dataset.py")


# ===========================================================================
# 🔶 规格部分：CLI
# ===========================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="下载数据集")
    p.add_argument("-n", "--num-shards", type=int, default=-1,
                   help="训练 shard 数量（-1 = 全部 6542 个）")
    p.add_argument("-w", "--workers", type=int, default=4, help="并发下载数")
    p.add_argument("--toy", action="store_true", help="只下载 tinyshakespeare")
    args = p.parse_args()

    if args.toy:
        download_tiny_shakespeare()
    if args.num_shards != -1:
        download_shards(args.num_shards, args.workers)
