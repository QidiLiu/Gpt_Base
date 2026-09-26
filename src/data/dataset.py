"""
数据集：下载预训练 shard，以及 SFT/评测任务的 parquet。

教程卷1 会解释：
  - 为什么最后一个 shard 固定当验证集
  - 为什么不用 HuggingFace datasets（nanochat 把它删了，本项目也删）
  - 为什么下载要 .tmp + rename（多卡并发安全）
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

# ---------------------------------------------------------------------------
# 预训练数据：karpathy/climbmix-400b-shuffle
# ---------------------------------------------------------------------------
# 实测每个 shard 的结构（92 MB 压缩包）：
#   84 个 row group，86,016 篇文档，单列 string
#   每个 row group 1024 篇，约 3 MB 未压缩
#   文档长度 min=11 / 中位数≈2400 / max≈110000 字符
PRETRAIN_REPO = os.environ.get("GPT_PRETRAIN_REPO", "karpathy/climbmix-400b-shuffle")
PRETRAIN_MAX_SHARD = 6542
SHARD_FILENAME = "shard_{:05d}.parquet"

# 玩具数据集：tinyshakespeare（1.1 MB 纯文本），卷2 起步用
TINY_SHAKESPEARE_URL = (
    f"{GITHUB_RAW_MIRROR}/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def shard_url(index: int) -> str:
    return (
        f"{HF_ENDPOINT}/datasets/{PRETRAIN_REPO}/resolve/main/"
        + SHARD_FILENAME.format(index)
    )


def download_shards(num_shards: int, workers: int = 4) -> str:
    """
    下载前 num_shards 个训练 shard，外加 1 个永远作为验证集的 shard。

    为什么固定「最后一个 shard」当 val？
      因为训练是无限循环的（多 epoch），如果 val 从训练集里随机切，
      同一个 val 会被反复看到。单独留一个 shard 不参与训练，才是最干净的 val。
    """
    data_dir = get_data_dir()
    indices = list(range(min(num_shards, PRETRAIN_MAX_SHARD)))
    indices.append(PRETRAIN_MAX_SHARD)  # 验证 shard，永远最后下载

    log0(f"需要 {len(indices)} 个 shard（{num_shards} 训练 + 1 验证）-> {data_dir}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(
            lambda i: download_file(shard_url(i),
                                    os.path.join(data_dir, SHARD_FILENAME.format(i)),
                                    desc=f"shard_{i:05d}"),
            indices,
        ))
    return data_dir


def list_parquet_files(split: str = "train") -> list[str]:
    """
    列出某个 split 的 parquet 路径，按文件名排序。

    约定：最后一个 shard 是 val，其余全是 train。
    这个约定在 download_shards / dataloader / evaluate 三处共用。
    """
    data_dir = get_data_dir()
    files = sorted(f for f in os.listdir(data_dir)
                   if f.endswith(".parquet") and not f.endswith(".tmp"))
    paths = [os.path.join(data_dir, f) for f in files]
    assert paths, (
        f"{data_dir} 里没有 parquet。先跑 `bash script/train_base.sh smoke`，"
        "它会自动下载。"
    )
    return paths[:-1] if split == "train" else paths[-1:]


def download_tiny_shakespeare() -> str:
    """
    卷2 的起步数据集：1.1 MB 纯文本。
    存在的意义是「秒级拿到数据」，让你先专心看模型，别被下载卡住。
    """
    d = os.path.join(get_base_dir_shakespeare(), "")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "input.txt")
    if not os.path.exists(path):
        log0("下载 tinyshakespeare（约 1.1 MB）...")
        download_file(TINY_SHAKESPEARE_URL, path, desc="tinyshakespeare/input.txt")
    return path


def get_base_dir_shakespeare() -> str:
    from common import get_base_dir
    return os.path.join(get_base_dir(), "toy")


# ---------------------------------------------------------------------------
# 任务数据（SFT / 评测）：手动替代 HuggingFace datasets
# ---------------------------------------------------------------------------
def load_hub_dataset(repo_id: str, subset: str = "default", split: str = "train"):
    """
    极简版 load_dataset。

    HuggingFace 的每个数据集在 Hub 上都有自动生成的 parquet 导出，
    我们只要：调 API 列出 shard 列表 -> 下载 -> 用 pyarrow 读。
    对本项目需要的规模，这已经够了，而且少一个巨型依赖。

    返回一个带 __len__ / __getitem__ / shuffle 的轻量对象（见 data/tasks.py）。
    """
    from data.tasks import HubDataset  # 延迟导入避免循环依赖
    import pyarrow as pa
    import pyarrow.parquet as pq
    import urllib.request

    slug = repo_id.replace("/", "--")
    shards_dir = os.path.join(get_task_dir(), slug, subset, split)
    manifest_path = os.path.join(shards_dir, "manifest.json")

    # manifest 最后写，它存在就代表下载完了 —— 这是比加锁更省事的完成标记
    if not os.path.exists(manifest_path):
        os.makedirs(shards_dir, exist_ok=True)
        log0(f"下载任务数据集 {repo_id} [{subset}/{split}] ...")
        url = f"{HF_ENDPOINT}/api/datasets/{repo_id}/parquet/{subset}/{split}"
        with urllib.request.urlopen(url, timeout=60) as r:
            shard_urls = json.loads(r.read())
        names = []
        for i, su in enumerate(shard_urls):
            name = f"{i:05d}.parquet"
            download_file(su, os.path.join(shards_dir, name), desc=f"{repo_id}/{name}")
            names.append(name)
        with open(manifest_path, "w") as f:
            json.dump(names, f)

    with open(manifest_path) as f:
        names = json.load(f)
    tables = [pq.read_table(os.path.join(shards_dir, n)) for n in names]
    return HubDataset(pa.concat_tables(tables))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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
