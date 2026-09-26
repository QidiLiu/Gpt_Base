"""
公共设施：设备、精度、日志、路径、网络镜像、硬件参数表。

这一层不含任何 GPT 逻辑，只解决「跑起来」的问题。
教程卷0 会逐段解释这里的每个决定。
"""

import os
import re
import sys
import time
import logging

import torch

# ---------------------------------------------------------------------------
# 0. 网络镜像
# ---------------------------------------------------------------------------
# 实测结论（2026-09，本机环境）：
#   pypi.org                      不可达   -> 已在 ~/.config/uv/uv.toml 配腾讯镜像
#   huggingface.co                不可达   -> 必须走 hf-mirror.com
#   raw.githubusercontent.com      不可达   -> 必须走 ghproxy.net
# 所以所有网络访问都从这里取 base url，不在业务代码里硬编码。
# 想用官方源时：export HF_ENDPOINT=https://huggingface.co

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
GITHUB_RAW_MIRROR = os.environ.get(
    "GITHUB_RAW_MIRROR",
    "https://ghproxy.net/https://raw.githubusercontent.com",
)
# 必须在 import huggingface_hub / 使用其默认端点之前设置
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)


# ---------------------------------------------------------------------------
# 1. 路径
# ---------------------------------------------------------------------------
def get_base_dir() -> str:
    """
    所有数据、tokenizer、checkpoint 的根目录。

    默认放 ~/.cache/gpt_base，与系统缓存目录约定一致，不污染项目目录。
    想换位置：export GPT_BASE_DIR=/some/path
    """
    d = os.environ.get("GPT_BASE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "gpt_base"
    )
    os.makedirs(d, exist_ok=True)
    return d


def get_data_dir() -> str:
    """预训练 parquet shard 目录。最后一个 shard 固定作为验证集（见 data/dataset.py）。"""
    d = os.path.join(get_base_dir(), "data_climbmix")
    os.makedirs(d, exist_ok=True)
    return d


def get_tokenizer_dir() -> str:
    d = os.path.join(get_base_dir(), "tokenizer")
    os.makedirs(d, exist_ok=True)
    return d


def get_task_dir() -> str:
    """SFT / 评测任务的 parquet 目录。"""
    d = os.path.join(get_base_dir(), "task_data")
    os.makedirs(d, exist_ok=True)
    return d


def get_runs_dir() -> str:
    """训练产物：checkpoint、日志、曲线图。默认放在项目内，方便直接看。"""
    d = os.path.join(os.environ.get("GPT_RUNS_DIR", "runs"))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 2. 设备与精度
# ---------------------------------------------------------------------------
def get_dist_info() -> tuple[bool, int, int, int]:
    """
    返回 (is_ddp, rank, local_rank, world_size)。

    本项目只有单卡，所以恒为 (False, 0, 0, 1)。
    保留这个函数是为了让 optim/muon.py 里的分布式分支在单卡下
    走「退化路径」而不是「不存在的路径」—— 教程卷6 会专门讲这件事。
    """
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return (True, int(os.environ["RANK"]),
                int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]))
    return False, 0, 0, 1


def autodetect_device_type() -> str:
    """按 cuda -> mps -> cpu 的优先级自动选设备。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _detect_compute_dtype():
    """
    决定「计算精度」—— 矩阵乘和激活值用什么 dtype。

    为什么不直接用 torch.amp.autocast？
      autocast 靠 context manager 隐式改变算子行为，很难追踪到底哪些算子
      跑在低精度上，出问题时不好定位。
      本项目采用 nanochat 的做法：权重永远存 fp32（优化器需要），
      在自定义的 Linear.forward 里显式把权重转成 COMPUTE_DTYPE 再做矩阵乘。
      精度控制点只有一处，可读、可调试。
    """
    override = os.environ.get("GPT_COMPUTE_DTYPE")
    if override:
        return _DTYPE_MAP[override], f"环境变量指定 GPT_COMPUTE_DTYPE={override}"
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) >= (8, 0):
            return torch.bfloat16, f"自动检测：CUDA SM{major}{minor}，原生支持 bf16"
        return torch.float32, f"自动检测：CUDA SM{major}{minor}，Ampere 之前无 bf16，退回 fp32"
    return torch.float32, "自动检测：非 CUDA（CPU / MPS），使用 fp32"


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()


def compute_init(device_type: str = "cuda"):
    """
    统一的初始化入口。返回 (ddp, rank, local_rank, world_size, device)。

    本项目只有 1 张卡，所以 ddp 恒为 False、world_size 恒为 1。
    但保留这套返回值的形状，是为了让你读 optim/muon.py 时，
    能看清「分布式分支在单卡下是怎么退化的」（教程卷6）。
    """
    assert device_type in ("cuda", "mps", "cpu"), f"非法 device_type: {device_type}"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "指定了 cuda 但 torch.cuda 不可用"

    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
        # 允许 tf32：fp32 矩阵乘也能走 tensor core
        torch.set_float32_matmul_precision("high")

    device = torch.device(device_type if device_type != "cuda" else "cuda:0")
    return False, 0, 0, 1, device


def compute_cleanup():
    pass


def synchronize(device_type: str = "cuda"):
    """计时前后必须同步，否则测到的是「kernel 入队时间」而不是「真正算完的时间」。"""
    return torch.cuda.synchronize if device_type == "cuda" else (lambda: None)


def get_max_memory(device_type: str = "cuda"):
    """
    返回一个「取当前峰值显存（字节）」的函数。
    非 CUDA 平台恒返回 0。用法：get_max_memory(device_type)()
    """
    return torch.cuda.max_memory_allocated if device_type == "cuda" else (lambda: 0)


# ---------------------------------------------------------------------------
# 3. 日志
# ---------------------------------------------------------------------------
class _ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        level = record.levelname
        if level in self.COLORS:
            record.levelname = (
                f"{self.COLORS[level]}{self.BOLD}{level}{self.RESET}"
            )
        msg = super().format(record)
        if level == "INFO":
            # 把数字高亮，方便在长日志里一眼扫到关键指标
            msg = re.sub(r"(\d+\.?\d*\s*(?:GB|MB|MiB|%|tok/s|it/s))", rf"{self.BOLD}\1{self.RESET}", msg)
        return msg


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColoredFormatter("%(asctime)s - %(message)s", "%H:%M:%S"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    return logging.getLogger("gpt_base")


logger = setup_logging()


def log0(msg="", **kw):
    """只在主进程打印。单卡下就是普通 print，多卡下避免刷屏。"""
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(msg, **kw)


# ---------------------------------------------------------------------------
# 4. 硬件参数表（用于 MFU / MBU）
# ---------------------------------------------------------------------------
# 注意：GeForce 卡在 bf16/fp16 矩阵乘时，**FP32 累加是 FP16 累加的一半速度**。
# PyTorch 默认用 FP32 累加，所以这里登记的是「累加后的真实峰值」，
# 而不是厂商宣传的 FP16-accumulate 数字。RTX 4060 Ti 官方 165.2 TFLOPS
# 是 FP16-accumulate 的值，训练时实际约 82.6 TFLOPS。
_PEAK_FLOPS = {
    # Blackwell
    "gb200": 2.5e15, "b200": 2.25e15, "b100": 1.8e15,
    # Hopper
    "h200": 989e12, "h100": 989e12, "h800": 989e12,
    # Ampere
    "a100": 312e12, "a800": 312e12, "a40": 149.7e12, "a30": 165e12,
    # Ada
    "l40s": 362e12, "l4": 121e12,
    "4090": 165.2e12,
    "4060 ti": 82.6e12,   # 165.2 / 2，fp32 累加减半
    "5090": 209.5e12,
    # Ampere 消费卡
    "3090": 71e12,
}

_PEAK_BANDWIDTH = {  # 字节/秒
    "h200": 4.8e12, "h100": 3.35e12, "a100": 2.0e12, "a800": 2.0e12,
    "l40s": 864e9, "l4": 300e9,
    "4090": 1.01e12,
    "4060 ti": 288e9,     # GDDR6X 18Gbps × 256bit / 8
    "5090": 1.79e12,
    "3090": 936e9,
}


def _lookup(table: dict, device_name: str, default: float, unit: str):
    name = device_name.lower()
    for key, val in table.items():
        if key in name:
            return val
    logger.warning(f"未登记的 GPU「{device_name}」，{unit} 视为 {default}")
    return default


def get_peak_flops(device_name: str) -> float:
    return _lookup(_PEAK_FLOPS, device_name, float("inf"), "峰值算力")


def get_peak_bandwidth(device_name: str) -> float:
    return _lookup(_PEAK_BANDWIDTH, device_name, float("inf"), "峰值带宽")


# ---------------------------------------------------------------------------
# 5. 下载工具
# ---------------------------------------------------------------------------
def download_file(url: str, dest: str, desc: str = "", max_retries: int = 5) -> str:
    """
    带重试的下载，下载到 .tmp 再 rename，保证原子性。

    为什么要 .tmp + rename？
      多进程（torchrun 8 卡）会同时下载同一个文件。直接写目标文件的话，
      另一个进程可能读到写了一半的内容。rename 在同一文件系统内是原子的。
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    import requests

    # 目标文件所在的目录可能还不存在（例如第一次下载玩具数据）
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp = dest + f".tmp"
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                t0 = time.time()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1MB
                        f.write(chunk)
                        done += len(chunk)
                speed = done / max(time.time() - t0, 1e-6) / 1e6
                msg = f"下载完成 {desc or dest}"
                if total:
                    msg += f"（{total/1e6:.1f} MB, {speed:.1f} MB/s）"
                log0(msg)
            os.rename(tmp, dest)
            return dest
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            log0(f"下载失败 {desc or url}（第 {attempt}/{max_retries} 次）：{e}，{wait}s 后重试")
            time.sleep(wait)
    return dest


def human_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"
