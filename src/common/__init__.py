"""
公共设施：设备、精度、日志、路径、网络镜像、硬件参数表。

这一层不含任何 GPT 逻辑，只解决「跑起来」的问题。
教程卷0 会逐段解释这里的每个决定。
"""

from common.utils import (  # noqa: F401
    # 网络
    HF_ENDPOINT,
    GITHUB_RAW_MIRROR,
    # 路径
    get_base_dir,
    get_data_dir,
    get_tokenizer_dir,
    get_task_dir,
    get_runs_dir,
    # 设备与精度
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    autodetect_device_type,
    compute_init,
    compute_cleanup,
    get_dist_info,
    synchronize,
    get_max_memory,
    # 日志
    logger,
    log0,
    setup_logging,
    # 硬件参数
    get_peak_flops,
    get_peak_bandwidth,
    # 工具
    download_file,
    human_time,
)

# checkpoint 相关（延迟导入以避免 common -> model 的循环依赖）
from common.checkpoint import (  # noqa: E402
    save_checkpoint,
    load_checkpoint,
    find_latest,
    load_latest,
)
