"""
Checkpoint 的存与取。

刻意做成「扁平的多文件」而不是 nanochat 那种分 rank 的 ZeRO 存档：
单卡项目不需要优化器状态分片。但保留了几个 nanochat 的好设计：
  · 存档目录按 model_tag 分（d4 / d6 / d12 ...），不同实验互不干扰
  · metadata 存成 JSON 而不是 pickle —— 可以直接 cat 出来看
  · 记录完整 run_config，这样「这个模型是怎么训出来的」永远不会丢
"""

import os
import re
import json
import math

import torch

from common import log0


def _model_path(ckpt_dir: str, step: int) -> str:
    return os.path.join(ckpt_dir, f"model_{step:06d}.pt")


def _meta_path(ckpt_dir: str, step: int) -> str:
    return os.path.join(ckpt_dir, f"meta_{step:06d}.json")


def _json_safe(obj):
    """把 meta 里的非有限浮点（inf / nan）换成 None。

    为什么要这一步：`best_val_bpb` 在「还没验证过」时是 float('inf')。
    json.dump 默认会写成 `Infinity` —— 那是**非标准 JSON**，
    Python 自己读得回来，但 jq 和其它语言的解析器会直接报错。
    而这个文件的卖点正是「存成 JSON 可以直接 cat 出来看」。

    Python 把 null 读回 None，调用方按 None 判断「还没有值」即可。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_checkpoint(ckpt_dir: str, step: int, model_state: dict,
                    meta: dict) -> None:
    """
    保存：模型权重（.pt）+ 元信息（.json）。

    为什么要 .tmp + rename？
      断电/中断时如果直接写目标文件，会留下一个半截的 .pt，
      下次 resume 就 load 不出来。rename 是同文件系统内的原子操作。
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    mp = _model_path(ckpt_dir, step)
    torch.save(model_state, mp + ".tmp")
    os.replace(mp + ".tmp", mp)
    with open(_meta_path(ckpt_dir, step), "w", encoding="utf-8") as f:
        # allow_nan=False：万一还有漏网的非有限值，宁可报错也不写脏 JSON
        json.dump(_json_safe(meta), f, indent=2, default=str, allow_nan=False)
    log0(f"  已存档 step={step} -> {mp}")


def load_checkpoint(ckpt_dir: str, step: int, device, kind: str = "model"):
    """
    载入。kind:
        "model" -> 返回 state_dict
        "meta"  -> 返回 metadata dict
    """
    if kind == "model":
        path = _model_path(ckpt_dir, step)
        assert os.path.exists(path), f"找不到存档 {path}"
        return torch.load(path, map_location=device, weights_only=True)
    if kind == "meta":
        path = _meta_path(ckpt_dir, step)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"未知的 kind: {kind}")


def find_latest(ckpt_dir: str):
    """找出最大的 step 存档号；没有则返回 None。"""
    if not os.path.isdir(ckpt_dir):
        return None
    steps = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"model_(\d{6})\.pt", f)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def load_latest(run_root: str, tag: str, device):
    """
    给评测 / SFT / 对话脚本用的便捷入口：
    找到 <run_root>/base_checkpoints/<tag> 里最新的存档并加载。

    返回 (model, tokenizer, meta) —— meta 里带 model_config，
    调用方可以用它重建出完全一样的模型。
    """
    from common.config import ModelConfig
    from data.tokenizer import get_tokenizer

    ckpt_dir = os.path.join(run_root, "base_checkpoints", tag)
    step = find_latest(ckpt_dir)
    assert step is not None, (
        f"{ckpt_dir} 里没有存档。先跑 `bash script/train_base.sh <mode>`。"
    )
    meta = load_checkpoint(ckpt_dir, step, "cpu", "meta")
    # ModelConfig 有嵌套的默认值，用 filter 丢掉不认识的新字段
    fields = set(ModelConfig.__dataclass_fields__)
    cfg_dict = {k: v for k, v in meta["model_config"].items() if k in fields}
    cfg = ModelConfig(**cfg_dict)

    # 复用 build_model 的 meta device 流程，然后用存档覆盖
    from model.gpt import build_model
    model = build_model(cfg, device=device)
    model.load_state_dict(load_checkpoint(ckpt_dir, step, device, "model"))
    return model, get_tokenizer(), meta
