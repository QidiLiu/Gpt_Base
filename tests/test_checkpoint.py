"""
`src/common/checkpoint.py` 的正确性测试（卷5 只读代码的回归护栏）。

背景：续训路径长期只恢复模型权重、丢弃全部 meta，导致
`dataloader_state` / `best_val_bpb` / `history` 三样写了不读。
本文件把 checkpoint 的存取契约钉住，让「续训」名副其实。

跑法：uv run pytest tests/test_checkpoint.py -v
"""

import json
import math
import os
import pathlib

import pytest
import torch

from common.checkpoint import (
    save_checkpoint, load_checkpoint, find_latest, load_latest,
)


# ===========================================================================
# 基础往返
# ===========================================================================
def test_save_load_model_roundtrip(tmp_path):
    d = str(tmp_path)
    state = {"w": torch.randn(4, 4), "b": torch.zeros(4)}
    save_checkpoint(d, 7, state, {"step": 7})
    got = load_checkpoint(d, 7, "cpu", "model")
    assert set(got) == set(state)
    assert torch.equal(got["w"], state["w"])


def test_save_load_meta_roundtrip(tmp_path):
    d = str(tmp_path)
    meta = {"model_config": {"n_layer": 4, "n_embd": 128},
            "run_config": {"mode": "smoke"}, "step": 12,
            "dataloader_state": {"pq_idx": 1, "rg_idx": 40, "epoch": 2},
            "history": [{"step": 1, "train_loss": 9.0},
                        {"step": 2, "val_bpb": 1.7}]}
    save_checkpoint(d, 12, {"w": torch.zeros(2)}, meta)
    got = load_checkpoint(d, 12, "cpu", "meta")
    for k, v in meta.items():
        assert got[k] == v, f"meta[{k!r}] 往返不一致：{got[k]!r} != {v!r}"


def test_load_checkpoint_rejects_unknown_kind(tmp_path):
    d = str(tmp_path)
    save_checkpoint(d, 1, {"w": torch.zeros(1)}, {})
    with pytest.raises(ValueError, match="未知的 kind"):
        load_checkpoint(d, 1, "cpu", "optimizer")


def test_load_missing_checkpoint_asserts(tmp_path):
    with pytest.raises(AssertionError, match="找不到存档"):
        load_checkpoint(str(tmp_path), 999, "cpu", "model")


# ===========================================================================
# find_latest
# ===========================================================================
def test_find_latest_empty_dir_returns_none(tmp_path):
    assert find_latest(str(tmp_path)) is None
    assert find_latest(str(tmp_path / "nope")) is None


def test_find_latest_picks_max_step(tmp_path):
    d = str(tmp_path)
    for step in (5, 120, 3, 896):
        save_checkpoint(d, step, {"w": torch.zeros(1)}, {"step": step})
    assert find_latest(d) == 896


def test_find_latest_ignores_non_conforming_names(tmp_path):
    """只认 model_XXXXXX.pt —— 位数不对的、别的前缀的都不该被当成存档。"""
    d = pathlib.Path(tmp_path)
    save_checkpoint(str(d), 10, {"w": torch.zeros(1)}, {"step": 10})
    for junk in ("model_12.pt", "model_1234567.pt", "optimizer_000010.pt",
                 "model_000010.pt.tmp", "model_abcdeff.pt", "notes.txt"):
        (d / junk).write_text("x")
    assert find_latest(str(d)) == 10, "位数不对/前缀不对的文件不该被认成存档"


# ===========================================================================
# JSON 必须是标准 JSON（P1#1 顺带发现的问题）
# ===========================================================================
def test_meta_json_is_standard_json(tmp_path):
    """best_val_bpb 为 inf 时不能写出 Infinity。

    `Infinity` 是非标准 JSON：Python 读得回来，但 jq / 其它语言的解析器会报错，
    而这个文件的卖点正是「存成 JSON 可以直接 cat 出来看」。
    """
    d = pathlib.Path(tmp_path)
    save_checkpoint(str(d), 1, {"w": torch.zeros(1)},
                    {"best_val_bpb": float("inf"), "step": 1})
    text = (d / "meta_000001.json").read_text(encoding="utf-8")
    assert "Infinity" not in text, f"JSON 里出现了非标准的 Infinity：{text[:200]}"
    assert "NaN" not in text

    def strict(x):
        raise ValueError(f"非标准 JSON 常量：{x}")

    parsed = json.loads(text, parse_constant=strict)   # 严格模式
    assert parsed["best_val_bpb"] is None, "inf 应被存成 null"


def test_meta_json_keeps_finite_values(tmp_path):
    d = pathlib.Path(tmp_path)
    save_checkpoint(str(d), 1, {"w": torch.zeros(1)},
                    {"best_val_bpb": 1.7136, "step": 1})
    text = (d / "meta_000001.json").read_text(encoding="utf-8")
    assert json.loads(text)["best_val_bpb"] == 1.7136, "正常值不该被改动"


def test_nan_in_meta_does_not_corrupt_json(tmp_path):
    d = pathlib.Path(tmp_path)
    save_checkpoint(str(d), 1, {"w": torch.zeros(1)}, {"bad": float("nan")})
    got = load_checkpoint(str(d), 1, "cpu", "meta")
    assert got["bad"] is None, "nan 应被存成 null"


# ===========================================================================
# 续训三元组（P1#1 的核心）
# ===========================================================================
def test_resume_triple_roundtrips(tmp_path):
    """dataloader_state / best_val_bpb / history 三样必须能原样存回读出。

    这正是续训时 train_base 要恢复的东西。之前它们只写不读，
    「续训」名不副实：数据流从头重放、best 指标重置、曲线断成两截。
    """
    d = str(tmp_path)
    dl_state = {"pq_idx": 2, "rg_idx": 57, "epoch": 1}
    history = [{"step": 1, "train_loss": 9.01},
               {"step": 1, "val_bpb": 1.90},
               {"step": 2, "train_loss": 8.5},
               {"step": 2, "val_bpb": 1.71}]
    save_checkpoint(d, 2, {"w": torch.zeros(1)}, {
        "step": 2, "best_val_bpb": 1.71,
        "dataloader_state": dl_state, "history": history})

    meta = load_checkpoint(d, 2, "cpu", "meta")
    assert meta["dataloader_state"] == dl_state
    assert meta["best_val_bpb"] == 1.71
    assert meta["history"] == history, "曲线数据应逐条一致，不能只恢复最后一条"


def test_resume_with_unset_best_bpb_reads_back_as_none(tmp_path):
    """还没验证过时存的 inf -> 读回是 null，调用方据此还原成 inf。"""
    d = str(tmp_path)
    save_checkpoint(d, 1, {"w": torch.zeros(1)}, {"best_val_bpb": float("inf")})
    meta = load_checkpoint(d, 1, "cpu", "meta")
    assert meta["best_val_bpb"] is None
    # train_base 里的还原写法
    _bpb = meta.get("best_val_bpb")
    restored = float("inf") if _bpb is None else float(_bpb)
    assert restored == float("inf")


# ===========================================================================
# 原子写
# ===========================================================================
def test_save_leaves_no_tmp_files(tmp_path):
    d = pathlib.Path(tmp_path)
    save_checkpoint(str(d), 3, {"w": torch.zeros(1)}, {"step": 3})
    leftovers = [p.name for p in d.iterdir() if p.name.endswith(".tmp")]
    assert not leftovers, f"存档后残留 .tmp 文件：{leftovers}"


def test_model_file_loads_with_weights_only(tmp_path):
    """存档用 weights_only=True 载入，不该需要 pickle 信任。"""
    d = str(tmp_path)
    save_checkpoint(d, 1, {"w": torch.randn(3)}, {"step": 1})
    got = load_checkpoint(d, 1, "cpu", "model")
    assert got["w"].shape == (3,)


# ===========================================================================
# load_latest
# ===========================================================================
def test_load_latest_asserts_when_no_checkpoint(tmp_path):
    with pytest.raises(AssertionError, match="没有存档"):
        load_latest(str(tmp_path), "d99", "cpu")
