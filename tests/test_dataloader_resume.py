"""
`iter_documents` 的 resume_state 行为测试（卷1 第 05 章手抄目标的护栏）。

为什么单独一个文件：这个行为在 2026-09 之前**从未被执行过** ——
`make_dataloader(resume_state=...)` 没有任何调用方，
所以 train_base 存进 checkpoint 的 `dataloader_state` 写了永远不读，
「精确续训」只是 docstring 里的一句话。

P1#1 把续训接上之后，这条路径第一次真正会跑，所以必须有判据守着：
否则一个 off-by-one 就会让续训悄悄重复训练已见过的数据，而且不报错。

跑法：uv run pytest tests/test_dataloader_resume.py -v
"""

import types

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


SCHEMA = pa.schema([("text", pa.string())])


def _make_shard(path, n_row_groups, rows_per_group=2, tag=""):
    """写一个 shard：第 i 个 row group 的内容是 f"{tag}RG{i}_*"，各不相同。"""
    w = pq.ParquetWriter(str(path), SCHEMA)
    for i in range(n_row_groups):
        rows = [f"{tag}RG{i}_{j}" for j in range(rows_per_group)]
        w.write_table(pa.table({"text": pa.array(rows)}, schema=SCHEMA))
    w.close()
    return str(path)


@pytest.fixture
def shard(tmp_path):
    return _make_shard(tmp_path / "shard_00000.parquet", n_row_groups=5)


@pytest.fixture
def two_shards(tmp_path):
    a = _make_shard(tmp_path / "shard_00000.parquet", 5, tag="A")
    b = _make_shard(tmp_path / "shard_00001.parquet", 2, tag="B")
    return [a, b]


def _iter(mod, paths, **kw):
    """把 list_parquet_files 换掉，其余走真实实现。"""
    mod.list_parquet_files = lambda split="train": paths
    return mod.iter_documents("train", tokenizer_batch_size=16, **kw)


# ===========================================================================
@pytest.fixture
def fake_files(monkeypatch):
    """返回一个函数：设置 list_parquet_files 并返回 iter_documents。"""
    def _make(paths, **kw):
        from data import dataloader as D
        monkeypatch.setattr(D, "list_parquet_files",
                            lambda split="train": paths)
        return D.iter_documents("train", tokenizer_batch_size=16, **kw)
    return _make


def test_without_resume_starts_at_row_group_0(shard, fake_files):
    it = fake_files([shard])
    batch, state = next(it)
    assert batch == ["RG0_0", "RG0_1"], f"应从第 0 个 row group 开始，得到 {batch}"
    assert state["rg_idx"] == 0
    assert state["pq_idx"] == 0
    # 参考实现里 epoch 是 1 起的（`epoch = 1` 而不是 0）
    assert state["epoch"] == 1


@pytest.mark.parametrize("resume_rg,expect", [
    (0, "RG1_0"),    # ★ 关键：往前跳 1 个，不能重复喂已训过的 RG0
    (1, "RG2_0"),
    (2, "RG3_0"),
    (3, "RG4_0"),
])
def test_resume_skips_exactly_one_row_group(shard, fake_files, resume_rg, expect):
    """给定 resume_state，必须从 rg_idx + 1 开始（而不是从 rg_idx 重来）。

    这就是 dataloader.py docstring 里「往前跳 1 个」那条坑：
    上报的 state 是「下一个要读的」，所以恢复时要 +1，否则每次续训
    都会重复训练最后一个 row group。
    """
    it = fake_files([shard], resume_state={"pq_idx": 0, "rg_idx": resume_rg,
                                           "epoch": 1})
    batch, state = next(it)
    assert batch[0] == expect, (
        f"resume rg_idx={resume_rg} 应从 {expect} 开始，实际 {batch}")
    assert state["rg_idx"] == resume_rg + 1, "上报的应是下一个要读的 rg"
    assert state["epoch"] == 1, "epoch 应原样透传"


def test_resume_advances_state_monotonically(shard, fake_files):
    """连续取批，state 必须一路前进，不能原地打转。"""
    it = fake_files([shard])
    seen = []
    for _ in range(5):
        batch, state = next(it)
        seen.append((batch[0], (state["pq_idx"], state["rg_idx"])))
    for (text, (pq, rg)), (nxt_text, (nq, nrg)) in zip(seen, seen[1:]):
        assert nrg > rg or nq > pq, f"state 没有前进：{(pq, rg)} -> {(nq, nrg)}"


def test_resume_crosses_file_boundary(two_shards, fake_files):
    """pq_idx 指向最后一个 shard 时，应正确翻到该文件并跳过 1 个 rg。"""
    it = fake_files(two_shards, resume_state={"pq_idx": 1, "rg_idx": 0,
                                              "epoch": 1})
    batch, state = next(it)
    assert batch[0] == "BRG1_0", f"应从 B 文件的第 1 个 rg 开始，得到 {batch}"
    assert state["pq_idx"] == 1


def test_resume_state_keys_are_complete(shard, fake_files):
    """state 必须带齐 pq_idx / rg_idx / epoch —— 缺一个续训就不精确。"""
    _, state = next(fake_files([shard]))
    assert set(state) >= {"pq_idx", "rg_idx", "epoch"}, (
        f"state 字段不全：{state}")


def test_batches_are_sliced_to_tokenizer_batch_size(tmp_path, monkeypatch):
    """一批的条数不应超过 tokenizer_batch_size。

    ⚠ 必须用 monkeypatch 而不是 `mod.list_parquet_files = ...` 直接赋值：
      直接赋值会**永久**改掉模块属性，fixture 结束时不会还原，
      污染同一 session 里后续的所有测试（曾经真的发生过）。
    """
    p = _make_shard(tmp_path / "shard_00000.parquet", n_row_groups=3,
                    rows_per_group=10, tag="S")
    from data import dataloader as D
    monkeypatch.setattr(D, "list_parquet_files", lambda split="train": [p])
    it = D.iter_documents("train", tokenizer_batch_size=4)
    batch, _ = next(it)
    assert 1 <= len(batch) <= 4, f"一批 {len(batch)} 条，应 <= 4"


def test_generator_does_not_read_whole_file(tmp_path, monkeypatch):
    """必须是生成器：调用它本身不该把 parquet 全读进内存。"""
    p = _make_shard(tmp_path / "shard_00000.parquet", n_row_groups=5,
                    rows_per_group=2, tag="L")
    from data import dataloader as D
    monkeypatch.setattr(D, "list_parquet_files", lambda split="train": [p])
    it = D.iter_documents("train", tokenizer_batch_size=4)
    assert hasattr(it, "__next__"), "iter_documents 必须是生成器（用 yield）"
    assert isinstance(it, types.GeneratorType)
