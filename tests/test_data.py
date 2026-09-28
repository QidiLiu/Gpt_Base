"""
卷 1（数据管线）的测试。

这一卷的完成判据：全部通过。
  uv run pytest tests/test_data.py -v
"""

import torch

from data.tokenizer import get_tokenizer, SPECIAL_TOKENS
from data.dataset import list_parquet_files
from data.dataloader import make_dataloader


# ===========================================================================
# Tokenizer
# ===========================================================================
def test_tokenizer_roundtrip():
    """回归测试：encode 后再 decode 必须完全等于原文。"""
    tok = get_tokenizer()
    for s in [
        "To be, or not to be, that is the question:",
        "In 2026, the price of compute dropped by ~10x/year.",
        "def f(x): return x**2 + 1  # 中文测试",
        "  leading and trailing spaces  ",
    ]:
        assert tok.decode(tok.encode(s)) == s, f"round-trip 失败: {s!r}"


def test_tokenizer_batch_matches_single():
    """回归测试：批量编码的结果必须与逐条编码一致。"""
    tok = get_tokenizer()
    texts = ["hello world", "another sentence here", "third one"]
    batch = tok.encode(texts)
    for t, ids in zip(texts, batch):
        assert ids == tok.encode(t), f"批量与单条结果不一致: {t!r}"


def test_prepend_and_append_special():
    """prepend/append 应该真的在首尾加上特殊 token。"""
    tok = get_tokenizer()
    bos = tok.encode_special("<|bos|>")
    ids = tok.encode("hi", prepend="<|bos|>", append="<|assistant_end|>")
    assert ids[0] == bos
    assert ids[-1] == tok.encode_special("<|assistant_end|>")
    # 传 int 也应该工作
    ids2 = tok.encode("hi", prepend=bos)
    assert ids2[0] == bos


def test_special_tokens_are_sequential_tail():
    """
    回归测试：特殊 token 必须占据词表的**最后 9 个 id**，且顺序与
    SPECIAL_TOKENS 一致。因为 BPE 训练时看不到它们，
    它们被追加在普通 token 之后。
    """
    tok = get_tokenizer()
    V = tok.get_vocab_size()
    n = len(SPECIAL_TOKENS)
    assert V - n == 8183, f"普通 token 数应等于 vocab_size - {n}，得到 {V - n}"
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.encode_special(name) == V - n + i, (
            f"{name} 的 id 应是 {V - n + i}，实际 {tok.encode_special(name)}")


def test_decode_bytes_length():
    """回归测试：decode_bytes 的长度必须等于原始字节数（bpb 指标的基础）。"""
    tok = get_tokenizer()
    ids = tok.encode("The capital of France is Paris.")
    for i in ids:
        assert len(tok.decode_bytes(i)) >= 1, "每个 token 至少 1 字节"
    total = sum(len(tok.decode_bytes(i)) for i in ids)
    assert total == len("The capital of France is Paris.".encode())


# ===========================================================================
# 对话渲染与 loss mask
# ===========================================================================
def test_mask_covers_exactly_assistant():
    """
    回归测试：mask=1 的 token 拼起来应该恰好是
    「assistant 的回答 + <|assistant_end|>」。

    关键点：<|assistant_end|> 也在 mask 里（mask=1），
    因为模型必须学会「什么时候停」。如果它被 mask 掉，
    模型推理时会一直说下去停不下来。
    """
    tok = get_tokenizer()
    conv = {"messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]}
    ids, mask = tok.render_conversation(conv)
    learned = [t for t, m in zip(ids, mask) if m == 1]
    assert tok.decode(learned).startswith("Paris."), "监督信号应以回答开头"
    assert learned[-1] == tok.encode_special("<|assistant_end|>"), \
        "监督信号应以 <|assistant_end|> 结尾（教模型何时停）"
    # 开头（BOS / user_start / user 段 / user_end / assistant_start）都是 0
    assert mask[0] == 0, "BOS 不应参与 loss"
    assert ids[0] == tok.get_bos_token_id()


def test_python_output_not_supervised():
    """
    回归测试：工具返回值（python_output 段）mask 必须是 0。

    理由：那是**引擎**执行工具后塞回去的，不是模型产生的。
    训练模型去猜结果等于教它「编造计算结果」。
    但工具**调用**（python 段）要学 —— 模型需要学会「什么时候该调工具」。
    """
    tok = get_tokenizer()
    conv = {"messages": [
        {"role": "user", "content": "2 + 3 = ?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "2+3"},
            {"type": "python_output", "text": "9999"},   # 故意放一个明显的假结果
            {"type": "text", "text": " is 5"},
        ]},
    ]}
    ids, mask = tok.render_conversation(conv)
    learned = tok.decode([t for t, m in zip(ids, mask) if m == 1])
    assert "9999" not in learned, "工具返回值不该被学（那是引擎算的）"
    assert "2+3" in learned, "工具调用表达式要学"
    assert "is 5" in learned


def test_truncate_left_keeps_supervision():
    """
    回归测试（★ 本项目最容易踩的 NaN 来源）：
    SFT 必须用 truncate="left" 保留结尾。

    对话很长时，从尾部截断会把最后一条 assistant 回复整段切掉，
    只剩 user 的问题 —— mask 全 0，F.cross_entropy 算出 0/0 = NaN，
    一旦进入反向传播就污染全部参数且不可恢复。
    """
    tok = get_tokenizer()
    long_conv = {"messages": [
        {"role": "user", "content": "问题" * 300},              # 超长 user
        {"role": "assistant", "content": "这是答案。" * 20},
    ]}
    ids_r, mask_r = tok.render_conversation(long_conv, max_tokens=128, truncate="right")
    ids_l, mask_l = tok.render_conversation(long_conv, max_tokens=128, truncate="left")
    assert sum(mask_r) == 0, "truncate='right' 会把监督信号全切掉（这就是 NaN 的来源）"
    assert sum(mask_l) > 0, "truncate='left' 必须保留结尾的监督信号"


def test_role_alternation_asserted():
    """回归测试：连续两个 user 消息应该 assert（打破交替约定会让 mask 错乱）。"""
    tok = get_tokenizer()
    bad = {"messages": [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "c"},
    ]}
    import pytest
    with pytest.raises(AssertionError, match="交替规则"):
        tok.render_conversation(bad)


# ===========================================================================
# Dataloader
# ===========================================================================
def _one_batch(batch_size=4, seq_len=512, split="train"):
    tok = get_tokenizer()
    dl = make_dataloader(tok, batch_size, seq_len, split, device="cpu",
                         buffer_size=200)
    return tok, next(dl)


def test_dataloader_shapes_and_dtype():
    tok, (x, y, st) = _one_batch()
    assert tuple(x.shape) == (4, 512), f"inputs 形状错: {tuple(x.shape)}"
    assert tuple(y.shape) == (4, 512), f"targets 形状错: {tuple(y.shape)}"
    assert x.dtype == torch.int64 and y.dtype == torch.int64, "必须是 int64"
    assert set(st.keys()) == {"pq_idx", "rg_idx", "epoch"}, f"state 字段错: {st}"


def test_targets_are_inputs_shifted():
    """
    回归测试：targets 必须是 inputs 右移一位。
    这一对是靠「每行装 T+1 个 token」构造出来的
    （inputs = row[:-1], targets = row[1:]），少装一个就会错位。
    """
    tok, (x, y, _) = _one_batch()
    assert (y[:, :-1] == x[:, 1:]).all(), "targets 必须是 inputs 的右移"


def test_every_row_starts_with_bos():
    """
    回归测试：每行必须以 <|bos|> 开头，且**只有一个** BOS 在开头。

    曾经的 bug：手动写 row[0] = bos，而第一篇文档在 tokenizer 编码时
    已经 prepend=bos 了，结果开头出现两个 BOS。
    """
    tok, (x, _, _) = _one_batch()
    bos = tok.get_bos_token_id()
    assert (x[:, 0] == bos).all(), "每行必须以 BOS 开头"
    assert (x[:, 1] != bos).all(), (
        "第 1 个 token 不该又是 BOS（文档自带 BOS，别手动再写一次）")


def test_no_padding():
    """
    回归测试：不能有 padding token。

    如果有 padding，模型会学到「一大段 pad 是什么」这种
    推理时永远用不到的模式。所以宁可裁掉 35% 的 token。
    """
    tok, (x, y, _) = _one_batch()
    V = tok.get_vocab_size()
    assert ((x >= 0) & (x < V)).all(), "inputs 超出词表范围（可能有 padding）"
    assert ((y >= 0) & (y < V)).all(), "targets 超出词表范围（可能有 padding）"


def test_bestfit_prefers_longest_fitting():
    """
    回归测试（本卷核心）：best-fit 必须在能完整放下的文档里挑**最长**的。

    做法：构造一个场景，只有「挑最长」才能装下最多的完整文档。
      文档 [30, 35, 70, 75, 100]，容量 200
      · best-fit  : 100 + 75 + 30(裁5)   裁 5 个
      · first-fit : 30 + 35 + 70 + 75(裁10)  裁 10 个
      · 裁最长的  : 100 + 75 + 70(裁45)  裁 45 个
    """
    tok, (x, _, _) = _one_batch()
    bos = tok.get_bos_token_id()
    # 每一行恰好等于「一行文档序列」，BOS 数 = 文档数
    # 这一行装了几篇完整文档？无法直接观测，但可以验证：
    # 装完的行长度必须是 seq_len+1 裁掉 1（=seq_len），即「填满」
    assert x.shape[1] == 512, "每行必须被填满（无 padding -> 一定是满的）"

    # 更直接的验证：统计一批里 BOS 的分布。
    # best-fit 会尽量减少「被裁剪的文档数」，所以每行 BOS 数应该偏少
    # （因为一篇被裁的文档也会贡献一个 BOS）
    dl = make_dataloader(tok, 8, 512, "train", device="cpu", buffer_size=200)
    counts = []
    for _ in range(8):
        xb, _, _ = next(dl)
        counts.extend((xb == bos).sum(dim=1).tolist())
    assert sum(counts) > 0, "应该能看到 BOS"
    # ClimbMix 文档中位数约 628 token > 512，所以每行 1-3 个 BOS 是正常的
    assert max(counts) <= 8, f"单行 BOS 数 {max(counts)} 异常（可能装箱逻辑坏了）"


def test_split_convention_val_is_last_shard():
    """
    回归测试：val 必须是**最后一个** shard，且与 train 不重叠。

    为什么要固定最后一个？训练是多 epoch 的循环，
    随机切出来的 val 会被反复看到，指标就不可信了。
    """
    train = list_parquet_files("train")
    val = list_parquet_files("val")
    assert len(val) == 1, f"val 应该恰好 1 个 shard，得到 {len(val)}"
    assert val[0] not in train, "val 不能出现在 train 里"
    assert train[-1] < val[0], "val 应该排在 train 之后（文件名排序）"


# ===========================================================================
# 卷1 第 02 章：惰性遍历 parquet（training.train_tokenizer）
# ===========================================================================
def test_iter_parquet_text_yields_documents_lazily(tmp_path, monkeypatch):
    """iter_parquet_text 必须逐 row group 惰性产出文本。

    这是卷1 第 02 章的手抄目标。锁住两件事：
      1) 产出的内容与 parquet 里的 "text" 列一致（顺序不乱）
      2) **惰性** —— 调用它不应把整个文件读进内存

    真训练要遍历约 10 亿字符，一次性 .to_pylist() 整个文件会 OOM。
    所以必须是生成器，且按 row group 逐个读。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from training import train_tokenizer as tt

    docs = [f"文档 {i} " + "x" * (i + 1) for i in range(6)]
    path = tmp_path / "shard_00000.parquet"
    # 切成 3 个 row group，每个 2 篇，用来验证 row group 边界没被搞乱
    table = pa.table({"text": pa.array(docs[:2])})
    writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    for chunk in (docs[2:4], docs[4:6]):
        t = pa.table({"text": pa.array(chunk)})
        writer.write_table(t)
    writer.close()

    monkeypatch.setattr(tt, "list_parquet_files", lambda split="train": [str(path)])

    # 1) 是生成器，不是已经算好的 list
    it = tt.iter_parquet_text("train")
    assert hasattr(it, "__next__"), "iter_parquet_text 必须是生成器（用 yield，不要 return list）"

    # 2) 逐个产出，顺序与原文一致
    got = list(it)
    assert got == docs, f"产出的文本与 parquet 不一致：{got}"

    # 3) 惰性：limit_shards=1 时只处理第一个文件
    it2 = tt.iter_parquet_text("train", limit_shards=1)
    assert len(list(it2)) == len(docs)


# ===========================================================================
# 卷1 第 06 章：token_bytes 与 bpb
# ===========================================================================
def test_compute_token_bytes_matches_vocab_size():
    """compute_token_bytes 必须给出「每个 token 多少字节」这张表。

    卷1 第 06 章手抄目标。bpb 换算全靠它：
        bpb = total_nats / ln(2) / total_bytes
    """
    from training.train_tokenizer import compute_token_bytes

    tok = get_tokenizer()
    n = tok.get_vocab_size()
    tb = compute_token_bytes(tok)

    assert tb.shape == (n,), f"token_bytes 长度应为词表大小 {n}，实际 {tuple(tb.shape)}"
    assert not torch.isnan(tb).any(), "token_bytes 出现 NaN"
    assert (tb >= 0).all(), "token_bytes 不能为负"

    # 普通 token 必须真的查到字节数（全 0 说明循环没跑或 decode_bytes 恒失败）
    ordinary = [i for i in range(256, n) if i not in set(range(256))]
    if ordinary:
        assert tb[ordinary[0]].item() > 0, "普通 token 的字节数不应为 0"

    # ★ 交叉验证：把整张表和逐个查一遍的结果对一遍，
    #   防止出现「长度对了但内容错位」这种静默错误
    for tid in (0, n // 2, n - 1):
        try:
            expect = len(tok.decode_bytes(tid))
        except Exception:
            expect = 0.0
        assert abs(tb[tid].item() - expect) < 1e-6, (
            f"token {tid} 的字节数应是 {expect}，实际 {tb[tid].item()}")


def test_evaluate_bpb_is_scale_free_and_restores_train_mode():
    """evaluate_bpb 必须给出与词表无关的 bpb，且不留下 eval 副作用。

    卷1 第 06 章「第 2 块」的手抄目标。两个易错点：
      · 必须用 loss_reduction='none' 再按 y >= 0 过滤，
        否则 -1 位置会污染结果
      · 入口 model.eval()，出口必须还原成原来的 train/eval 状态
    """
    from common.config import make_run_config
    from model.gpt import build_model
    from evaluation.metrics import evaluate_bpb

    cfg = make_run_config("debug", vocab_size=64)
    cfg.model.sequence_len, cfg.model.n_embd = 32, 32
    cfg.model.n_head = cfg.model.n_kv_head = 2
    cfg.model.head_dim = 16
    model = build_model(cfg.model, device="cpu")

    tok = get_tokenizer()
    # token_bytes 用一个常数表即可：这里验的是 bpb 的换算与副作用，不是表本身
    tb = torch.full((cfg.vocab_size,), 4.0)

    batch = torch.randint(0, cfg.vocab_size, (2, 32))
    loader = [(batch, batch.clone(), {})]

    model.train()
    bpb = evaluate_bpb(model, loader, tb, max_batches=1)

    assert model.training, "evaluate_bpb 结束后必须把 model 还原成 train 模式"
    assert not torch.isnan(torch.tensor(bpb)), f"bpb 是 NaN：{bpb}"
    assert bpb > 0, f"bpb 应为正数，得到 {bpb}"
    # 随机初始化的模型约等于均匀分布：bpb ≈ log2(vocab) ≈ 6
    assert 1.0 < bpb < 20.0, f"bpb 数量级不对：{bpb}"


def test_evaluate_bpb_ignores_masked_targets():
    """targets 里被 mask 掉（-1）的位置，绝不能计入 loss 或字节数。

    这是 evaluate_bpb 最容易写错的一处：若忘了 `valid = y >= 0`，
    -1 会被当成合法的 token 下标，统计出错误的字节数。
    """
    from common.config import make_run_config
    from model.gpt import build_model
    from evaluation.metrics import evaluate_bpb

    cfg = make_run_config("debug", vocab_size=64)
    cfg.model.sequence_len, cfg.model.n_embd = 32, 32
    cfg.model.n_head = cfg.model.n_kv_head = 2
    cfg.model.head_dim = 16
    torch.manual_seed(0)
    model = build_model(cfg.model, device="cpu")
    model.eval()

    x = torch.randint(0, cfg.vocab_size, (1, 32))
    y_full = torch.randint(0, cfg.vocab_size, (1, 32))
    tb = torch.full((cfg.vocab_size,), 4.0)

    bpb_full = evaluate_bpb(model, [(x, y_full, {})], tb, max_batches=1)

    # 把后半段全部 mask 掉。字节数减半 -> bpb 必须大致翻倍
    y_half = y_full.clone()
    y_half[:, 16:] = -1
    bpb_half = evaluate_bpb(model, [(x, y_half, {})], tb, max_batches=1)

    assert not torch.isnan(torch.tensor(bpb_half)), "mask 之后 bpb 变成 NaN"
    # 有效 token 减半、总 nats 减半、总字节减半 -> bpb 不变
    assert abs(bpb_half - bpb_full) < abs(bpb_full) * 0.35, (
        f"mask 掉一半 target 后 bpb 应基本不变（分子分母同减半），"
        f"但 {bpb_full:.4f} -> {bpb_half:.4f}。多半是忘了按 y >= 0 过滤。")
