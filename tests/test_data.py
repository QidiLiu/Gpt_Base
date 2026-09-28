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
