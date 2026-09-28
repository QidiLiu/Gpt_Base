"""
BPE Tokenizer：训练用 rustbpe，推理用 tiktoken（GPT-4 风格）。

教程卷1 会解释：
  - BPE 到底在做什么（从字符开始，反复合并最高频的相邻对）
  - 为什么特殊 token 不参与训练，而是追加在词表末尾
  - 对话模板与 loss mask 的关系（这里 render_conversation 产出的 mask
    就是 SFT 时决定「哪些 token 参与 loss」的唯一依据）
"""

import os
import copy
import pickle
from functools import lru_cache

import rustbpe
import tiktoken
import torch

from common import get_tokenizer_dir, log0

# ---------------------------------------------------------------------------
# 特殊 token
# ---------------------------------------------------------------------------
# 设计约束（这些决定会直接影响 SFT 的行为，改动前请想清楚）：
#   1) <|bos|> 放在最前，每篇文档/每个序列都以它开头。
#      dataloader 的 best-fit 打包算法依赖「每行以 BOS 开头」这个不变量。
#   2) 下面 8 个只在 SFT/推理时用，预训练阶段它们从不出现。
#      如果预训练没见过，SFT 里的 embedding 就是随机初始化 —— 靠足够的
#      SFT 样本把它训出来（nanochat 就是这么做的）。
#   3) python_start/python_end/output_start/output_end 构成一个工具调用协议：
#      assistant 采样到 python_start 后面跟一段代码，引擎执行后把结果
#      以 output_start/.../output_end 强制喂回去。
SPECIAL_TOKENS = [
    "<|bos|>",           # 序列/文档开始
    "<|user_start|>",    # user 消息开始
    "<|user_end|>",
    "<|assistant_start|>",  # assistant 消息开始
    "<|assistant_end|>",
    "<|python_start|>",  # assistant 调用 python 工具
    "<|python_end|>",
    "<|output_start|>",  # 工具输出回传给 assistant
    "<|output_end|>",
]

# 与 GPT-4 的 split pattern 唯一的差异：数字用 \p{N}{1,2} 而不是 {1,3}。
# 原因：小词表下把 token 花在长数字上不划算。作者实测 2 是 32K 词表的甜点。
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    """
    薄薄一层包装：训练用 rustbpe，编码用 tiktoken。

    为什么要两个库？
      rustbpe 的训练很快（Rust 实现），但推理没有高效实现；
      tiktoken 推理极快（Rust + SIMD），但不能训练。
      训练一次、推理多次，所以训练用 rustbpe，推理用 tiktoken，
      中间用「mergeable_ranks」这个中间表示搭桥。
    """

    def __init__(self, enc, bos_token: str = "<|bos|>"):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    # -- 训练 ---------------------------------------------------------------
    @classmethod
    def train(cls, text_iterator, vocab_size: int, log_every: int = 100):
        """
        从文本流训练 BPE。

        注意 vocab_size 这里**不含**特殊 token：特殊 token 在 __init__ 之后
        追加到词表末尾，训练时看不到它们。
        """
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, (
            f"扣掉 {len(SPECIAL_TOKENS)} 个特殊 token 后至少要 256，"
            f"得到 {vocab_size_no_special}"
        )
        log0(f"训练 BPE：目标词表 {vocab_size_no_special}（+{len(SPECIAL_TOKENS)} 特殊 token = {vocab_size}）")
        t = rustbpe.Tokenizer()
        t.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)

        # 桥接：rustbpe 的 mergeable_ranks -> tiktoken 的 mergeable_ranks
        pattern = t.get_pattern()
        mergeable_ranks = {bytes(k): v for k, v in t.get_mergeable_ranks()}
        offset = len(mergeable_ranks)
        special_tokens = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        log0(f"BPE 训练完成，{offset} 个普通 token + {len(SPECIAL_TOKENS)} 个特殊 token")
        enc = tiktoken.Encoding(
            name="gpt_base_bpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,   # bytes -> 合并优先级 rank
            special_tokens=special_tokens,     # 名称 -> token id
        )
        return cls(enc)

    # -- 存取 ---------------------------------------------------------------
    def save(self, tokenizer_dir: str | None = None):
        tokenizer_dir = tokenizer_dir or get_tokenizer_dir()
        os.makedirs(tokenizer_dir, exist_ok=True)
        path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.enc, f)
        log0(f"tokenizer 已保存 -> {path}")

    @classmethod
    def load(cls, tokenizer_dir: str | None = None):
        tokenizer_dir = tokenizer_dir or get_tokenizer_dir()
        path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        assert os.path.exists(path), (
            f"找不到 {path}。先跑 `bash script/train_base.sh smoke`，"
            "它会自动训练 tokenizer。"
        )
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    # -- 基本接口 -----------------------------------------------------------
    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    @lru_cache(maxsize=64)
    def encode_special(self, text: str) -> int:
        """把特殊 token 字符串（如 "<|bos|>"）转成 id。"""
        return self.enc.encode_single_token(text)

    def id_to_token(self, token_id: int) -> str:
        return self.enc.decode([token_id])

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """
        text 可以是 str 或 list[str]（后者走批量编码，快很多）。

        prepend/append 可以是 token id 字符串，也可以是特殊 token 名。
        dataloader 用 prepend="<|bos|>" 给每篇文档加 BOS。
        """
        if prepend is not None and not isinstance(prepend, int):
            prepend = self.encode_special(prepend)
        if append is not None and not isinstance(append, int):
            append = self.encode_special(append)

        if isinstance(text, str):
            ids = [self.enc.encode_ordinary(text)]
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
        else:
            raise ValueError(f"不能编码的类型: {type(text)}")

        for row in ids:
            # insert(0)/append 都是 O(n)，在长列表上不便宜。
            # 这里为了可读性保留直观写法；真要抠性能可以改成拼接。
            if prepend is not None:
                row.insert(0, prepend)
            if append is not None:
                row.append(append)
        return ids[0] if isinstance(text, str) else ids

    def __call__(self, *a, **kw):
        return self.encode(*a, **kw)

    def decode(self, ids) -> str:
        return self.enc.decode(ids)

    def decode_bytes(self, token_id: int) -> bytes:
        """拿到单个 token 对应的原始字节。bpb 指标要用它算「每个 token 覆盖几个字节」。"""
        return self.enc.decode_single_token_bytes(token_id)

    # -- 对话渲染（SFT / RL 用）----------------------------------------------
    def render_conversation(self, conversation: dict, max_tokens: int = 2048,
                            truncate: str = "right"):
        """
        把一段对话渲染成 (ids, mask)。

        mask[i] == 1  -> 第 i 个 token **参与 loss**
        mask[i] == 0  -> 第 i 个 token **不参与 loss**（只是上下文）

        这个 mask 是 SFT 的全部秘密：预训练时模型学「接着往下写」，
        但 SFT 只想让它学「assistant 该说什么」。所以 user 消息、BOS、
        各种结构性 token 全部 mask=0，只有 assistant 真正吐出的内容 mask=1。

        注意 python_output 段是 mask=0：那是引擎执行工具后塞回去的，
        不是模型自己产生的，不该训练模型去猜。

        ── truncate 参数：一个必须踩过一次才明白的坑 ──────────────
          "right"（从尾部切掉多余的）—— 用于**预训练/评测**，保留开头
          "left" （从头部切掉多余的）—— 用于 **SFT**，保留结尾

        为什么 SFT 必须用 left？
          对话很长时（>max_tokens），从尾部截断会把**最后一条 assistant
          回复整段切掉**，只剩 user 的问题 —— 这时 mask 全是 0，
          交叉熵在 ignore_index 全命中时算出 0/0 = **NaN**。
          实测 512 长度的 SFT 数据里有 23% 的样本落入这个陷阱。

          保留尾部才是对的：最近一轮才是我们要学的东西；
          至于「对话开头被切掉」是无所谓的，模型只需要学「在这样的上下文
          状态下该说什么」。
        """
        assert truncate in ("right", "left"), f"truncate 必须是 right/left，得到 {truncate}"
        ids: list[int] = []
        mask: list[int] = []

        def add(token_ids, mask_val: int):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        messages = conversation["messages"]
        # system 消息合并进后面的 user 消息（不支持独立 system 段）
        if messages[0]["role"] == "system":
            messages = copy.deepcopy(messages)
            assert messages[1]["role"] == "user", "system 之后必须是 user"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        assert len(messages) >= 1

        S = self.encode_special
        bos, user_s, user_e = self.bos_token_id, S("<|user_start|>"), S("<|user_end|>")
        asst_s, asst_e = S("<|assistant_start|>"), S("<|assistant_end|>")
        py_s, py_e = S("<|python_start|>"), S("<|python_end|>")
        out_s, out_e = S("<|output_start|>"), S("<|output_end|>")

        add(bos, 0)
        for i, message in enumerate(messages):
            expected = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected, (
                f"第 {i} 条消息来自 {message['role']}，但按交替规则应该是 {expected}"
            )
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "user 消息只接受纯字符串"
                add(user_s, 0)
                add(self.encode(content), 0)
                add(user_e, 0)
            else:
                add(asst_s, 0)
                if isinstance(content, str):
                    add(self.encode(content), 1)
                elif isinstance(content, list):
                    # content 是 list 时表示含工具调用，见 tasks/gsm8k.py 的解析
                    for part in content:
                        vids = self.encode(part["text"])
                        if part["type"] == "text":
                            add(vids, 1)
                        elif part["type"] == "python":
                            add(py_s, 1); add(vids, 1); add(py_e, 1)
                        elif part["type"] == "python_output":
                            add(out_s, 0); add(vids, 0); add(out_e, 0)
                        else:
                            raise ValueError(f"未知 part 类型: {part['type']}")
                else:
                    raise ValueError(f"未知 content 类型: {type(content)}")
                add(asst_e, 1)

        if truncate == "right":
            return ids[:max_tokens], mask[:max_tokens]
        # left：保留结尾（最近一轮对话）
        return ids[-max_tokens:], mask[-max_tokens:]

    def render_for_completion(self, conversation: dict) -> list[int]:
        """
        RL 采样用：渲染对话，但**砍掉最后一条 assistant 消息**，
        只留 <|assistant_start|> 作为「请你续写」的信号。
        """
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "最后一条必须是 assistant"
        messages.pop()
        ids, _ = self.render_conversation(conversation)
        ids.append(self.encode_special("<|assistant_start|>"))
        return ids

    def visualize(self, ids, mask=None, with_token_id=False) -> str:
        """把 token 序列染色打印出来，调试 mask 时非常有用。"""
        R, G, Y, X = "\033[91m", "\033[92m", "\033[90m", "\033[0m"
        out = []
        for i, tid in enumerate(ids):
            m = mask[i] if mask is not None else 0
            color = G if m == 1 else R
            s = f"{color}{self.decode([tid])}{X}"
            if with_token_id:
                s += f"{Y}({tid}){X}"
            out.append(s)
        return "|".join(out)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------
def get_tokenizer() -> BPETokenizer:
    return BPETokenizer.load()


def get_token_bytes(device="cpu") -> torch.Tensor:
    """
    每个 token 对应多少字节。计算 bpb 时要用：
        bpb = loss / ln(2) / (每个 token 的平均字节数)
    """
    path = os.path.join(get_tokenizer_dir(), "token_bytes.pt")
    assert os.path.exists(path), (
        f"找不到 {path}，它由 tokenizer 训练脚本写出"
    )
    return torch.load(path, map_location=device)
