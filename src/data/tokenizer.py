"""
BPE Tokenizer：训练用 rustbpe，推理用 tiktoken（GPT-4 风格）。

▓▓ 这一章要你敲的部分 ▓▓
    BPETokenizer 的全部方法。

    带 🔶 的是「规格」（照抄即可），带 ❗ 的是「要你想清楚」的部分。

────────────────────────────────────────────────────────────────
卷1 的知识地图
────────────────────────────────────────────────────────────────
    ch02  为什么需要分词器    -> 下面 SPECIAL_TOKENS / SPLIT_PATTERN 的设计理由
    ch03  手写一个玩具 BPE    -> BPETokenizer.train 的核心思路
    ch04  对话模板与 loss mask -> render_conversation（本文件最难的一个函数）
"""

import os
import copy
import pickle
from functools import lru_cache

import rustbpe
import tiktoken
import torch

from common import get_tokenizer_dir, log0

# ===========================================================================
# 🔶 规格部分：直接抄，理解设计理由即可
# ===========================================================================

# 特殊 token 不参与 BPE 训练（它们不存在自然文本里），
# 而是**追加在词表末尾**，id 紧接在普通 token 之后。
#
# 三个设计约束（改动前请想清楚）：
#   1) <|bos|> 放最前，每篇文档/序列都以它开头。
#      dataloader 的 best-fit 装箱算法依赖「每行以 BOS 开头」这个不变量。
#   2) 下面 8 个只在 SFT/推理时用，预训练阶段它们从不出现。
#      所以它们的嵌入是随机初始化的，靠 SFT 样本训出来。
#   3) python_start/python_end/output_start/output_end 构成一个
#      **工具调用协议**，卷 7 会详细讲。
SPECIAL_TOKENS = [
    "<|bos|>",              # 序列/文档开始
    "<|user_start|>",       # user 消息开始
    "<|user_end|>",
    "<|assistant_start|>",  # assistant 消息开始
    "<|assistant_end|>",
    "<|python_start|>",     # assistant 调用 python 工具
    "<|python_end|>",
    "<|output_start|>",     # 工具输出回传给 assistant
    "<|output_end|>",
]

# 与 GPT-4 的 split pattern 唯一的差异：数字用 \p{N}{1,2} 而不是 {1,3}。
# 原因：小词表下把 token 花在长数字上不划算。作者实测 2 是 32K 词表的甜点。
# 详见 tutorial/卷1 第 02 章「SPLIT_PATTERN 逐段解读」。
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


# ===========================================================================
# ❗ 要你敲的部分
# ===========================================================================
class BPETokenizer:
    """
    薄薄一层包装：训练用 rustbpe，编码用 tiktoken。

    为什么要两个库？
      rustbpe 训练快（Rust 实现），但没有高效的推理实现；
      tiktoken 推理极快（Rust + SIMD），但**不能训练**。
      训练一次、推理无数次，所以训练用 rustbpe、推理用 tiktoken，
      中间用「mergeable_ranks」这个中间表示搭桥。
    """

    def __init__(self, enc, bos_token: str = "<|bos|>"):
        # ❗ 保存 enc 和 bos_token_id
        raise NotImplementedError(
            "待实现：__init__ —— 保存 self.enc；\n"
            "      self.bos_token_id = self.encode_special(bos_token)\n"
            "      提示：encode_special 定义在下面，所以调用它没问题。")

    # -- 训练 ---------------------------------------------------------------
    @classmethod
    def train(cls, text_iterator, vocab_size: int):
        """
        从文本流训练 BPE，然后搭桥成 tiktoken 编码器。

        返回一个 cls 实例。

        ── 你要想清楚的三个点 ──────────────────────────────
        1) `vocab_size` 传进来的**含**特殊 token，但训练时只能用不含的。
           用什么算？减多少？（提示：`len(SPECIAL_TOKENS)`）
           减完的数至少要 256，不够要 assert。

        2) 搭桥那一段是全文件最关键的地方：
             rustbpe 给你的三样东西：
               tokenizer.get_pattern()               -> 正则字符串
               tokenizer.get_mergeable_ranks()       -> [(bytes, rank), ...]
             tiktoken 要的两样东西：
               pat_str=pattern
               mergeable_ranks={bytes: rank} 字典
               special_tokens={名称: id} 字典
           关键问题：**特殊 token 的 id 从几开始？**
           提示：普通 token 占了 [0, len(mergeable_ranks))。

        3) text_iterator 是生成器，一次只吐一批文本。
           rustbpe 的 train_from_iterator 签名是
           `train_from_iterator(iterator, vocab_size, pattern=...)`
        """
        raise NotImplementedError(
            "待实现：train ——\n"
            "  1) vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)\n"
            "  2) t = rustbpe.Tokenizer(); t.train_from_iterator(...)\n"
            "  3) pattern = t.get_pattern()\n"
            "     mergeable_ranks = {bytes(k): v for k, v in t.get_mergeable_ranks()}\n"
            "     offset = len(mergeable_ranks)\n"
            "     special_tokens = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}\n"
            "  4) enc = tiktoken.Encoding(name=..., pat_str=pattern,\n"
            "                            mergeable_ranks=..., special_tokens=...)\n"
            "  5) return cls(enc)\n"
            "参考实现：git show solution:src/data/tokenizer.py")

    # -- 存取 ---------------------------------------------------------------
    def save(self, tokenizer_dir: str | None = None):
        """
        把 enc 存成 tokenizer.pkl。

        为什么用 pickle 而不是 json？
          tiktoken.Encoding 是个 Rust 对象，json 表达不了。
        """
        raise NotImplementedError(
            "待实现：save —— makedirs(ok_exist_ok=True)；"
            "pickle.dump(self.enc, f) 写到 <dir>/tokenizer.pkl")

    @classmethod
    def load(cls, tokenizer_dir: str | None = None):
        """从 tokenizer.pkl 反序列化。没找到文件时要 assert 并给出「先跑 train_base.sh」的提示。"""
        raise NotImplementedError(
            "待实现：load —— open(path,'rb') + pickle.load(f) + cls(enc)\n"
            "提示：文件不存在时用 assert os.path.exists(path), <一句友好的提示>")

    # -- 基本接口 -----------------------------------------------------------
    def get_vocab_size(self) -> int:
        """词表大小（含特殊 token）。tiktoken 里对应哪个属性？"""
        raise NotImplementedError("待实现：return self.enc.???")

    def get_bos_token_id(self) -> int:
        """BOS 的 token id。"""
        raise NotImplementedError("待实现：return self.???")

    def get_special_tokens(self):
        """所有特殊 token 的集合。"""
        raise NotImplementedError("待实现：return self.enc.???")

    @lru_cache(maxsize=64)
    def encode_special(self, text: str) -> int:
        """
        把特殊 token 字符串（如 "<|bos|>"）转成 id。

        为什么加 @lru_cache？
          推理时每秒会调用它成千上万次（每个 token 都要查）。
          特殊 token 只有 9 个，缓存后几乎是 O(1) 的字典查。
        """
        raise NotImplementedError("待实现：return self.enc.???(text)")

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """
        text 可以是 str 或 list[str]（后者走批量编码，快很多）。

        prepend/append 可以是 token id 字符串，也可以是特殊 token 名。
        dataloader 用 prepend="<|bos|>" 给每篇文档加 BOS。

        ── 你要想清楚的点 ──────────────────────────────
        返回值：str 输入 -> list[int]；list 输入 -> list[list[int]]
        prepend/append 是 str 时要先转成 id（注意别只处理 prepend）。
        """
        raise NotImplementedError(
            "待实现：encode ——\n"
            "  1) prepend/append 若不是 int，先 self.encode_special(x)\n"
            "  2) str -> [self.enc.encode_ordinary(text)]；list -> self.enc.encode_ordinary_batch(text, num_threads=num_threads)\n"
            "  3) 遍历每一行 insert(0, prepend) / append(append)\n"
            "  4) 返回 ids[0] 或 ids\n"
            "提示：encode_ordinary 会**忽略**特殊 token，所以 <|bos|> 必须手动加。")

    def __call__(self, *a, **kw):
        """让 tokenizer 可以当函数用：tok(text) == tok.encode(text)"""
        return self.encode(*a, **kw)

    def decode(self, ids) -> str:
        """ids -> 文本。"""
        raise NotImplementedError("待实现：return self.enc.???(ids)")

    def decode_bytes(self, token_id: int) -> bytes:
        """
        拿到单个 token 对应的原始字节。

        bpb 指标要用它算「每个 token 覆盖几个字节」（教程卷1 第 06 章）。
        """
        raise NotImplementedError("待实现：return self.enc.???(token_id)")

    # -- 对话渲染（SFT / RL 用）----------------------------------------------
    def render_conversation(self, conversation: dict, max_tokens: int = 2048,
                            truncate: str = "right"):
        """
        把一段对话渲染成 (ids, mask)。**本文件最难的一个函数。**

        mask[i] == 1  -> 第 i 个 token **参与 loss**
        mask[i] == 0  -> 第 i 个 token **不参与 loss**（只是上下文）

        ── 你要想清楚的四个点 ────────────────────────────

        (1) 为什么需要 mask？
            预训练学的是「无条件接着写下一个 token」；
            SFT 只想让模型学「assistant 该说什么」。
            如果 user 侧也计 loss，模型会同时学「怎么提问」，
            既浪费容量（推理时没人替它提问），又稀释了真正的目标。

        (2) 每个结构 token 的 mask 分别是多少？
            提示：先想「这个 token 是谁产生的」。
              · BOS                    -> 不是任何人产生的
              · <|user_start|> 等      -> 模板产生的
              · user 的内容            -> 用户产生的
              · <|assistant_start|>    -> 模板产生的，但它标记「轮到你说了」
              · assistant 的内容        -> 模型产生的  <- 核心
              · <|assistant_end|>      -> 想想模型如果不学这个会怎样
              · <|python_start|> 等    -> 模型调用工具时产生的
              · python 表达式          -> 模型产生的
              · <|output_*|> 和结果     -> **引擎**产生的

        (3) content 可能是 str，也可能是 list[dict]（含工具调用）。
            list 里每项有 "type": text / python / python_output。
            三种 type 的 mask 分别是多少？为什么 python_output 是 0？

        (4) truncate 参数：SFT 必须用 "left"。
            提示：对话很长时，从尾部截断会把**最后一条 assistant 回复整段切掉**，
            只剩 user 的问题 —— 这时 mask 全是 0，交叉熵在 ignore_index
            全命中时算出 0/0 = NaN。实测 512 长度下 23% 的样本会踩中。
            「保留结尾」和「保留开头」分别对应哪个参数？

        另外：role 是严格交替的（user, assistant, user, ...），
        要 assert 检查。system 消息要合并进后面的 user 消息。
        """
        raise NotImplementedError(
            "待实现：render_conversation ——\n"
            "  1) 定义内部辅助 add(token_ids, mask_val)，同时 extend ids 和 mask\n"
            "  2) 处理 system 消息（合并进下一条 user）\n"
            "  3) 取 9 个特殊 token 的 id\n"
            "  4) 遍历 messages，按 role 分支打 mask\n"
            "  5) 按 truncate 参数决定从哪头切\n"
            "  验证：uv run python scratch/mask_demo.py（你得先自己写这个脚本）\n"
            "参考实现：git show solution:src/data/tokenizer.py")

    def render_for_completion(self, conversation: dict) -> list[int]:
        """
        RL 采样用：渲染对话，但**砍掉最后一条 assistant 消息**，
        只留 <|assistant_start|> 作为「请你续写」的信号。
        """
        raise NotImplementedError(
            "待实现：render_for_completion ——\n"
            "  deepcopy 后 pop() 掉最后一条 message，再 render_conversation，\n"
            "  最后 append 一个 <|assistant_start|>")

    def visualize(self, ids, mask=None, with_token_id=False) -> str:
        """
        把 token 序列染色打印出来，调试 mask 时非常有用。
        绿 = mask 1（参与 loss），红 = mask 0。
        """
        raise NotImplementedError("待实现：visualize —— 拼字符串即可")


# ===========================================================================
# 🔶 规格部分：模块级便捷函数
# ===========================================================================
def get_tokenizer() -> BPETokenizer:
    return BPETokenizer.load()


def get_token_bytes(device="cpu") -> torch.Tensor:
    """
    每个 token 对应多少字节。计算 bpb 时要用：
        bpb = (每 token 的 nats) / ln(2) / (每 token 的平均字节数)
    """
    path = os.path.join(get_tokenizer_dir(), "token_bytes.pt")
    assert os.path.exists(path), (
        f"找不到 {path}。它由 python -m training.train_tokenizer 写出"
    )
    return torch.load(path, map_location=device)
