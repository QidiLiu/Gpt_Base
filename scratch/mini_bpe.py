"""
一个可训练、可推理、可序列化的迷你 BPE。只为了看清原理，不追求任何性能。

第 03 章。★ 踩过的坑见 train() 附近的注释。
"""

import json
import re
import time
from collections import Counter

# 生产级预切分正则（与 nanochat / tiktoken 的 SPLIT_PATTERN 同构，
# 只是把 \p{L}\p{N} 换成 Python re 的 \w、把数字长度改成可配）
#
# ★ 关键：最后一个非空白兜底分支「 ?[^\s\w]++[\r\n]*」不能省。
#   `++` 是「一次或多次」，它保证标点符号（? ! , . : ; 等）被吃掉。
#   我第一版写成 `[^\r\n\w]?+\w+`（`?+` 是「零或一次」），结果 `?` 匹配不上，
#   而 re.finditer **只返回匹配、不报告跳过** —— 那个问号就静默消失了。
#   解码时才发现对不上原文。这是最隐蔽的一类 bug。
GPT2_PAT = (
    r"""'(?i:[sdmt]|ll|ve|re)"""
    r"""|[^\r\n\w]?+\w+"""          # 可选非字母前缀 + 一串字母
    r"""|\d{1,2}"""                # 1-2 位数字一组
    r"""| ?[^\s\w]++[\r\n]*"""      # 可选空格 + 标点（一次或多次）+ 换行
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""  # 换行、行尾空白、兜底空白
)


class MiniBPE:
    def __init__(self, num_merges: int, min_frequency: int = 2):
        # 合并表：{符号 bytes: rank}。rank 越小优先级越高（= 训练顺序）
        self.merges = {}
        self.num_merges = num_merges
        self.min_frequency = min_frequency

    def pre_tokenize(self, text: str) -> list[bytes]:
        """
        把文本切成不可再分的块。合并只能在块内进行。

        ★ re.finditer 会**静默跳过**不匹配的字符，所以必须验证
          「所有块拼起来 == 原文」。这行断言是本章最值得抄走的东西。
        """
        blocks = [m.group(0).encode("utf-8") for m in re.finditer(GPT2_PAT, text)]
        assert b"".join(blocks) == text.encode("utf-8"), (
            "预切分正则有字符没匹配上！re.finditer 不会报错，只会静默丢掉它们。"
            "通常是缺少标点兜底分支。"
        )
        return blocks

    # ── 训练 ────────────────────────────────────────────────
    def train(self, text: str):
        # ★ 坑：符号必须用 bytes 表示，不能用 chr(byte) 转成 str。
        #   b"12" 是两个符号 b"1"、b"2" 合并的结果；
        #   而 chr(49)+chr(50) 拼出来是字符串 "12"，与真正的 "12" 无法区分。
        #   后果：merge 表里会混进「两个数字字节」的假象，编码时张冠李戴。
        #   生产实现（rustbpe / tiktoken）全程用 bytes 作为符号，
        #   正是为了避开这个坑。
        words = Counter()
        for block in self.pre_tokenize(text):
            words[self._symbols(block)] += 1     # 初始符号 = 每个单字节

        for _ in range(self.num_merges):
            pairs = Counter()
            for word, cnt in words.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += cnt
            if not pairs:
                break
            (a, b), n = pairs.most_common(1)[0]
            if n < self.min_frequency:
                break
            self.merges[a + b] = len(self.merges)     # rank = 训练顺序
            new_words = Counter()
            for word, cnt in words.items():
                new_words[self._merge_word(word, a, b)] += cnt
            words = new_words

    @staticmethod
    def _symbols(block: bytes) -> tuple:
        """把一个块拆成「单字节符号」的 tuple。"""
        return tuple(block[i:i + 1] for i in range(len(block)))

    @staticmethod
    def _merge_word(word, a, b):
        out, i = [], 0
        ab = a + b
        while i < len(word):
            if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
                out.append(ab); i += 2
            else:
                out.append(word[i]); i += 1
        return tuple(out)

    # ── 推理 ────────────────────────────────────────────────
    def _merge_all(self, block: bytes) -> list[bytes]:
        """
        对单个块应用全部 merge 规则：按 rank 从小到大反复应用，直到没有
        规则能匹配。这与训练时的贪心过程等价，所以编码结果是唯一的。
        """
        parts = list(self._symbols(block))
        while True:
            best_rank, best_i = None, None
            for i in range(len(parts) - 1):
                r = self.merges.get(parts[i] + parts[i + 1])
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        return parts

    def encode(self, text: str) -> list[bytes]:
        out = []
        for block in self.pre_tokenize(text):
            out.extend(self._merge_all(block))
        return out

    def decode(self, tokens: list[bytes]) -> str:
        return b"".join(tokens).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": {k.hex(): v for k, v in self.merges.items()}}, f)


if __name__ == "__main__":
    from data.dataset import download_tiny_shakespeare
    p = download_tiny_shakespeare()
    corpus = open(p, encoding="utf-8").read()

    t0 = time.time()
    bpe = MiniBPE(num_merges=2000)
    bpe.train(corpus * 40)
    print(f"训练 2000 条 merge 规则耗时 {time.time()-t0:.1f}s（纯 Python，很慢）\n")

    test = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer"
    t0 = time.time()
    toks = bpe.encode(test)
    dt = time.time() - t0
    nb = len(test.encode())
    print(f"测试文本: {nb} 字节")
    print(f"编码结果: {len(toks)} token, 压缩率 {nb/len(toks):.2f} bytes/token, 耗时 {dt*1000:.0f}ms")
    assert bpe.decode(toks) == test, "解码应与原文完全一致"
    print("解码回原文一致 [OK]\n")
    for i in range(0, len(toks), 12):
        print("  " + "|".join(t.decode() for t in toks[i:i+12]))

    print("\n--- 对比：代码和数字的压缩率明显更低 ---")
    code = "def compute(a, b): return 2026 * a + b  # note: x=1000000"
    ct = bpe.encode(code)
    print(f"  {len(code.encode())} 字节 -> {len(ct)} token, 压缩率 {len(code.encode())/len(ct):.2f}")
    print("  " + "|".join(t.decode() for t in ct))

    print("\n--- merge 表：rank 就是训练顺序（小的先应用）---")
    items = sorted(bpe.merges.items(), key=lambda kv: kv[1])
    for i, (tok, rank) in enumerate(items):
        if i < 12 or i >= len(items) - 3:
            print(f"  rank {rank:5d} : {tok.decode('utf-8', 'replace')!r}")
        elif i == 12:
            print("  ...")

    print("\n=== 预切分为什么不能省 ===")
    print(f"  'the cat sat' -> 预切分 {[b.decode() for b in bpe.pre_tokenize('the cat sat')]}")
    print("  同一个词在不同上下文里被切成不同 token：")
    for ctx in ["I like the", "the", "say the"]:
        full = ctx + " cat"
        print(f"    {full!r:22s} -> " + "|".join(t.decode() for t in bpe.encode(full)))
