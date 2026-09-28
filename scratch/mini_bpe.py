"""
一个可训练、可推理、可序列化的迷你 BPE。只为了看清原理，不追求任何性能。

第 03 章手抄目标。main 分支上是骨架 —— 完整实现在 solution 分支：
    git show solution:scratch/mini_bpe.py

流程分三段，对应下面三个方法：
    1) 预切分  pre_tokenize   文本 -> 不可再分的块（合并只能在块内进行）
    2) 训练    train          统计相邻对频率，反复合并最高频的那一对
    3) 推理    _merge_all     按 rank 从小到大反复套用规则，直到没有规则能匹配
"""

import json
import re
import time
from collections import Counter

# ── 🔶 规格部分：预切分正则（照抄，理解每条分支的作用）────────────
# 生产级预切分正则（与 nanochat / tiktoken 的 SPLIT_PATTERN 同构，
# 只是把 \p{L}\p{N} 换成 Python re 的 \w、把数字长度改成可配）
#
# ★ 关键：最后一个非空白兜底分支「 ?[^\s\w]++[\r\n]*」不能省。
#   `++` 是「一次或多次」，它保证标点符号（? ! , . : ; 等）被吃掉。
#   如果写成 `[^\r\n\w]?+\w+`（`?+` 是「零或一次」），`?` 就匹配不上，
#   而 re.finditer **只返回匹配、不报告跳过** —— 那个问号会静默消失，
#   解码时才发现对不上原文。这是最隐蔽的一类 bug（本章「坑 B」）。
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

    # ── 阶段 1：预切分 ───────────────────────────────────────
    def pre_tokenize(self, text: str) -> list[bytes]:
        """
        把文本切成不可再分的块。合并只能在块内进行。

        ── 你要写的 ──────────────────────────────────────
        1) [m.group(0).encode("utf-8") for m in re.finditer(GPT2_PAT, text)]
        2) ★ 断言 b"".join(blocks) == text.encode("utf-8")

        ── 为什么第 2 步是本章最值得抄走的东西 ──────────────
          re.finditer **只返回匹配、不报告跳过**。正则少写一个分支，
          那个字符就被静默丢掉，函数照样正常返回，没有任何报错。
          只有「所有块拼起来 == 原文」这个断言能当场抓住它。
        """
        raise NotImplementedError(
            "待实现：pre_tokenize ——\n"
            "  blocks = [m.group(0).encode('utf-8') for m in re.finditer(GPT2_PAT, text)]\n"
            "  assert b''.join(blocks) == text.encode('utf-8'), (\n"
            "      '预切分正则有字符没匹配上！re.finditer 不会报错，只会静默丢掉它们。'\n"
            "      '通常是缺少标点兜底分支。')\n"
            "  return blocks\n"
            "\n"
            "  验证：uv run python scratch/mini_bpe.py\n"
            "参考实现：git show solution:scratch/mini_bpe.py")

    # ── 阶段 2：训练合并表 ───────────────────────────────────
    def train(self, text: str):
        """
        从语料里学出合并表。

        ── ★ 坑（本章「坑 A」）：符号必须用 bytes，不能用 chr(byte) 转 str ──
          b"12" 是「符号 b'1' 和 b'2' 合并」的结果；
          而 chr(49)+chr(50) 拼出来是字符串 "12"，与真正的 "12" 无法区分。
          后果：merge 表里混进「两个数字字节」的假象，编码时张冠李戴。
          生产实现（rustbpe / tiktoken）全程用 bytes 作为符号，正是为了避开它。

        ── 你要写的（每轮合并做 4 件事）────────────────────
        1) 用 Counter 统计所有块的「单字节符号 tuple」的出现次数
        2) 重复 num_merges 轮：
             ① 统计所有相邻对的**加权**总频率（出现 cnt 次的词贡献 cnt）
             ② pairs 为空 -> break；最高频 < min_frequency -> break
             ③ 记录规则 self.merges[a + b] = len(self.merges)  ← rank 就是训练顺序
             ④ 把所有词里的 (a, b) 替换成 a+b，统计成新的 Counter
        """
        raise NotImplementedError(
            "待实现：train ——\n"
            "  words = Counter()\n"
            "  for block in self.pre_tokenize(text):\n"
            "      words[self._symbols(block)] += 1      # 初始符号 = 每个单字节\n"
            "  for _ in range(self.num_merges):\n"
            "      pairs = Counter()\n"
            "      for word, cnt in words.items():\n"
            "          for i in range(len(word) - 1):\n"
            "              pairs[(word[i], word[i+1])] += cnt    # ★ 加权频率\n"
            "      if not pairs: break\n"
            "      (a, b), n = pairs.most_common(1)[0]\n"
            "      if n < self.min_frequency: break\n"
            "      self.merges[a + b] = len(self.merges)          # rank = 训练顺序\n"
            "      new_words = Counter()\n"
            "      for word, cnt in words.items():\n"
            "          new_words[self._merge_word(word, a, b)] += cnt\n"
            "      words = new_words\n"
            "\n"
            "  验证：uv run python scratch/mini_bpe.py\n"
            "参考实现：git show solution:scratch/mini_bpe.py")

    @staticmethod
    def _symbols(block: bytes) -> tuple:
        """把一个块拆成「单字节符号」的 tuple。🔶 已给"""
        return tuple(block[i:i + 1] for i in range(len(block)))

    @staticmethod
    def _merge_word(word, a, b):
        """把 word 里所有相邻的 (a, b) 合并成 a+b，返回新 tuple。

        注意用 **tuple** 而不是 list —— Counter 的键必须可哈希。
        （本章「常见坑」：一开始用 list，Counter 直接抛 TypeError。）
        """
        raise NotImplementedError(
            "待实现：_merge_word ——\n"
            "  out, i, ab = [], 0, a + b\n"
            "  while i < len(word):\n"
            "      if i + 1 < len(word) and word[i] == a and word[i+1] == b:\n"
            "          out.append(ab); i += 2\n"
            "      else:\n"
            "          out.append(word[i]); i += 1\n"
            "  return tuple(out)\n"
            "\n"
            "参考实现：git show solution:scratch/mini_bpe.py")

    # ── 阶段 3：推理 ─────────────────────────────────────────
    def _merge_all(self, block: bytes) -> list[bytes]:
        """
        对单个块应用全部 merge 规则。

        ── 关键：不是「按训练顺序走一遍」，而是反复套用、
          每次都挑当前 rank 最小的可匹配对，直到没有规则能匹配。
          这与训练时的贪心过程等价，所以**编码结果是唯一的**。

        ── 你要写的 ──────────────────────────────────────
        1) parts = list(self._symbols(block))
        2) 循环：扫所有相邻对，用 self.merges.get(...) 找 rank 最小的那个
        3) 找到就把这两片合成一片（注意切片赋值的长度要匹配）
        4) 找不到就 break
        """
        raise NotImplementedError(
            "待实现：_merge_all ——\n"
            "  parts = list(self._symbols(block))\n"
            "  while True:\n"
            "      best_rank, best_i = None, None\n"
            "      for i in range(len(parts) - 1):\n"
            "          r = self.merges.get(parts[i] + parts[i+1])\n"
            "          if r is not None and (best_rank is None or r < best_rank):\n"
            "              best_rank, best_i = r, i\n"
            "      if best_i is None: break\n"
            "      parts[best_i:best_i+2] = [parts[best_i] + parts[best_i+1]]\n"
            "  return parts\n"
            "\n"
            "参考实现：git show solution:scratch/mini_bpe.py")

    def encode(self, text: str) -> list[bytes]:
        """先预切分，再对每个块套用全部规则。🔶 已给"""
        out = []
        for block in self.pre_tokenize(text):
            out.extend(self._merge_all(block))
        return out

    def decode(self, tokens: list[bytes]) -> str:
        """bytes 拼回字符串。🔶 已给"""
        return b"".join(tokens).decode("utf-8", errors="replace")

    def save(self, path):
        """merge 表的键是 bytes，json 存不了 -> 存 hex。🔶 已给"""
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
