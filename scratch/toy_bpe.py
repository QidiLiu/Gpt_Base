"""手写一个玩具版 BPE（纯 Python），看合并过程。

第 02 章验证 2 的手抄目标。main 分支上是骨架，完整实现在 solution 分支：
    git show solution:scratch/toy_bpe.py

这是全项目最短的一个手抄目标（约 20 行），但值得自己敲一遍：
不敲一遍，BPE 就只是个黑盒 API。
"""
from collections import Counter


def toy_bpe(text: str, num_merges: int = 20) -> list[tuple[str, str, int]]:
    """
    最朴素的 BPE 实现（为了看清原理，不追求效率）。

    1) 从「字符」开始：每个「词」是一个字符 tuple
    2) 反复合并全局最高频的相邻对
    3) 记录每一步合并了什么

    返回：合并历史 [(符号A, 符号B, 出现次数), ...]

    注意：词用 tuple 而不是 list，因为 Counter 的键必须是可哈希的。
    （第 03 章的 mini_bpe 会踩这个坑：符号要用 bytes 表示。）

    ── 你要写的（每轮合并 4 件事）────────────────────────
    1) freq = Counter(tuple(w) for w in text.split())   ← 按空白切词，每个词是字符 tuple
    2) 重复 num_merges 轮：
         ① 统计所有相邻对的**加权**总频率
            （一个词出现 cnt 次，它内部的相邻对就贡献 cnt）
         ② pairs 为空 -> break
         ③ best, best_n = pairs.most_common(1)[0]；best_n < 2 -> break
            （只出现一次就不值得为它增加词表）
            记 history.append((best[0], best[1], best_n))
         ④ 把所有词里的 (A, B) 替换成新符号 A+B，统计成新的 Counter
    """
    raise NotImplementedError(
        "待实现：toy_bpe ——\n"
        "  freq = Counter(tuple(w) for w in text.split())\n"
        "  history = []\n"
        "  for _ in range(num_merges):\n"
        "      pairs = Counter()\n"
        "      for word, cnt in freq.items():\n"
        "          for i in range(len(word) - 1):\n"
        "              pairs[(word[i], word[i+1])] += cnt     # ★ 加权频率\n"
        "      if not pairs: break\n"
        "      best, best_n = pairs.most_common(1)[0]\n"
        "      if best_n < 2: break\n"
        "      history.append((best[0], best[1], best_n))\n"
        "      new_freq = Counter()\n"
        "      for word, cnt in freq.items():\n"
        "          merged, i = [], 0\n"
        "          while i < len(word):\n"
        "              if i < len(word)-1 and (word[i], word[i+1]) == best:\n"
        "                  merged.append(word[i] + word[i+1]); i += 2\n"
        "              else:\n"
        "                  merged.append(word[i]); i += 1\n"
        "          new_freq[tuple(merged)] += cnt\n"
        "      freq = new_freq\n"
        "  return history\n"
        "\n"
        "  验证：uv run python scratch/toy_bpe.py\n"
        "参考实现：git show solution:scratch/toy_bpe.py")
    return history


if __name__ == "__main__":
    text = """
the quick brown fox jumps over the lazy dog
the fox is quick and the dog is lazy
a quick fox jumps over a lazy dog
"""
    print("=== 合并历史（按发生顺序，频率高的先合）===")
    for i, (a, b, n) in enumerate(toy_bpe(text, 15)):
        print(f"  第 {i+1:2d} 步: 合并 {a!r} + {b!r}   (语料中出现 {n} 次)")

    print("\n=== 换个文本，合并质量如何变化 ===")
    random_text = "zq wv xp kl mn bv tr gh yz cd" * 3
    hist = toy_bpe(random_text, 15)
    print(f"  只合并了 {len(hist)} 次就停了（剩下的相邻对都只出现 1 次）")
    for a, b, n in hist[:5]:
        print(f"  合并 {a!r} + {b!r}  (n={n})")
