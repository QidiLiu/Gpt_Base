"""手写一个玩具版 BPE（纯 Python），看合并过程。第 02 章验证 2。"""
from collections import Counter


def toy_bpe(text: str, num_merges: int = 20) -> list[tuple[str, str, int]]:
    """
    最朴素的 BPE 实现（为了看清原理，不追求效率）。

    1) 从「字符」开始：每个「词」是一个字符 tuple
    2) 反复合并全局最高频的相邻对
    3) 记录每一步合并了什么

    返回：合并历史 [(符号A, 符号B, 出现次数), ...]

    注意：词用 tuple 而不是 list，因为 Counter 的键必须是可哈希的。
    """
    freq = Counter(tuple(w) for w in text.split())

    history = []
    for _ in range(num_merges):
        # 统计所有相邻对的「加权总频率」
        pairs = Counter()
        for word, cnt in freq.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += cnt
        if not pairs:
            break
        # 取全局最高频的那一对
        best, best_n = pairs.most_common(1)[0]
        if best_n < 2:          # 只出现一次就不值得为它增加词表
            break
        history.append((best[0], best[1], best_n))
        # 把所有 (A,B) 替换成新符号 AB
        new_freq = Counter()
        for word, cnt in freq.items():
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            new_freq[tuple(merged)] += cnt
        freq = new_freq
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
