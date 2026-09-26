"""
BOS-aligned best-fit 装箱：可视化 + 统计 + 与其它策略的对比。

第 05 章的动手验证。跑：uv run python scratch/packing_demo.py
"""

import random
import statistics

import pyarrow.parquet as pq

from data.tokenizer import get_tokenizer
from data.dataset import list_parquet_files


# ===========================================================================
# 装箱策略（纯逻辑，脱离 torch 便于理解与对比）
# ===========================================================================
def pack_row_bestfit(buf, capacity):
    """
    BOS-aligned best-fit（nanochat / 本项目采用）。

    规则 1：优先选「能完整放下的最长文档」
            —— 长文档最难安排，先处理能减少浪费
    规则 2：都放不下时，裁剪「最短的」填满
            —— 裁最短的，浪费最少
    返回 (这一行用掉的 token 列表, 统计)
    """
    row, buf = [], list(buf)
    cropped, order = 0, []
    while buf and len(row) < capacity:
        rem = capacity - len(row)
        bi, bl = -1, 0
        for i, d in enumerate(buf):
            if len(d) <= rem and len(d) > bl:
                bi, bl = i, len(d)
        if bi >= 0:
            d = buf.pop(bi)
            row.extend(d)
            order.append((bi, len(d), len(d)))       # (原下标, 原长, 装入长)
        else:
            si = min(range(len(buf)), key=lambda i: len(buf[i]))
            d = buf.pop(si)
            row.extend(d[:rem])
            cropped += len(d) - rem
            order.append((si, len(d), rem))
            break
    return row, dict(cropped=cropped, order=order)


def pack_row_firstfit(buf, capacity):
    """first-fit：按 buffer 顺序取第一个能完整放下的（对照组）。"""
    row, buf = [], list(buf)
    cropped, order = 0, []
    while buf and len(row) < capacity:
        rem = capacity - len(row)
        bi = next((i for i, d in enumerate(buf) if len(d) <= rem), -1)
        if bi >= 0:
            d = buf.pop(bi)
            row.extend(d)
            order.append((bi, len(d), len(d)))
        else:
            si = min(range(len(buf)), key=lambda i: len(buf[i]))
            d = buf.pop(si)
            row.extend(d[:rem])
            cropped += len(d) - rem
            order.append((si, len(d), rem))
            break
    return row, dict(cropped=cropped, order=order)


def pack_row_worstcrop(buf, capacity):
    """反面对照：规则 2 改成「裁最长的」。浪费必然更多。"""
    row, buf = [], list(buf)
    cropped, order = 0, []
    while buf and len(row) < capacity:
        rem = capacity - len(row)
        bi, bl = -1, 0
        for i, d in enumerate(buf):
            if len(d) <= rem and len(d) > bl:
                bi, bl = i, len(d)
        if bi >= 0:
            d = buf.pop(bi)
            row.extend(d)
            order.append((bi, len(d), len(d)))
        else:
            si = max(range(len(buf)), key=lambda i: len(buf[i]))   # 裁最长的！
            d = buf.pop(si)
            row.extend(d[:rem])
            cropped += len(d) - rem
            order.append((si, len(d), rem))
            break
    return row, dict(cropped=cropped, order=order)


PACKERS = {"best-fit": pack_row_bestfit,
           "first-fit": pack_row_firstfit,
           "裁最长的": pack_row_worstcrop}


def run_packer(docs, capacity, packer, max_rows=200):
    """
    用给定策略把 docs 装进若干行，返回统计。

    ★ 关键：必须按「实际用掉了哪些文档」来推进缓冲区。
      早先的写法是 `buf = buf[len(st["order"]):]`（按个数从队首删），
      这是错的 —— best-fit 挑的是**靠后**的长文档（比如下标 818），
      但队首只前进了 1 个，缓冲区几乎没消耗，跑 200 行都在同一批文档里打转，
      测出来的裁剪率恒为 0。
    """
    buf = list(docs)
    rows, used, cropped, ndocs, full_docs = 0, 0, 0, 0, 0
    while buf and rows < max_rows:
        row, st = packer(buf, capacity)
        if not row:
            break
        # 按 order 里记录的**原下标**删掉真正用掉的文档
        used_idx = sorted({i for i, _, _ in st["order"]})
        drop = set(used_idx)
        buf = [d for k, d in enumerate(buf) if k not in drop]
        rows += 1
        used += len(row)
        cropped += st["cropped"]
        ndocs += len(st["order"])
        full_docs += sum(1 for _, a, b in st["order"] if a == b)
    return dict(rows=rows, used=used, cropped=cropped, ndocs=ndocs,
                full_docs=full_docs)


# ===========================================================================
# 可视化
# ===========================================================================
def viz(doc_lengths, capacity, width=72):
    """把一行画成 ASCII 图，每篇文档一个颜色字母。"""
    scale = width / capacity
    bar = ["·"] * width
    colors = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    pos = 0
    for k, (orig, loaded) in enumerate(doc_lengths):
        w = max(1, int(loaded * scale))
        c = colors[k % len(colors)]
        for j in range(w):
            if pos + j < width:
                bar[pos + j] = c
        pos += w
    return "".join(bar)


def viz_cut(order, capacity, width=72):
    """
    画出一行的构成。
    小写字母 = 这篇文档被**完整**装入
    大写字母 = 这篇文档被**裁剪**了
    """
    scale = width / capacity
    bar = ["·"] * width
    letters = "abcdefghijklmnopqrstuvwxyz"
    pos = 0
    for k, (_, orig, loaded) in enumerate(order):
        w = max(1, int(loaded * scale))
        c = letters[k % 26]
        if loaded < orig:
            c = c.upper()
        for j in range(w):
            if pos + j < width:
                bar[pos + j] = c
        pos += w
    return "".join(bar)


# ===========================================================================
if __name__ == "__main__":
    random.seed(0)
    tok = get_tokenizer()
    bos = tok.get_bos_token_id()

    # ── 载入真实文档 ──
    print("=== 读取真实 ClimbMix 文档 ===")
    pf = pq.ParquetFile(list_parquet_files("train")[0])
    texts = []
    for rg_idx in range(2):
        texts += pf.read_row_group(rg_idx).column("text").to_pylist()
    texts = texts[:4000]
    docs = [tok.encode(t, prepend=bos) for t in texts]
    # ★ 必须打乱：parquet 里的文档可能按长度聚集，不打乱的话
    #   run_packer 会一直取到同一批短文档，测不出真实的裁剪率。
    random.shuffle(docs)
    lens = sorted(len(d) for d in docs)
    q = lambda p: lens[int(p * (len(lens) - 1))]
    print(f"文档数 {len(docs)}  长度 min={lens[0]} p25={q(.25)} "
          f"中位数={q(.5)} p75={q(.75)} max={lens[-1]}")
    print(f"（>2048 token 的占 {sum(1 for x in lens if x > 2048)/len(lens)*100:.1f}%）")

    # ── 实验 A：不同 seq_len 下的效果 ──
    print("\n=== A. 不同 seq_len 下的装箱效果（best-fit）===")
    print(f"{'T':>6} {'行数':>5} {'填充率':>8} {'每行文档数':>10} "
          f"{'完整装入率':>10} {'裁剪/已用token':>13}")
    for T in (256, 512, 1024, 2048, 4096):
        st = run_packer(docs, T + 1, pack_row_bestfit)
        fill = st["used"] / (st["rows"] * (T + 1)) * 100
        crop_ratio = st["cropped"] / max(st["used"], 1) * 100
        print(f"{T:6d} {st['rows']:5d} {fill:7.1f}% {st['ndocs']/st['rows']:10.2f} "
              f"{st['full_docs']/max(st['ndocs'],1)*100:9.1f}% {crop_ratio:12.1f}%")
    print("\n  · 填充率恒为 100%：这是设计目标（无 padding）")
    print("  · 裁剪占比随 T 增大而下降：行越长，越装得下更多完整文档")

    # ── 实验 B：三种装箱策略对比 ──
    print("\n=== B. 三种装箱策略对比（T=512）===")
    print(f"{'策略':>10} {'填充率':>8} {'每行文档数':>10} {'完整装入率':>10} {'裁剪/已用token':>13}")
    for name, fn in PACKERS.items():
        st = run_packer(docs, 513, fn)
        fill = st["used"] / (st["rows"] * 513) * 100
        crop = st["cropped"] / max(st["used"], 1) * 100
        print(f"{name:>10} {fill:7.1f}% {st['ndocs']/st['rows']:10.2f} "
              f"{st['full_docs']/max(st['ndocs'],1)*100:9.1f}% {crop:12.1f}%")
    print("\n  best-fit 的裁剪量最低 —— 这就是它存在的理由。")
    print("  「裁最长的」是反面对照：规则 2 改成裁最长的，浪费必然更多。")

    # ── 实验 C：手工构造，直观看到 best-fit 的决策 ──
    print("\n=== C. 手工构造：直观看到 best-fit 的决策（容量 200）===")
    made = [[0] * 30, [1] * 35, [2] * 70, [3] * 75, [4] * 100]
    print("文档长度: " + ", ".join(f"{'ABCDE'[i]}={len(d)}" for i, d in enumerate(made)))
    for name, fn in PACKERS.items():
        row, st = fn(made, 200)
        desc = ", ".join(f"{'ABCDE'[i]}({a}->{b})" for i, a, b in st["order"])
        cut = "  ".join(str(a - b) for _, a, b in st["order"] if a > b)
        print(f"  {name:>10}: 装入 {desc}")
        print(f"  {'':>10}  填充 {len(row)}/200  裁剪掉 {st['cropped']} token"
              f"{'  -> ' + cut if cut else ''}")
        print(f"  {'':>10}  [{viz_cut(st['order'], 200)}]   (大写字母=被裁剪的文档)")
    print("\n  图例：`·` 是空隙（不会有，因为始终填满），大写=被裁剪的文档。")
    print("  best-fit 把 100/75 这种大块先安排掉，剩下的空间用小文档填得更整齐。")
