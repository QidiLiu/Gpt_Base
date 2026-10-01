"""
词表大小 × 模型深度对参数量的影响。第 02 章的验证脚本。

两段：
  1) 固定 aspect_ratio=64 的「尺度研究」—— 看清 D 变大时嵌入占比怎么变
  2) 本项目**真实档位**的占比 —— 这才是你实际会训的模型

⚠ 为什么第 2 段必须单列：本项目各档的 aspect_ratio 不一样
   （debug=16 / smoke=32 / ablation=64 / full=32），
   所以「d24 是 D=1536」这种从第 1 段推出来的结论**不能套到 full 档**。
   真实情况：full 档的 d24 是 D=768。
"""
import sys

from common.config import PRESETS, build_model_config, make_run_config, scaling_params


def embed_ratio(cfg) -> tuple[float, float, float]:
    """返回 (嵌入参数, 矩阵参数, 占比)。口径同 scaling_params。"""
    embed = 2 * cfg.vocab_size * cfg.n_embd        # wte + lm_head（未绑权重时两份）
    matrices = scaling_params(cfg) - cfg.n_embd * cfg.vocab_size
    return embed, matrices, embed / (embed + matrices)


print("=" * 74)
print("第 1 段：固定 aspect_ratio=64 的尺度研究（V 变化的影响）")
print("=" * 74)
print(f"{'depth':>5} {'D':>5} {'V':>6} {'wte+head':>12} {'矩阵':>12} {'嵌入占比':>8} {'判断':>7}")
print("-" * 74)
for depth in (4, 6, 12, 24):
    for V in (8192, 16384, 32768):
        cfg = build_model_config(depth, 64, 64, sequence_len=1024, vocab_size=V)
        embed, matrices, ratio = embed_ratio(cfg)
        flag = "OK" if ratio < 0.2 else ("勉强" if ratio < 0.4 else "偏大")
        print(f"{depth:5d} {cfg.n_embd:5d} {V:6d} {embed:12,} {matrices:12,} "
              f"{ratio * 100:7.1f}% {flag:>7}")
    print()

print("=" * 74)
print("第 2 段：本项目的真实档位（各自的 aspect_ratio 不同！）")
print("=" * 74)
print(f"{'档位':>10} {'aspect':>7} {'D':>5} {'V':>6} {'嵌入':>12} {'矩阵':>12} {'占比':>7}")
print("-" * 74)
for mode, p in PRESETS.items():
    cfg = make_run_config(mode, vocab_size=p["vocab_size"]).model
    embed, matrices, ratio = embed_ratio(cfg)
    print(f"{mode:>10} {p['aspect_ratio']:7d} {cfg.n_embd:5d} {cfg.vocab_size:6d} "
          f"{embed:12,} {matrices:12,} {ratio * 100:6.1f}%")
print()

print("=" * 74)
print("第 3 段：full 档（d24 / D=768）换词表的代价")
print("=" * 74)
print(f"{'V':>7} {'嵌入':>12} {'占比':>7} {'判断':>8}")
print("-" * 74)
for V in (8192, 16384, 32768, 65536):
    cfg = build_model_config(24, 32, 64, sequence_len=1024, vocab_size=V)
    embed, _, ratio = embed_ratio(cfg)
    flag = "太小，白白浪费压缩率" if ratio < 0.10 else (
        "健康" if ratio < 0.25 else ("勉强" if ratio < 0.35 else "偏大"))
    print(f"{V:7d} {embed:12,} {ratio * 100:6.1f}% {flag:>16}")
print()
print("通用规则：嵌入占比落在 10%~25% 之间最健康。")
print("经验公式 V ≈ 6×D 左右起步，更小的 V 会明显伤压缩率。")
print()
print("★ 本项目选 16384 的真实原因（不是「占比太高」）：")
print("  1) tokenizer 训练成本约 3 倍（合并轮次 3 倍）")
print("  2) 放弃与 nanochat 的 token 级对齐")
print("  3) smoke / ablation 档共用同一个 tokenizer 文件")
sys.stdout.flush()