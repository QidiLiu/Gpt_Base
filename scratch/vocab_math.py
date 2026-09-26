from common.config import build_model_config, scaling_params

print("词表大小对参数量的影响（aspect_ratio=64）")
print(f"{'depth':>5} {'D':>5} {'V':>6} {'wte+head':>12} {'scaling总参数':>14} {'嵌入占比':>8} {'<20%?':>7}")
print("-" * 66)
for depth in (4, 6, 12, 24):
    for V in (8192, 16384, 32768):
        cfg = build_model_config(depth, 64, 64, sequence_len=1024, vocab_size=V)
        embed = 2 * V * cfg.n_embd          # wte + lm_head（不绑权重时是两份）
        total = embed + (scaling_params(cfg) - cfg.n_embd * V)  # scaling_params 已含 lm_head
        ratio = embed / total
        flag = "OK" if ratio < 0.2 else "偏大"
        print(f"{depth:5d} {cfg.n_embd:5d} {V:6d} {embed:12,} {total:14,} "
              f"{ratio*100:7.1f}% {flag:>7}")
    print()
