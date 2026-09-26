"""证明 loss 不可比、bpb 可比。第 06 章验证 2。"""
import math

LOSS = 5.0
scenarios = [
    ("字符级（词表 256，每 token 1 字节）", 1.0),
    ("本项目 BPE（词表 8192，实测 5.5 字节）", 5.5),
    ("GPT-2 BPE（词表 50257，约 4.1 字节）", 4.1),
    ("超大词表（词表 256000，约 6.5 字节）", 6.5),
]

print(f"=== 两个模型 loss 都是 {LOSS} nats/token ===\n")
print(f"{'场景':<44} {'bytes/token':>11} {'bpb':>8}")
print("-" * 67)
for name, bpt in scenarios:
    print(f"{name:<44} {bpt:11.2f} {LOSS/math.log(2)/bpt:8.3f}")
print("\n→ 同样的 loss，bpb 相差 5 倍以上。所以 loss 不能跨词表比较。")
print("→ bpb 衡量「预测一个字节要多少 bit」，与分词方式无关。")

BPP = 1.0
print(f"\n=== 反过来：bpb 都是 {BPP} ===\n")
print(f"{'场景':<44} {'bytes/token':>11} {'loss(nats)':>11}")
print("-" * 67)
for name, bpt in scenarios:
    print(f"{name:<44} {bpt:11.2f} {BPP*bpt*math.log(2):11.3f}")
print("\n→ 同样的真实能力，loss 相差 5 倍。bpb 才是那个不变量。")
