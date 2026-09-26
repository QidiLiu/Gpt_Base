"""loss mask 演示。第 04 章验证 1-3。"""
import torch
import torch.nn.functional as F
from data.tokenizer import get_tokenizer

tok = get_tokenizer()

# ── 验证 1：简单对话的 mask ──
conv = {"messages": [
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]}
ids, mask = tok.render_conversation(conv)
print(f"{'id':>6} {'mask':>5}  token")
print("-" * 40)
for t, m in zip(ids, mask):
    print(f"{t:6d} {m:5d}  {tok.decode([t])!r}")
print(f"\n参与 loss 的 token: {sum(mask)} / {len(mask)}")
learned_ids = [t for t, m in zip(ids, mask) if m == 1]
print(f"它们拼起来是: {tok.decode(learned_ids)!r}")
# 注意监督信号里**包含** <|assistant_end|>（mask=1 是刻意的）：
# 模型必须学会「什么时候停」，所以收尾 token 也要学。
assert tok.decode(learned_ids).startswith("Paris."), "监督信号应以 assistant 的回答开头"
assert learned_ids[-1] == tok.encode_special("<|assistant_end|>"), "监督信号应以 <|assistant_end|> 结尾"
print("[OK] 监督信号 = assistant 的回答 + <|assistant_end|>（教模型何时停）")

# ── 验证 2：带工具调用的对话 ──
print("\n" + "=" * 60)
print("带工具调用的对话（GSM8K 的格式）")
print("=" * 60)
conv2 = {"messages": [
    {"role": "user", "content": "Mimi has 2 x 12 = "},
    {"role": "assistant", "content": [
        {"type": "text",          "text": "2*12"},
        {"type": "python_output", "text": "24"},
        {"type": "text",          "text": " sea shells."},
    ]},
]}
ids2, mask2 = tok.render_conversation(conv2)
print(tok.visualize(ids2, mask2))
print("\n绿=参与loss(1)  红=不参与(0)")
learned = tok.decode([t for t, m in zip(ids2, mask2) if m == 1])
print(f"模型要学的: {learned!r}")
assert "24" not in learned, "python_output 的结果不该被学"
assert "2*12" in learned and "sea shells" in learned
print("[OK] 工具调用表达式和自然语言要学，工具返回值不学")

# ── 验证 3：截断策略把监督信号全切掉 ──
print("\n" + "=" * 60)
print("截断策略：为什么 SFT 必须 truncate='left'")
print("=" * 60)
long_conv = {"messages": [
    {"role": "user", "content": "问题" * 200},        # 超长 user
    {"role": "assistant", "content": "这是一个很长很长的回答。" * 40},
]}
for mode in ("right", "left"):
    ids3, mask3 = tok.render_conversation(long_conv, max_tokens=128, truncate=mode)
    n_sup = sum(mask3)
    # 模拟 GPT.forward：随机 logits（词表取 10000 足够容纳所有真实 id）
    V = 10000
    tgt = torch.tensor([i - 1 for i in ids3[1:]], dtype=torch.long)   # 右移一位
    keep = torch.tensor([m for m in mask3[:-1]], dtype=torch.bool)
    tgt = torch.where(keep, tgt, torch.full_like(tgt, -1))            # mask=0 -> -1
    loss = F.cross_entropy(torch.randn(len(tgt), V), tgt, ignore_index=-1)
    print(f"  truncate={mode:5s}: {len(ids3)} token, 参与loss {n_sup:4d} 个"
          f"  ->  实际 loss = {loss.item()}")
