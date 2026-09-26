"""bpb 的完整手工验证。第 06 章验证 1。"""
import math
import torch
from data.tokenizer import get_tokenizer, get_token_bytes
from data.dataloader import make_dataloader
from common import load_latest, get_runs_dir

tok = get_tokenizer()
tb = get_token_bytes()
n_normal = tb.numel() - 9   # 扣掉 9 个特殊 token

print("=== token_bytes 表的分布 ===")
sizes = {}
for tid in range(n_normal):
    sizes[int(tb[tid])] = sizes.get(int(tb[tid]), 0) + 1
print(f"  普通 token {n_normal} 个，按字节数分布：")
for k in sorted(sizes):
    print(f"    {k} 字节: {sizes[k]:5d} 个 ({sizes[k]/n_normal*100:5.1f}%)")
mean_bytes = tb[:n_normal].sum().item() / n_normal
print(f"  平均每 token {mean_bytes:.3f} 字节")

V = tok.get_vocab_size()
print(f"\n=== 理论参考：均匀分布（随机初始化）===")
print(f"  对 {V} 个 token 完全不确定 -> 每 token {math.log(V):.4f} nats")
print(f"  换算 bpb = log2({V}) / {mean_bytes:.3f} = {math.log(V)/math.log(2)/mean_bytes:.4f}")

model, _, meta = load_latest(get_runs_dir(), "d4", "cuda")
model.eval()
dl = make_dataloader(tok, 8, model.config.sequence_len, "val", "cuda")
tbc = tb.to("cuda")

with torch.no_grad():
    total_nats, total_bytes, total_tokens = 0.0, 0.0, 0
    for i, batch in enumerate(dl):
        if i >= 8:
            break
        x, y = batch[0].to("cuda"), batch[1].to("cuda")
        loss_vec = model(x, y, loss_reduction="none")
        valid = y >= 0
        total_nats += loss_vec[valid].sum().item()
        total_bytes += tbc[y[valid]].sum().item()
        total_tokens += int(valid.sum())

bpb = total_nats / (math.log(2) * total_bytes)
avg_loss = total_nats / total_tokens
print(f"\n=== 训练好的 d4 模型（step {meta.get('step')}）===")
print(f"  平均 loss            = {avg_loss:.4f} nats/token")
print(f"  平均每 token 字节数  = {total_bytes/total_tokens:.3f}")
print(f"  bpb                  = {bpb:.4f}")
print(f"  验证 loss/ln2/bytes  = {avg_loss/math.log(2)/(total_bytes/total_tokens):.4f}  [应等于 bpb]")
print(f"  训练日志记录的 bpb   = {meta.get('best_val_bpb'):.4f}")
print(f"\n  学到的知识量 = {math.log(V)/math.log(2)/mean_bytes:.3f} - {bpb:.3f} "
      f"= {math.log(V)/math.log(2)/mean_bytes - bpb:.3f} bpb 的下降")
