from data.tokenizer import get_tokenizer
from data.dataloader import make_dataloader

tok = get_tokenizer()
bos = tok.get_bos_token_id()
dl = make_dataloader(tok, batch_size=4, seq_len=512, split="train", device="cpu")
x, y, st = next(dl)

print("inputs ", tuple(x.shape), x.dtype)
print("targets", tuple(y.shape))
print("state  ", st)
print()

# 不变量 1：targets 就是 inputs 右移一位
assert (y[:, :-1] == x[:, 1:]).all(), "targets 必须是 inputs 的右移"
print("[OK] targets == inputs 右移一位")

# 不变量 2：每行以 BOS 开头
assert (x[:, 0] == bos).all(), "每行必须以 BOS 开头"
print("[OK] 每行以 <|bos|> 开头")

# 不变量 3：没有 padding —— 所有 token id 都在 [0, vocab) 内
V = tok.get_vocab_size()
assert ((x >= 0) & (x < V)).all() and ((y >= 0) & (y < V)).all()
print(f"[OK] 无 padding，全部落在 [0, {V})")

# 观察：文档边界 = BOS 的位置
n_bos = (x[0] == bos).sum().item()
print(f"\n第 0 行里有 {n_bos} 个 BOS，即约 {n_bos} 篇文档")
print("前 20 个 token：")
print(" ", tok.visualize(x[0, :20].tolist(), with_token_id=True))
