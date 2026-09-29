"""
推理引擎：带 KV cache 的高效生成。

对应教程卷7 第 37-38 章。

────────────────────────────────────────────────────────────────
为什么需要 KV cache？
────────────────────────────────────────────────────────────────
朴素生成（model.generate）每生成一个 token 就把整条序列重新前向一次：

    生成第 1 个 token：前向 1 个位置   -> 算 1 次
    生成第 2 个 token：前向 2 个位置   -> 算 2 次
    ...
    生成第 L 个 token：前向 L 个位置   -> 算 L 次
    总计 O(L²) 次矩阵乘

但注意：**第 2 次前向里，前 L-1 个位置的 k 和 v 与第 1 次完全相同**
（因为输入没变，模型是因果的）。这部分计算是纯浪费。

KV cache 就是把每个位置算好的 k、v 存下来，下次只算新增的那个位置：

    生成第 t 个 token：只前向 1 个位置
    总计 O(L) 次矩阵乘

对 L=1000 的序列，这是 1000 倍的计算量差距。
推理（decode）阶段的瓶颈是**显存带宽**而不是算力
（每步只算 1 个 token，要读整个模型的权重 + 整个 KV cache），
所以缓存 KV 同时也大幅降低了带宽压力。

────────────────────────────────────────────────────────────────
形状约定
────────────────────────────────────────────────────────────────
    q      (B, T, H,  D)   T 是「本次要算的位置数」
    k_cache(v_cache)  (B, S, H, D)   S 是「cache 容量（含已缓存的）」
    cache_seqlens      (B,)   每个 batch 元素已填到第几个位置

与 nanochat / FA3 的约定一致：head 维在最后，T 在第 1 维。
"""

import torch
import torch.nn.functional as F

from common import COMPUTE_DTYPE


class KVCache:
    """
    预分配的 KV cache。

    关键设计：**预分配 + 原地写入**，生成过程中零显存分配。
    如果每个 token 都 `torch.cat` 一次，不仅慢，还会因为
    「旧 tensor 不能释放」而在显存上出现锯齿状峰值。
    """

    def __init__(self, batch_size, n_kv_head, head_dim, n_layers,
                 max_seq_len, device, dtype=None):
        dtype = dtype or COMPUTE_DTYPE
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # 形状 (n_layers, B, S, H, D)：层放最外层，方便一次 get_layer 切片
        self.k_cache = torch.zeros(n_layers, batch_size, max_seq_len,
                                   n_kv_head, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(n_layers, batch_size, max_seq_len,
                                   n_kv_head, head_dim, device=device, dtype=dtype)
        # 已填到第几个位置（FA3 需要 int32）
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Smear trick 需要记住上一个 token 的嵌入
        self.prev_embedding = None

    def get_layer(self, layer_idx: int):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get_pos(self) -> int:
        """当前写指针位置（假设 batch 内所有行同步前进）。"""
        return int(self.cache_seqlens[0].item())

    def advance(self, n_tokens: int) -> None:
        self.cache_seqlens += n_tokens

    def prefill_from(self, other: "KVCache") -> None:
        """
        把另一个 cache 的内容复制进来并扩容 batch 维。

        用途：batch=1 prefill 一次 prompt，然后复制成 N 份并行采样 N 条回复。
        这是 RL rollout 的关键优化 —— prompt 的前向只算一次，
        而不是 N 次。见 generate() 的第 1、2 步。
        """
        assert self.get_pos() == 0, "目标 cache 必须还是空的"
        assert self.n_layers == other.n_layers
        assert self.n_kv_head == other.n_kv_head
        assert self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        pos = other.get_pos()
        self.k_cache[:, :, :pos] = other.k_cache[:, :, :pos]
        self.v_cache[:, :, :pos] = other.v_cache[:, :, :pos]
        self.cache_seqlens.fill_(pos)
        if other.prev_embedding is not None:
            # 扩 batch：(1, 1, D) -> (B, 1, D)
            self.prev_embedding = other.prev_embedding.expand(
                self.batch_size, -1, -1).clone()

    def memory_bytes(self) -> int:
        return self.k_cache.numel() * self.k_cache.element_size() * 2


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """
    从 (B, vocab) 的 logits 采出 (B, 1) 的 token。

    采样顺序（很重要）：
        1) temperature == 0  -> 直接 argmax（贪心，完全确定）
        2) top_k 裁剪        -> 只保留概率最高的 k 个，其余置 -inf
        3) 除以 temperature
        4) softmax
        5) multinomial

    为什么 top_k 要在除温度之前？
      温度是对「logits 做缩放」，它不改变排序。
      先除温度再 top_k，被裁掉的 -inf 位置依然是 -inf，结果一样 ——
      但先 top_k 可以保证「只有 k 个数参与后续计算」，
      在 vocab 很大时是实打实的性能差异。
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = logits.topk(k, dim=-1)
        vals = vals / temperature
        choice = F.softmax(vals, dim=-1).multinomial(1, generator=rng)
        return idx.gather(1, choice)
    probs = F.softmax(logits / temperature, dim=-1)
    return probs.multinomial(1, generator=rng)


# ===========================================================================
# 推理侧 prompt 渲染
# ===========================================================================
def render_chat_prompt(tokenizer, user_text: str,
                      system: str | None = None) -> list[int]:
    """
    把「一条 user 消息 + 等待 assistant 开口」渲染成 token id 序列。

    返回的序列收在 ``<|assistant_start|>``，调用方 forward 整段之后，
    decode 追加出来的部分就是 assistant 的回复。

    ── ★ 为什么不能 f-string 拼特殊 token ──────────────────────
    BPETokenizer.encode() 内部走的是 tiktoken 的 **encode_ordinary**，
    而 encode_ordinary 的定义就是「**忽略** special tokens」——
    它会把 "<|user_start|>" 当成普通文本再切一遍。

    实测（8192 词表的真实 tokenizer，文本 "What is 12 * 7?"）：

        错：tokenizer.encode(f"<|user_start|>{q}<|user_end|><|assistant_start|>")
            -> 34 个 token，特殊 id 8184 **根本没出现**，
               被切成 [60, 124, 305, 263, 95, 321, ...]
        对：render_chat_prompt(tok, q)
            -> 12 个 token，8184 / 8185 / 8186 各出现 1 次

    两边 decode 回文本**完全一样**，所以这个 bug 在日志里看不出来 ——
    但训练时（render_conversation 用 encode_special）模型只见过 1 个 id，
    评测时喂 12 个，格式对不上，准确率因此失去意义。

    ── 正确写法 ───────────────────────────────────────────────
    特殊 token 用 encode_special 逐个取 id，普通文本用 encode 编码，
    最后把 id 列表拼起来。
    """
    S = tokenizer.encode_special
    ids = [tokenizer.get_bos_token_id()]
    if system:
        # system 合并进 user（与 render_conversation 的约定一致）
        user_text = f"{system}\n\n{user_text}"
    ids += [S("<|user_start|>")]
    ids += tokenizer.encode(user_text)
    ids += [S("<|user_end|>"), S("<|assistant_start|>")]
    return ids


# ===========================================================================
# 工具调用状态机
# ===========================================================================
class _Row:
    """生成过程中每一行的状态。"""

    def __init__(self, tokens: list[int]):
        self.tokens = list(tokens)
        self.forced = []       # 待强制注入的 token（工具输出）
        self.in_python = False
        self.py_expr = []
        self.done = False


def safe_eval_math(expr: str, max_seconds: int = 2):
    """
    极简的「计算器工具」：只允许纯算术表达式和 str.count()。

    安全措施（四道）：
      1. 字符白名单 —— 纯算术只放行数字与 `*+-/.()`；
         带字母的分支另有一套白名单（字母/数字/引号/括号/点/下划线/空格）
      2. 危险词黑名单 —— __ / import / eval / open / getattr ...
      3. __builtins__ 清空 —— eval 的 globals 为空，拿不到任何内置函数。
         光靠黑名单挡不住 `print` / `len` 这类**本身不算危险词**的内置函数，
         是这一条兜住的
      4. signal.SIGALRM 超时 —— 兜底，防止慢表达式把 CPU 卡死

    关于 `9**9**9`：它其实是被第 1 条里的 `**` 检查直接拒掉的
    （`if "**" in expr: return None`），**不是**靠超时。
    超时只是最后一道保险 —— 例如超长表达式的正常求值。

    两条已知限制：
      · signal.SIGALRM 只在**主线程**可用。将来若把生成放进 DataLoader
        worker 或后台线程，`signal.signal()` 会抛 ValueError。
      · 这不是安全沙箱（真正的沙箱要隔离进程 + seccomp / seccomp-bpf），
        只够在本地玩具模型上用。教程卷7 会说明这个区别。
    """
    import signal

    expr = expr.replace(",", "")
    pure_math = all(ch in "0123456789*+-/.() " for ch in expr)
    if pure_math:
        if "**" in expr:
            return None  # 禁掉幂运算，防 9**9**9
    else:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                      "0123456789'\"()._ ")
        if not all(ch in allowed for ch in expr):
            return None
        bad = ["__", "import", "exec", "eval", "compile", "open", "file",
               "input", "globals", "locals", "vars", "dir", "getattr",
               "setattr", "delattr", "hasattr"]
        low = expr.lower()
        if any(b in low for b in bad):
            return None
        if ".count(" not in expr:
            return None

    def handler(signum, frame):
        raise TimeoutError("计算器超时")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(max_seconds)
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ===========================================================================
# 引擎
# ===========================================================================
class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, tokens: list[int], num_samples: int = 1,
                 max_tokens: int = 128, temperature: float = 1.0,
                 top_k: int = None, seed: int = 42, use_tools: bool = True):
        """
        流式生成。每次 yield (token_column, token_masks)：
            token_column  长度 num_samples 的列表，这一步各行吐出的 token
            token_masks   同长度，1=模型采样得到，0=被强制注入（工具输出）

        三步走：
          1) batch=1 prefill 一次 prompt
          2) 把 KV cache 复制成 num_samples 份（prompt 计算只做一次）
          3) 逐 token 解码，同时跑工具调用状态机
        """
        m = self.model.config
        device = self.model.get_device()
        rng = torch.Generator(device=device).manual_seed(seed)
        kv_kwargs = dict(n_kv_head=m.n_kv_head, head_dim=m.head_dim,
                         n_layers=m.n_layer)

        # ── 1) prefill ──
        prefill_cache = KVCache(1, max_seq_len=max(len(tokens), 1),
                                device=device, **kv_kwargs)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=prefill_cache)[:, -1, :]
        logits = logits.expand(num_samples, -1).contiguous()

        # ── 2) 复制成 N 份 ──
        need = len(tokens) + max_tokens
        cache = KVCache(num_samples, max_seq_len=need, device=device, **kv_kwargs)
        cache.prefill_from(prefill_cache)
        del prefill_cache

        # ── 3) 解码循环 ──
        S = self.tokenizer.encode_special
        py_s, py_e = S("<|python_start|>"), S("<|python_end|>")
        out_s, out_e = S("<|output_start|>"), S("<|output_end|>")
        asst_end, bos = S("<|assistant_end|>"), self.tokenizer.get_bos_token_id()

        rows = [_Row(tokens) for _ in range(num_samples)]
        for _ in range(max_tokens):
            if all(r.done for r in rows):
                break
            next_ids = sample_next_token(logits, rng, temperature, top_k)
            sampled = next_ids[:, 0].tolist()

            column, masks = [], []
            for i, row in enumerate(rows):
                if row.forced:
                    tid, mask = row.forced.pop(0), 0   # 强制注入的 token 不训练
                else:
                    tid, mask = sampled[i], 1
                column.append(tid)
                masks.append(mask)
                row.tokens.append(tid)

                if tid in (asst_end, bos):
                    row.done = True

                # ── 工具调用状态机 ──
                if not use_tools:
                    continue
                if tid == py_s:
                    row.in_python, row.py_expr = True, []
                elif tid == py_e and row.in_python:
                    row.in_python = False
                    if row.py_expr:
                        result = safe_eval_math(self.tokenizer.decode(row.py_expr))
                        if result is not None:
                            row.forced = ([out_s] + self.tokenizer.encode(str(result))
                                          + [out_e])
                    row.py_expr = []
                elif row.in_python:
                    row.py_expr.append(tid)

            yield column, masks
            ids = torch.tensor(column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=cache)[:, -1, :]

    def generate_batch(self, tokens, num_samples=1, **kw):
        """
        非流式版本，返回 (结果列表, mask 列表)。
        终止 token（<|assistant_end|> / BOS）不计入结果。
        """
        asst_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [list(tokens) for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        done = [False] * num_samples
        for column, col_masks in self.generate(tokens, num_samples, **kw):
            for i, (t, mk) in enumerate(zip(column, col_masks)):
                if done[i]:
                    continue
                if t in (asst_end, bos):
                    done[i] = True
                else:
                    results[i].append(t)
                    masks[i].append(mk)
            if all(done):
                break
        return results, masks
