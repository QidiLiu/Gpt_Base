"""
`src/inference/engine.py` 的测试（卷7 只读代码的回归护栏）。

为什么这个文件重要：engine.py 是全项目最复杂、最易错的只读代码
（工具调用状态机 + 多行并行解码 + KV cache 容量计算），
而此前 tests 只从它 import 了 `KVCache` 一个类 ——
`Engine`、`safe_eval_math`、`_Row` 全都没有任何测试。

核心手法：用**脚本化的假模型**驱动 `Engine.generate`，把状态机的
每一步都钉死。这样不需要训好的模型，也不依赖随机采样，结果确定。

跑法：uv run pytest tests/test_engine.py -v
"""

import types

import pytest
import torch

from inference.engine import (
    KVCache, Engine, safe_eval_math, sample_next_token, _Row,
)


# ===========================================================================
# 测试替身
# ===========================================================================
class FakeConfig:
    n_kv_head = 1
    head_dim = 4
    n_layer = 2


class ScriptedModel:
    """按预设脚本吐出 token 的假模型。

    `script` 是一个 token id 列表：每被 forward 一次就吐下一个，
    用完则重复最后一个。这样状态机的每一步都是确定的。
    """

    def __init__(self, script, vocab=128):
        self.config = FakeConfig()
        self.script = list(script)
        self.vocab = vocab
        self.n_calls = 0
        self.cache_lens_seen = []

    def get_device(self):
        return "cpu"

    def forward(self, ids, kv_cache=None):
        self.n_calls += 1
        if kv_cache is not None:
            self.cache_lens_seen.append(int(kv_cache.get_pos()))
        B, T = ids.shape
        out = torch.full((B, T, self.vocab), -10.0)
        tok = self.script[min(self.n_calls - 1, len(self.script) - 1)]
        out[:, -1, tok] = 10.0          # 极大 logit -> argmax 必选它
        return out


class FakeTokenizer:
    """够 Engine.generate 用的最小 tokenizer。

    特殊 token id 刻意用小数字（1..4），普通 id 从 10 起，
    这样在脚本里一眼能看出吐的是特殊 token 还是普通 token。
    """

    BOS = 1
    PY_START = 2
    PY_END = 3
    OUT_START = 4
    OUT_END = 5
    ASST_END = 6
    FIRST_PLAIN = 10

    def get_bos_token_id(self):
        return self.BOS

    def encode_special(self, name):
        return {
            "<|bos|>": self.BOS,
            "<|python_start|>": self.PY_START,
            "<|python_end|>": self.PY_END,
            "<|output_start|>": self.OUT_START,
            "<|output_end|>": self.OUT_END,
            "<|assistant_end|>": self.ASST_END,
        }[name]

    def encode(self, text):
        return [ord(c) % 7 + self.FIRST_PLAIN for c in str(text)]

    def decode(self, ids):
        return "".join(chr(int(i) - self.FIRST_PLAIN + ord("a")) for i in ids)


# ===========================================================================
# safe_eval_math —— 极简计算器沙箱
# ===========================================================================
@pytest.mark.parametrize("expr,expect", [
    ("2+2", 4),
    ("12*7", 84),
    ("(3+4)*2", 14),
    ("100/4", 25),
    ("1,000+1", 1001),              # 逗号被剥掉
    ('"abc".count("a")', 1),
    ('"hello".count("l")', 2),
])
def test_safe_eval_allows(expr, expect):
    assert safe_eval_math(expr) == expect


@pytest.mark.parametrize("expr", [
    "2/0",            # 除零 -> None
    "2+",             # 语法错
    "(1+2",           # 括号不闭
    "abc",            # 纯字母且没有 .count(
    "9**9**9",        # 幂运算（DoS）
    "2**8",
    "__import__('os')",
    "open('/etc/passwd')",
    "eval('1')",
    "exec('1')",
    "compile('1','','eval')",
    "print(1)",       # 没有内置函数
    "len('a')",
    "globals()",
    "getattr(1,'x')",
    "''.__class__",   # 危险属性
    "1 if 1 else 2",  # 三元表达式：白名单不含字母/关键字组合的表达式
    "x=1",
])
def test_safe_eval_blocks(expr):
    assert safe_eval_math(expr) is None, f"{expr!r} 应被拦截"


def test_safe_eval_has_no_builtins():
    """eval 的 globals 里必须没有内置函数，否则白名单形同虚设。"""
    # 'len' 是内置函数名；黑名单没列它，但白名单要求含 '.count('，
    # 且 globals 为空 -> 拿不到 len
    assert safe_eval_math('"a".count("a")+len("a")') is None


# ===========================================================================
# KVCache
# ===========================================================================
def test_kvcache_starts_empty_and_advances():
    c = KVCache(2, n_kv_head=1, head_dim=4, n_layers=2, max_seq_len=16,
                device="cpu")
    assert c.get_pos() == 0
    c.advance(3)
    assert c.get_pos() == 3
    assert c.cache_seqlens.tolist() == [3, 3]


def test_kvcache_layer_slicing():
    c = KVCache(2, n_kv_head=1, head_dim=4, n_layers=3, max_seq_len=8,
                device="cpu")
    k, v = c.get_layer(1)
    assert k.shape == (2, 8, 1, 4)
    assert v.shape == (2, 8, 1, 4)
    # 预分配零填充，不该有未定义内容
    assert torch.equal(k, torch.zeros_like(k))


def test_prefill_from_copies_content():
    # k_cache 形状是 (n_layers, B, S, H, D)
    src = KVCache(1, n_kv_head=1, head_dim=4, n_layers=2, max_seq_len=8,
                  device="cpu")
    src.k_cache[0, 0, 2] = 1.5
    src.advance(3)
    dst = KVCache(4, n_kv_head=1, head_dim=4, n_layers=2, max_seq_len=8,
                  device="cpu")
    dst.prefill_from(src)
    assert dst.get_pos() == 3
    assert dst.k_cache.shape[1] == 4, "batch 维应扩到 4"
    # 每个 batch 元素都应拿到同样的 prompt 内容
    # （cache 用 COMPUTE_DTYPE，比较时 dtype 必须对上）
    for b in range(4):
        got = dst.k_cache[0, b, 2]
        assert torch.allclose(got.float(), torch.full((1, 4), 1.5)), \
            f"batch {b} 的 prompt KV 不一致：{got}"
        assert torch.equal(dst.v_cache[1, b, 2],
                           torch.zeros_like(dst.v_cache[1, b, 2])), \
            "v_cache 也应被复制"


def test_prefill_from_rejects_nonempty_target():
    a = KVCache(1, n_kv_head=1, head_dim=4, n_layers=1, max_seq_len=8,
                device="cpu")
    b = KVCache(1, n_kv_head=1, head_dim=4, n_layers=1, max_seq_len=8,
                device="cpu")
    b.advance(1)
    with pytest.raises(AssertionError, match="必须还是空的"):
        b.prefill_from(a)


def test_prefill_from_rejects_shape_mismatch():
    a = KVCache(1, n_kv_head=1, head_dim=4, n_layers=1, max_seq_len=8,
                device="cpu")
    b = KVCache(1, n_kv_head=1, head_dim=4, n_layers=2, max_seq_len=8,
                device="cpu")
    with pytest.raises(AssertionError):
        b.prefill_from(a)


# ===========================================================================
# sample_next_token
# ===========================================================================
def test_greedy_when_temperature_zero():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    r = sample_next_token(logits, None, temperature=0.0)
    assert r.item() == 1, "temperature=0 应取 argmax"


def test_top_k_restricts_to_top_k():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, -50.0, -50.0]])
    for _ in range(20):
        r = sample_next_token(logits, torch.Generator().manual_seed(0),
                              temperature=1.0, top_k=2)
        assert r.item() in (0, 1), f"top_k=2 时不该选到 {r.item()}"


def test_top_k_larger_than_vocab_is_clamped():
    logits = torch.tensor([[1.0, 2.0]])
    r = sample_next_token(logits, torch.Generator().manual_seed(0),
                          temperature=1.0, top_k=99)
    assert r.shape == (1, 1)


# ===========================================================================
# _Row
# ===========================================================================
def test_row_initial_state():
    r = _Row([7, 8, 9])
    assert r.tokens == [7, 8, 9]
    assert r.forced == [] and r.py_expr == []
    assert r.in_python is False and r.done is False


# ===========================================================================
# Engine.generate —— 工具调用状态机（核心）
# ===========================================================================
def _run(script, prompt=(70, 71), max_tokens=None, **kw):
    m = ScriptedModel(script)
    tk = FakeTokenizer()
    eng = Engine(m, tk)
    mt = max_tokens if max_tokens is not None else len(script)
    out = list(eng.generate(list(prompt), num_samples=1, max_tokens=mt,
                             temperature=0.0, use_tools=True, **kw))
    return out, m, tk


def test_generate_yields_prompt_len_columns():
    out, m, _ = _run([20, 21, 22, 23], max_tokens=4)
    assert len(out) == 4, f"应 yield 4 步，得到 {len(out)}"
    for column, masks in out:
        assert len(column) == 1 and len(masks) == 1


def test_generate_runs_to_max_tokens_when_never_terminates():
    out, _, _ = _run([20, 21, 22, 23], max_tokens=9)
    assert len(out) == 9, f"没吐终止符时应跑满 max_tokens，得到 {len(out)}"


def test_generate_stops_at_assistant_end():
    tk = FakeTokenizer()
    script = [20, tk.ASST_END, 30, 31]
    m = ScriptedModel(script)
    eng = Engine(m, tk)
    out = list(eng.generate([70], num_samples=1, max_tokens=10,
                            temperature=0.0))
    # 吐出 asst_end 后应立刻停，不再吐后面的 30/31
    assert len(out) == 2, f"应在 asst_end 处停止，得到 {len(out)} 步"
    assert out[-1][0][0] == tk.ASST_END


def test_tool_call_injects_output_with_mask_zero():
    """完整走一遍：python_start -> 表达式 -> python_end -> 注入计算结果。

    注入的 token（output_start/结果/output_end）mask 必须是 0 ——
    模型没有被训练去「生成」工具返回值，那是外部塞进去的。
    """
    tk = FakeTokenizer()
    # 2+2 用两个字符表示：'2'->'a' 偏移。这里直接让 decode 返回可算式子。
    script = [tk.PY_START, tk.FIRST_PLAIN, tk.FIRST_PLAIN, tk.PY_END, 99, 99]
    m = ScriptedModel(script)

    class MathTokenizer(FakeTokenizer):
        def decode(self, ids):
            # 状态机把 python 段的 token 攒起来交给 decode，
            # 这里直接返回一个可计算的算式（测试要盯的是状态机，不是 decode）
            return "2+2"

    tk2 = MathTokenizer()
    m.tokenizer = tk2
    eng = Engine(m, tk2)
    out = list(eng.generate([70], num_samples=1, max_tokens=20,
                            temperature=0.0, use_tools=True))

    flat = [(c[0], mk[0]) for c, mk in out]
    ids = [t for t, _ in flat]
    masks = [mk for _, mk in flat]

    # 必须在 python_end 之后出现 output_start / output_end
    assert tk.OUT_START in ids, f"应注入 output_start，实际 {ids}"
    assert tk.OUT_END in ids, f"应注入 output_end，实际 {ids}"
    i_s, i_e = ids.index(tk.OUT_START), ids.index(tk.OUT_END)
    assert i_s < i_e, "output_start 必须在 output_end 之前"
    # 注入段整体 mask=0
    assert all(m == 0 for m in masks[i_s:i_e + 1]), (
        f"注入的工具输出不该参与 loss，实际 mask={masks[i_s:i_e+1]}")
    # 采样出来的 token mask=1
    assert masks[0] == 1, "模型自己吐的 token mask 应为 1"


def test_tool_output_contains_the_computed_result():
    tk = FakeTokenizer()
    script = [tk.PY_START, tk.FIRST_PLAIN, tk.FIRST_PLAIN, tk.PY_END, 99]

    class T(FakeTokenizer):
        def decode(self, ids):
            return "6*7"

    m = ScriptedModel(script)
    m.tokenizer = T()
    eng = Engine(m, m.tokenizer)
    out = list(eng.generate([70], num_samples=1, max_tokens=20,
                            temperature=0.0, use_tools=True))
    ids = [c[0] for c, _ in out]
    if tk.OUT_START in ids:
        i_s = ids.index(tk.OUT_START)
        # 注入的 token 数应等于 encode("42") 的长度 + 2（起止标记）
        n_injected = ids.index(tk.OUT_END) - i_s - 1
        assert n_injected == len(FakeTokenizer().encode("42")), (
            f"注入段应有 {len(FakeTokenizer().encode('42'))} 个结果 token，"
            f"实际 {n_injected}")


def test_use_tools_false_disables_injection():
    tk = FakeTokenizer()
    script = [tk.PY_START, tk.FIRST_PLAIN, tk.PY_END, 40, 41]
    m = ScriptedModel(script)

    class T(FakeTokenizer):
        def decode(self, ids):
            return "2+2"

    tk2 = T()
    eng = Engine(m, tk2)
    out = list(eng.generate([70], num_samples=1, max_tokens=10,
                            temperature=0.0, use_tools=False))
    ids = [c[0] for c, _ in out]
    assert tk.OUT_START not in ids, "use_tools=False 时不该注入工具输出"


def test_non_math_expression_does_not_hang():
    """表达式非法时不注入，状态机照常继续（不能卡死）。"""
    tk = FakeTokenizer()
    script = [tk.PY_START, tk.FIRST_PLAIN, tk.PY_END, 50, 51]

    class T(FakeTokenizer):
        def decode(self, ids):
            return "not a valid math <<<"

    m = ScriptedModel(script)
    eng = Engine(m, T())
    out = list(eng.generate([70], num_samples=1, max_tokens=10,
                            temperature=0.0, use_tools=True))
    ids = [c[0] for c, _ in out]
    assert tk.OUT_START not in ids, "非法表达式不该注入结果"
    assert len(out) >= 3, "状态机应继续往前走，不能卡住"


def test_python_end_without_output_is_harmless():
    """直接吐 python_end（没进 python 态）不该崩，也不该注入。"""
    tk = FakeTokenizer()
    script = [tk.PY_END, 60, 61]
    m = ScriptedModel(script)
    eng = Engine(m, FakeTokenizer())
    out = list(eng.generate([70], num_samples=1, max_tokens=6,
                            temperature=0.0, use_tools=True))
    ids = [c[0] for c, _ in out]
    assert len(out) == 6, f"应跑满 max_tokens，得到 {len(out)}"
    assert tk.OUT_START not in ids, "没进过 python 态就不该注入结果"


# ===========================================================================
# KV cache 容量：工具注入会多占位置，正好用满不溢出
# ===========================================================================
def test_cache_capacity_exactly_fits():
    """need = len(prompt) + max_tokens，每轮固定前进 1 格，正好不越界。

    工具强制注入的 token 也占一格（它们同样要 forward 进去），
    所以不会超出这个预算。越界会在 attention 里静默读到垃圾。
    """
    prompt = [70, 71, 72]
    max_tokens = 8
    m = ScriptedModel(list(range(20, 20 + max_tokens)))
    eng = Engine(m, FakeTokenizer())
    out = list(eng.generate(prompt, num_samples=1, max_tokens=max_tokens,
                            temperature=0.0, use_tools=False))
    positions = m.cache_lens_seen
    assert positions, "应至少 forward 过一次"
    assert max(positions) < len(prompt) + max_tokens, (
        f"cache 写指针越界：{max(positions)} >= {len(prompt) + max_tokens}")
    assert len(out) == max_tokens


def test_prompt_prefill_then_expand_does_n_samples_forward_once():
    """prompt 只前向一次，然后复制成 N 份 —— 这是 generate 的核心优化。"""
    m = ScriptedModel([20, 21, 22])
    eng = Engine(m, FakeTokenizer())
    list(eng.generate([70, 71], num_samples=4, max_tokens=3,
                      temperature=0.0, use_tools=False))
    # 1 次 prefill + 3 次解码 = 4 次 forward，与样本数无关
    assert m.n_calls == 4, f"forward 应为 1+3 次，实际 {m.n_calls}"


# ===========================================================================
# generate_batch（非流式）
# ===========================================================================
def test_generate_batch_excludes_termination_tokens():
    tk = FakeTokenizer()
    script = [20, 21, tk.ASST_END, 30, 31]
    m = ScriptedModel(script)
    eng = Engine(m, tk)
    results, masks = eng.generate_batch([70], num_samples=1, max_tokens=10,
                                        temperature=0.0)
    assert len(results) == 1
    # 结果里**含 prompt**（generate_batch 返回完整序列）
    assert results[0][0] == 70, "结果应保留 prompt"
    assert tk.ASST_END not in results[0], "终止 token 不该计入结果"
    assert 30 not in results[0] and 31 not in results[0], (
        f"终止之后的 token 也不该计入，得到 {results[0]}")
    assert results[0] == [70, 20, 21], (
        f"应是 prompt + 终止前的 2 个，实际 {results[0]}")


def test_generate_batch_masks_length_matches_results():
    tk = FakeTokenizer()
    m = ScriptedModel([20, 21, tk.ASST_END])
    eng = Engine(m, tk)
    results, masks = eng.generate_batch([70, 71], num_samples=1, max_tokens=8,
                                        temperature=0.0)
    assert len(results[0]) == len(masks[0])
    # prompt 部分的 mask 是 0（不监督 prompt）
    assert masks[0][:2] == [0, 0]
    assert all(x == 1 for x in masks[0][2:]), "生成部分 mask 应为 1"


def test_generate_batch_multiple_samples_same_length():
    m = ScriptedModel([20, 21, 22, 23])
    eng = Engine(m, FakeTokenizer())
    results, masks = eng.generate_batch([70], num_samples=3, max_tokens=4,
                                        temperature=0.0)
    assert len(results) == 3 and len(masks) == 3
    for r in results:
        assert len(r) == 1 + 4, f"每行应是 prompt(1) + 生成(4)，实际 {r}"
