"""
`src/data/tasks.py` 的测试（卷7 只读代码的回归护栏）。

该模块此前**零测试覆盖**，而它承担 SFT 数据混合与全部评测题库 ——
render_mc 的格式（字母在选项后、等号与字母间无空格）直接决定
「训练目标与推理格式是否 token 级一致」，值得钉死。

注意：这里只测**纯函数**与不依赖网络的类方法。
需要联网拉 Hub 数据集的部分（load_hub_dataset / MMLU / GSM8K 的
真实数据）不在本文件范围 —— 那属于跑 train_sft.sh 时的路径。

跑法：uv run pytest tests/test_tasks.py -v
"""

import numpy as np
import pytest

from data.tasks import (
    HubDataset, render_mc, extract_gsm_answer, GSM_RE, Task,
)


# ===========================================================================
# render_mc —— 多选题格式
# ===========================================================================
def test_render_mc_letter_comes_after_choice():
    """字母放在选项**后面**：`- Paris=F`，不是 `- F: Paris`。"""
    out = render_mc("What is the capital of France?", "ABCD",
                    ["Paris", "Rome", "Bonn", "Berlin"])
    assert "- Paris=A" in out, f"格式不对：\n{out}"
    assert "- Rome=B" in out
    assert "- Berlin=D" in out
    # 明确不该出现的格式
    assert "- A: Paris" not in out


def test_render_mc_no_space_between_equals_and_letter():
    """等号与字母之间不能有空格 —— 否则 tokenizer 切出不同的 token。"""
    out = render_mc("q", "ABCD", ["x", "y", "z", "w"])
    assert "=A" in out
    assert "= A" not in out, f"出现了 '= A'，会与 assistant 的 'A' 不是同一个 token"
    # 每个选项行都该是 - <选项>=<字母>
    for line in out.splitlines():
        if line.startswith("- "):
            assert "= " not in line, f"选项行里有 '= '：{line!r}"


def test_render_mc_contains_question_and_instruction():
    out = render_mc("What is 2+2?", "ABCD", ["3", "4", "5", "6"])
    assert "What is 2+2?" in out
    assert out.startswith("Multiple Choice question:")
    assert "Respond only with the letter" in out


def test_render_mc_handles_fewer_choices_than_letters():
    """选项数少于字母数时不能崩（zip 会自然截断）。"""
    out = render_mc("q", "ABCDEFGH", ["a", "b"])
    assert "- a=A" in out and "- b=B" in out
    assert "=C" not in out


def test_render_mc_handles_more_choices_than_letters():
    """选项比字母多时只渲染到字母用完为止。"""
    out = render_mc("q", "AB", ["a", "b", "c"])
    assert "- a=A" in out and "- b=B" in out
    assert "=C" not in out, "字母用完了就不该再多渲染选项"


def test_render_mc_newline_separated():
    out = render_mc("q", "AB", ["a", "b"])
    assert out.count("\n") >= 3, "每个选项与问题都应各占一行"


# ===========================================================================
# extract_gsm_answer
# ===========================================================================
@pytest.mark.parametrize("text,expect", [
    ("blah blah\n#### 42", "42"),
    ("...\n#### -3", "-3"),
    ("the answer is #### 1,000 dollars", "1000"),   # 逗号被去掉
    ("#### 0", "0"),
    ("#### 3.5", "3.5"),
    ("#### -1,234.5", "-1234.5"),
])
def test_extract_gsm_answer(text, expect):
    assert extract_gsm_answer(text) == expect


@pytest.mark.parametrize("text", [
    "no marker at all",
    "",
    "#### ",              # 有标记但没数字
    None,
])
def test_extract_gsm_answer_returns_none_when_absent(text):
    assert extract_gsm_answer(text) is None


def test_gsm_regex_shape():
    """正则只吃数字/小数点/逗号/负号 —— 字母和单位不会被吃进来。"""
    m = GSM_RE.search("answer #### 42 apples")
    assert m is not None and m.group(1) == "42"
    assert GSM_RE.search("answer #### forty-two") is None, \
        "英文数字不该被匹配"


def test_extract_takes_first_marker_only():
    assert extract_gsm_answer("#### 1\n#### 2") == "1", \
        "只应取第一个 ####（GSM8K 的答案只有一个）"


# ===========================================================================
# HubDataset
# ===========================================================================
class _FakeTable:
    """最小化的 pyarrow Table 替身。"""
    def __init__(self, rows, columns=None):
        self.rows = rows
        self.column_names = list(columns or rows[0].keys())
        self.num_rows = len(rows)

    def __getitem__(self, col):
        return [_FakeCell(r[col]) for r in self.rows]


class _FakeCell:
    def __init__(self, v):
        self.v = v

    def as_py(self):
        return self.v


def test_hubdataset_len_and_getitem():
    ds = HubDataset(_FakeTable([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]))
    assert len(ds) == 2
    assert ds[0] == {"a": 1, "b": "x"}
    assert ds[1] == {"a": 2, "b": "y"}


def test_hubdataset_shuffle_is_deterministic_per_seed():
    t = _FakeTable([{"a": i} for i in range(10)])
    a = HubDataset(t).shuffle(42)
    b = HubDataset(t).shuffle(42)
    idx_a = [a[i]["a"] for i in range(len(a))]
    idx_b = [b[i]["a"] for i in range(len(b))]
    assert idx_a == idx_b, "同 seed 必须给出同顺序（否则训练不可复现）"


def test_hubdataset_shuffle_actually_permutes():
    t = _FakeTable([{"a": i} for i in range(10)])
    plain = [HubDataset(t)[i]["a"] for i in range(10)]
    shuf = HubDataset(t).shuffle(7)
    order = [shuf[i]["a"] for i in range(10)]
    assert sorted(order) == plain, "打乱后元素集合不能变"
    assert order != plain, "不该恰好还是原顺序"


def test_hubdataset_different_seed_differs():
    t = _FakeTable([{"a": i} for i in range(50)])
    d1 = HubDataset(t).shuffle(1)
    d2 = HubDataset(t).shuffle(2)
    o1 = [d1[i]["a"] for i in range(50)]
    o2 = [d2[i]["a"] for i in range(50)]
    assert o1 != o2, "不同 seed 应该给出不同顺序"


def test_hubdataset_shuffle_is_a_permutation():
    t = _FakeTable([{"a": i} for i in range(20)])
    s = HubDataset(t).shuffle(3)
    assert isinstance(s.permutation, np.ndarray)
    assert sorted(s.permutation.tolist()) == list(range(20))


def test_hubdataset_original_untouched_by_shuffle():
    """shuffle 应返回新对象，不改原实例。"""
    t = _FakeTable([{"a": i} for i in range(5)])
    base = HubDataset(t)
    base.shuffle(9)
    assert [base[i]["a"] for i in range(5)] == [0, 1, 2, 3, 4]


# ===========================================================================
# Task 基类契约
# ===========================================================================
def test_task_base_requires_abstract_methods():
    """Task 的抽象接口必须真的抛 NotImplementedError，
    否则子类漏实现时会静默走默认行为。"""
    t = Task()
    for meth, args in [("eval_type", ()), ("num_examples", ()),
                       ("get_example", (0,)), ("evaluate", ({}, ""))]:
        with pytest.raises(NotImplementedError):
            getattr(t, meth)(*args)


def test_task_getitem_calls_get_example():
    """Task.__getitem__ 应直接代理到 get_example。"""
    calls = []

    class T(Task):
        def eval_type(self):
            return "mc"

        def num_examples(self):
            return 10

        def get_example(self, i):
            calls.append(i)
            return {"idx": i}

        def evaluate(self, conversation, completion):
            return 1

    t = T()
    assert t[3] == {"idx": 3}
    assert calls == [3]


def test_task_reward_defaults_to_float_of_evaluate():
    class T(Task):
        def eval_type(self):
            return "mc"

        def num_examples(self):
            return 1

        def get_example(self, i):
            return {}

        def evaluate(self, conversation, completion):
            return 1

    r = T().reward({}, "")
    assert isinstance(r, float), f"reward 应返回 float，得到 {type(r)}"
    assert r == 1.0
