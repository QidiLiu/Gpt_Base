"""
判据自身的健全性自检（不是章节判据）。

背景：这个文件里的测试不测任何「要你敲的函数」，它测的是**判据本身**。
2026-09 的修复中发现：7 个章节判据在完整参考实现 solution 上依然失败，
22 章里有 8 章永远无法标记完成。根因是 HyperParams 的隐式契约
（构造函数只登记 key、值恒为 0）从没被任何测试固化，于是测试作者
按「构造器可以传值」的直觉写错了 3 处，且从未有人拿 solution 跑过判据。

所以这里把那些**隐式前提**显式钉住。任何人不小心破坏它们，本文件会立刻报警。
"""

import ast
import pathlib
import re
import subprocess
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
TESTS = REPO / "tests"


def current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


# ===========================================================================
# 1. HyperParams 的隐式契约：构造函数不赋值
# ===========================================================================
def _hyperparams_cls():
    """拿到 HyperParams；骨架态（main）下它是未实现的，此时跳过契约检查。"""
    sys.path.insert(0, str(SRC))
    from optim.muon import HyperParams
    try:
        HyperParams(step=0.0, lr=0.0)
    except NotImplementedError:
        pytest.skip("HyperParams 还是骨架（main 分支），契约检查留到实现之后")
    return HyperParams


def test_hyperparams_constructor_does_not_assign_values():
    """HyperParams(**kw) 只登记 key，值必须恒为 0。

    这是 2026-09 那批坏判据的直接根因：把 kwargs 当成赋值传进去，
    于是所有超参都是 0，偏差校正里 `1 - beta**0 == 0` 除零产生 NaN。
    把契约钉死，谁改坏都会立刻看到。
    """
    HyperParams = _hyperparams_cls()

    hp = HyperParams(step=1, lr=0.1, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.5)
    for key in ("step", "lr", "beta1", "beta2", "eps", "wd"):
        assert key in hp._t, f"构造函数应登记 key {key!r}"
        assert hp[key].item() == 0.0, (
            f"HyperParams.__init__ 不应赋值，但 {key!r} = {hp[key].item()}。"
            f"赋值只能通过 .set()，否则调用方会误以为 kwargs 生效了。"
        )

    # .set() 才是赋值入口
    hp.set(lr=0.25)
    assert abs(hp["lr"].item() - 0.25) < 1e-6


def test_hyperparams_values_are_0d_cpu_tensors():
    """0-D CPU tensor 是 torch.compile 不重编译的前提，别改成 0-D cuda 或 python 标量。"""
    HyperParams = _hyperparams_cls()

    hp = HyperParams(step=0, lr=0.0)
    hp.set(step=3, lr=0.1)
    for key in ("step", "lr"):
        t = hp[key]
        assert t.shape == torch.Size([]), f"{key} 必须是 0-D tensor，实际 {tuple(t.shape)}"
        assert t.device.type == "cpu", f"{key} 必须在 CPU（见 muon.py 的 0-D 技巧说明）"
        assert t.dtype == torch.float32, f"{key} 必须是 float32，实际 {t.dtype}"


# ===========================================================================
# 2. 数值型判据必须带 NaN 守卫
# ===========================================================================
def _test_functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            yield node


def test_boolean_reduction_tests_are_nan_safe():
    """用「比较结果的 .any()」当通过条件的判据，必须带 NaN 守卫。

    真实事故：test_adamw_step_actually_moves_params 断言 `(p != 1.0).any()`，
    而参数早已全变成 NaN —— `NaN != 1.0` 恒为真，于是这个测试**因为错误的
    原因而通过**，长期充当假护栏。

    这里只针对这一种确定的危险形状（布尔归约当判据），不做泛化的数值比较审查：
    后者噪音太大，会让人直接关掉这个检查。
    """
    suspicious = []
    for name in ("test_optim.py", "test_core.py", "test_data.py"):
        path = TESTS / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for fn in _test_functions(path):
            src = ast.get_source_segment(text, fn) or ""
            if "isnan" in src or "isfinite" in src:
                continue
            # 危险形状：assert ... (...  != ...).any()  /  (...).any() 当判据
            if re.search(r"assert[^\n]*\(\s*[^\n]*!=[^\n]*\)\s*\.\s*any\s*\(\s*\)", src):
                suspicious.append(f"{name}::{fn.name}")

    assert not suspicious, (
        "这些判据用「比较结果的 .any()」当通过条件，若变量已变成 NaN 会恒真"
        "（NaN != x 永远成立），请补一行 `assert not torch.isnan(x).any()`：\n  "
        + "\n  ".join(suspicious)
    )


# ===========================================================================
# 3. 章节判据不许被删空或改成永真
# ===========================================================================
def test_chapter_suites_still_have_enough_cases():
    """每个卷的判据文件不能被掏空 —— 判据变少 = 保护变弱，且不易察觉。"""
    expected = {
        "test_presets.py": 6,    # 卷0（只读，应全绿）
        "test_data.py": 19,      # 卷1
        "test_core.py": 18,      # 卷2-3
        "test_optim.py": 12,     # 卷4
    }
    for name, minimum in expected.items():
        path = TESTS / name
        assert path.exists(), f"判据文件不见了：{path}"
        n = sum(1 for _ in _test_functions(path))
        assert n >= minimum, (
            f"{name} 只剩 {n} 个用例（应 >= {minimum}）。"
            f"判据被删会让 progress.sh 失去意义。"
        )


def test_every_chapter_in_progress_sh_has_a_real_selector():
    """progress.sh 里每个章节的 -k 表达式，必须真的能选中至少一个用例。

    防止出现「判据表达式写错 -> 选中 0 个 -> 永远『未完成』或误判通过」。
    """
    script = (REPO / "script" / "progress.sh").read_text(encoding="utf-8")
    assert "-k '" in script, "progress.sh 看起来没有章节判据了"

    import re
    pairs = re.findall(r'\["([\d\-]+)"\]="([^"]+)"', script)
    assert len(pairs) >= 20, f"progress.sh 只解析到 {len(pairs)} 个章节（应 >= 20）"

    for chapter, expr in pairs:
        files = re.findall(r"(tests/test_\w+\.py)", expr)
        assert files, f"第 {chapter} 章的判据没写测试文件：{expr}"
        for f in files:
            assert (REPO / f).exists(), f"第 {chapter} 章引用了不存在的 {f}"


# ===========================================================================
# 4. 文档与实际状态不许脱节
# ===========================================================================
def test_tutorial_claims_match_reality():
    """教程里可证伪的数字声明，必须与仓库实际一致。"""
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    # scratch 脚本数量
    n_scratch = len(list((REPO / "scratch").glob("*.py")))
    m = re.search(r"scratch/\s+(\d+) 个实验脚本|scratch/\s+(\d+) 个可运行的实验脚本", readme)
    if m:
        claimed = int(m.group(1) or m.group(2))
        assert claimed == n_scratch, (
            f"README 说 scratch 有 {claimed} 个脚本，实际 {n_scratch} 个"
        )

    # 测试总数：只数「章节判据」，判据自检文件本身不计入
    n_tests = sum(
        sum(1 for _ in _test_functions(p))
        for p in TESTS.glob("test_*.py")
        if p.name != "test_judging_soundness.py"
    )
    m = re.search(r"tests/\s+(\d+) 个测试", readme)
    if m:
        assert int(m.group(1)) == n_tests, (
            f"README 说 {m.group(1)} 个测试，实际 {n_tests} 个"
            f"（章节判据，不含判据自检）"
        )


def _strip_py_comments(src: str) -> str:
    """去掉 Python 注释，保留代码。

    注释里写「别再硬编码 /home/xxx/Dev/...」是合理的说明，不该被判定为违规。
    """
    import io
    import tokenize
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type != tokenize.COMMENT:
                out.append(tok.string)
    except tokenize.TokenError:
        return src
    return "\n".join(out)


def test_no_hardcoded_foreign_home_paths():
    """禁止把作者机器上的绝对路径写进**代码**或文档。

    真实事故：scratch/audit_docs.py 里写死了作者的家目录，
    导致这个专门用来查文档问题的工具在别人机器上一跑就崩。

    注释与字符串示例不算 —— 那里出现路径通常是在说明问题本身。
    """
    pat = re.compile(r"/home/[a-z_][a-z0-9_-]*/(Dev|\.cache)/")
    offenders = []
    for p in sorted(REPO.glob("scratch/*.py")):
        for i, line in enumerate(_strip_py_comments(p.read_text(encoding="utf-8")).splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{p.relative_to(REPO)}:{i}")
    for p in sorted(REPO.glob("script/*.sh")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):     # shell 注释
                continue
            if pat.search(line):
                offenders.append(f"{p.relative_to(REPO)}:{i}")
    assert not offenders, (
        "这些代码里写死了别人的家目录路径，换机器就会崩：\n  " + "\n  ".join(offenders)
    )


def test_answer_holding_scripts_are_not_pre_solved():
    """手抄靶子脚本不许在 main 上是完整答案。

    这些脚本本身就是教程要你写的东西（toy_bpe / mini_bpe / packing_demo …），
    预先写好等于把答案直接发出去。至少要能看出「待实现」的痕迹。

    solution 分支是答案分支，完整实现是应该的，所以那里跳过。
    """
    if current_branch() == "solution":
        pytest.skip("solution 是答案分支，脚本本就该是完整实现")

    targets = {
        "mini_bpe.py": "03 章 手写一个玩具 BPE",
        "toy_bpe.py": "02 章 看 BPE 合并过程",
        "packing_demo.py": "05 章 best-fit 装箱",
        "naive_dataloader.py": "05 章 naive 基线",
    }
    # 注意：check_data.py 不在此列。它只有三条 assert（不变量检查），
    # 不含任何可抄的算法，而那些不变量在 dataloader.py 的 docstring 里已经写明，
    # 抽掉它只会删掉一个有用的验证工具，并不会收回任何答案。
    leaked = []
    for name, chapter in targets.items():
        path = REPO / "scratch" / name
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        if "NotImplementedError" not in src and "TODO" not in src and "待实现" not in src:
            leaked.append(f"scratch/{name}（{chapter}）")
    assert not leaked, (
        "这些是教程要你手抄的脚本，但 main 上是完整答案：\n  " + "\n  ".join(leaked)
        + "\n把它们也抽成骨架，或移到 solution 分支。"
    )
