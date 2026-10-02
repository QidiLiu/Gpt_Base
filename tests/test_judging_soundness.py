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


def is_skeleton_checkout() -> bool:
    """当前检出的是不是「骨架态」。

    刻意**不按分支名判断**：dev 被重置成与 solution 完全一致之后，
    「branch == 'solution' 才 skip」的写法在 dev 上就会误报
    （实测会在 dev 上挂 2 个）。判据改成看代码状态 ——
    骨架里必然有 NotImplementedError，答案分支里必然没有。
    """
    marker = REPO / "src/data/tokenizer.py"
    if not marker.exists():
        return False
    return "NotImplementedError" in marker.read_text(encoding="utf-8")


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
def _collected(path):
    """pytest 实际 collect 的用例数（参数化会让 1 个函数收集出多个）。"""
    r = subprocess.run([sys.executable, "-m", "pytest", str(path),
                        "--collect-only", "-q"],
                       cwd=REPO, capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    return int(m.group(1)) if m else -1


def test_chapter_suites_still_have_enough_cases():
    """每个卷的判据文件不能被掏空 —— 判据变少 = 保护变弱，且不易察觉。

    这里用 `>=` 而不是 `==`，是为了让「新增一个合法测试」不需要改这个文件。
    诚实的说明：当前 floor 就等于实际数量，所以
      · 「删除测试」-> 数量跌破 floor -> **能抓到**（与 == 等价）
      · 「新增测试」-> 放行（这是 == 做不到的，也正是改用 >= 的唯一理由）
    精确数字的核对不在这里，而在 `scratch/audit_docs.py`：它比对
    CHAPTER_TESTS 字典并在不符时退出非零。两者分工：
      本测试 = 快速、防灾难性掏空；audit_docs = 精确、需手动/CI 跑。
    """
    minimum = {
        "test_presets.py": 21,         # 卷0（只读，应全绿）
        "test_data.py": 20,            # 卷1
        "test_core.py": 38,            # 卷2-3
        "test_optim.py": 16,           # 卷4
        "test_metrics.py": 11,         # 卷7 只读代码护栏
        "test_checkpoint.py": 15,      # 卷5 只读代码护栏
        "test_dataloader_resume.py": 10,  # 卷1 第05章（精确续训）
        "test_engine.py": 53,          # 卷7 推理引擎（只读代码护栏）
        "test_tasks.py": 27,           # 卷7 任务与题库（只读代码护栏）
    }
    for name, floor in minimum.items():
        path = TESTS / name
        assert path.exists(), f"判据文件不见了：{path}"
        n = _collected(path)
        assert n >= floor, (
            f"{name} 只收集到 {n} 个用例，下限是 {floor} —— "
            f"判据被掏空了？精确数字请跑 scratch/audit_docs.py 核对。"
        )


def test_every_chapter_in_progress_sh_has_a_real_selector():
    """progress.sh 里每个章节的 -k 表达式，必须真的能选中至少一个用例。

    防止出现「判据表达式写错 -> 选中 0 个 -> 永远『未完成』或误判通过」。

    ★ 与旧版的区别：旧版只检查「引用的测试文件存在」，docstring 声称的
      「真的能选中至少一个用例」**从未实现** —— 表达式写成
      `-k '不存在的关键字'` 照样通过。现在真的跑一次 `pytest --collect-only -k`，
      选中 0 个就报错。
    """
    script = (REPO / "script" / "progress.sh").read_text(encoding="utf-8")
    assert "-k '" in script, "progress.sh 看起来没有章节判据了"

    import re
    import subprocess
    import sys
    pairs = re.findall(r'\["([\d\-]+)"\]="([^"]+)"', script)
    assert len(pairs) >= 20, f"progress.sh 只解析到 {len(pairs)} 个章节（应 >= 20）"

    empty = []
    for chapter, expr in pairs:
        files = re.findall(r"(tests/test_\w+\.py)", expr)
        assert files, f"第 {chapter} 章的判据没写测试文件：{expr}"
        for f in files:
            assert (REPO / f).exists(), f"第 {chapter} 章引用了不存在的 {f}"
        # ★ 只把 -k 后面**引号里那个表达式**传给 pytest。
        #   传整个 expr（"tests/test_core.py -k 'rmsnorm'"）是错的：
        #   pytest 会收到一个语法错误的 -k 表达式，然后**照样收集全部用例**，
        #   于是「选中 0 个」这个检查永远不会触发 —— 判据形同虚设。
        m_kw = re.search(r"-k\s+'([^']+)'", expr)
        assert m_kw, f"第 {chapter} 章的判据里找不到 -k '...'：{expr}"
        r = subprocess.run(
            [sys.executable, "-m", "pytest", *files, "-k", m_kw.group(1),
             "--collect-only", "-q", "-p", "no:cacheprovider"],
            cwd=REPO, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            # -k 表达式语法错误 / 文件不存在 —— 判据本身是坏的
            empty.append((chapter, f"{expr}  → pytest 退出码 {r.returncode}: "
                                    f"{r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}"))
            continue
        m_n = re.search(r"(\d+) tests? collected", r.stdout)
        if m_n is None or int(m_n.group(1)) == 0:
            empty.append((chapter, expr))
    assert not empty, (
        "这些章节的 -k 表达式选中 0 个用例（拼错了关键字？）：\n  "
        + "\n  ".join(f"第 {c} 章: {e}" for c, e in empty))


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

    # 测试总数：只数「章节判据」，判据自检文件本身不计入。
    # 用 pytest 实际 collect 的数（参数化会让 1 个函数收集出多个用例），
    # 与 audit_docs.py、文档里的口径保持一致。
    n_tests = sum(
        _collected(p) for p in sorted(TESTS.glob("test_*.py"))
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


def test_entry_scripts_have_a_working_default_mode():
    """入口脚本必须真的支持「不传参数」——它们的注释都写着「默认 smoke」。

    真实事故：5 个脚本都写 `MODE="$1"; shift || true`，而 _common.sh 开了
    `set -euo pipefail`，无参数调用直接 `unbound variable` 退出。
    _common.sh 里的 `MODE="${1:-smoke}"` 默认值被空串覆盖掉了。
    """
    scripts = sorted(REPO.glob("script/*.sh"))
    entries = [p for p in scripts if p.name != "_common.sh"]
    assert len(entries) >= 5, f"只找到 {len(entries)} 个入口脚本"

    problems = []
    for p in entries:
        lines = p.read_text(encoding="utf-8").splitlines()
        mode_lines = [(i, l) for i, l in enumerate(lines, 1)
                      if re.match(r'\s*MODE=', l)]
        if not mode_lines:
            continue                      # 不接管 MODE 的脚本跳过
        for i, line in mode_lines:
            if "${1:-" in line:
                continue                  # 有默认值，OK
            if line.startswith("MODE=\"$1\""):
                problems.append(
                    f"{p.name}:{i} 用了裸 $1，无参数调用会 unbound variable；"
                    f'应写成 MODE="${{1:-smoke}}"')
            elif "MODE=" in line and "$1" in line:
                problems.append(
                    f"{p.name}:{i} 读 $1 但没有默认值：{line.strip()!r}")

    assert not problems, (
        "入口脚本的「默认档位」是坏的（每个脚本的注释都承诺支持无参数调用）：\n  "
        + "\n  ".join(problems)
    )


def test_answer_holding_scripts_are_not_pre_solved():
    """手抄靶子脚本不许在 main 上是完整答案。

    这些脚本本身就是教程要你写的东西（toy_bpe / mini_bpe / packing_demo …），
    预先写好等于把答案直接发出去。至少要能看出「待实现」的痕迹。

    solution 分支是答案分支，完整实现是应该的，所以那里跳过。
    """
    if not is_skeleton_checkout():
        pytest.skip("当前不是骨架态（答案分支），脚本本就该是完整实现")

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
        "这些是教程要你手抄的脚本，但骨架态检出里是完整答案：\n  " + "\n  ".join(leaked)
        + "\n把它们也抽成骨架，或只放到答案分支。"
    )


# ===========================================================================
# 5. scratch 门禁表不许与实际行为脱节
# ===========================================================================
def test_scratch_gating_table_matches_reality():
    """README 里 scratch 脚本的 🔒/✅/✍ 标注，必须与实跑结果一致。

    真实事故：`mini_bpe.py` 被标成 🔒「写完 list_parquet_files 一行」，
    但实测在 main 上直接 exit=0 跑通 —— 门禁是假的。
    这类声明可证伪，所以就该用可证伪的方式守住。

    注意：门禁表只存在于**学习者视角**的文档里；答案分支的文档是
    「读者视角」，压根没有这张表。所以这里先判表在不在，而不是按分支名跳。
    """
    if not is_skeleton_checkout():
        pytest.skip("当前不是骨架态，答案分支的文档没有门禁表")

    readme_path = REPO / "doc/tutorial/README.md"
    readme = readme_path.read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*`scratch/(\w+\.py)`\s*\|([^|]*)\|([^|]*)\|", readme, re.M)
    if len(rows) < 9:
        # 读者视角的文档本来就没有这张表（它们假设代码已经写好）
        if "现在就能跑" not in readme:
            pytest.skip("当前文档是读者视角，没有 scratch 门禁表")
        raise AssertionError(
            f"门禁表只解析到 {len(rows)} 行（应 >= 9）。"
            f"骨架态的学习者文档必须带这张表。")

    import os
    env = dict(os.environ, PYTHONPATH=str(SRC), OMP_NUM_THREADS="1")
    mismatches = []

    for name, _dep, status in rows:
        status = status.strip()
        path = REPO / "scratch" / name
        if not path.exists():
            mismatches.append(f"{name}: 表里有，但文件不存在")
            continue

        try:
            r = subprocess.run([sys.executable, str(path)], cwd=REPO, env=env,
                               capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            mismatches.append(f"{name}: 跑超时，状态无法判定")
            continue

        blocked = "NotImplementedError" in (r.stdout + r.stderr)
        runs = r.returncode == 0

        if "✅" in status and not runs:
            mismatches.append(f"{name}: 表里标 ✅ 能跑，实际跑不了")
        elif ("🔒" in status or "✍" in status) and runs and not blocked:
            mismatches.append(
                f"{name}: 表里标 {status[:2]}（应被挡住），实际直接跑通了 —— 门禁是假的")

    assert not mismatches, (
        "scratch 门禁表与实际行为不符：\n  " + "\n  ".join(mismatches)
        + "\n要么改代码，要么改 README 的状态标注。"
    )
