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
        "test_presets.py": 28,         # 卷0（只读，应全绿）
        "test_data.py": 20,            # 卷1
        "test_core.py": 43,            # 卷2-3
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


# ===========================================================================
# 4b. 全仓文本不许有编码损坏（mojibake）
#
# ★ 真实事故（2026-10 审计发现）：两处 UTF-8 字符损坏已提交进仓库 ——
#     src/common/utils.py:150   「唯一的随机来源就是那<?>种子」
#     tests/test_presets.py:330 「模型存到了意外<?>目录」
#   两处都是**合法的 U+FFFD 编码**（`strict` 解码能通过），所以不是编码异常，
#   而是写入时就已损坏并提交了。引入者分别是 0cbee16 和 18823ef，
#   后者一路被 4 个 commit 带过而无人发现。
#
# 为什么之前没抓到：本文件里的编码检查绑在
# `test_no_hardcoded_foreign_home_paths` 的**路径模式**上（只扫 scratch/*.py
# 的 `/home/xxx/Dev/`），不是通用检查 —— src/、tests/、doc/ 都不在范围内。
#
# 价值高于它修的那两个字：把「字符损坏」从一次性清理变成持续守护。
# ===========================================================================
# 只扫「我们写的」文本。排除依赖、产物、缓存、以及本文件自己
# （本文件的 docstring 里就故意包含 U+FFFD 的字面说明）。
_TEXT_GLOBS = ("src/**/*.py", "tests/**/*.py", "script/*.sh", "scratch/*.py",
               "scratch/*.json", "doc/**/*.md", "*.md", "*.toml")
_TEXT_EXCLUDE_DIRS = {".venv", ".git", ".ruff_cache", ".pytest_cache",
                      "__pycache__", "runs", "node_modules"}
_TEXT_SELF = pathlib.Path(__file__).name


def _iter_repo_text_files():
    seen = set()
    for pattern in _TEXT_GLOBS:
        for p in sorted(REPO.glob(pattern)):
            if any(part in _TEXT_EXCLUDE_DIRS for part in p.parts):
                continue
            if p.name == _TEXT_SELF or not p.is_file():
                continue
            if p in seen:
                continue
            seen.add(p)
            yield p


def test_no_mojibake_anywhere_in_tracked_text():
    """全仓文本文件不得含 U+FFFD（替换字符）或非 UTF-8 编码。

    损坏的字节通常在写文件时就已变成 U+FFFD 并被提交，所以严格解码
    **不会报错** —— 必须显式扫这个码位。
    """
    offenders = []
    for p in _iter_repo_text_files():
        raw = p.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            offenders.append(f"{p.relative_to(REPO)}: 非法 UTF-8（{e.reason}）")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if "\ufffd" in line:
                # 报告整行，并把损坏位置标出来便于定位
                col = line.index("\ufffd") + 1
                n_bad = len(line) - len(line.replace("\ufffd", ""))
                excerpt = line.strip()[:60]
                offenders.append(
                    f"{p.relative_to(REPO)}:{lineno}: {n_bad} 个 U+FFFD "
                    f"(第 {col} 列起) | {excerpt}")
    assert not offenders, (
        "这些文件有编码损坏（U+FFFD 是「无法解码的字节」的替换字符）：\n  "
        + "\n  ".join(offenders)
        + "\n  损坏通常在写入时就发生并被提交，严格解码抓不到 —— "
          "只能显式扫这个码位。")


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


# ===========================================================================
# 消融显著性阈值必须是实测值，不能是拍脑袋的经验值
#
# ★ 2026-10 审计发现：纪律第 3 条「|Δ| < 0.02 时重复跑一次」里的 0.02
#   **从来没有被测量过** —— 它从项目最早的 commit (7319830) 就在了。
#
#   实测（3 次同配置 + 1 次换初始化，每组 39 分钟）：
#       同种子样本标准差 σ = 0.000134
#       σ_Δ = √2·σ         = 0.000190
#   0.02 比真实噪声大 **105 倍**（分母是 σ_Δ，见下面的 ratios_vs_old_threshold）。
#   后果：13 项消融里有 11 项被误判成「在噪声带内 / 测不出差异」，
#   而它们其实是 7σ~279σ 的真实效应。
#
#   这是本项目「不把具体取值当规律」那条纪律的一个实例：
#   **连判断显著性的阈值本身，都曾经是一个没测过的取值。**
# ===========================================================================
def _noise_floor_json():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parents[1]
                       / "scratch" / "ablation_noise_floor.json").read_text())


def test_ablation_threshold_is_the_measured_one_not_the_historical_guess():
    """
    ★ 阈值必须是实测的 0.001，不能回退成 0.02。

    0.02 这个数从最早的 commit 就在，是**经验值**。实测噪声只有它的 1/105
    （分母 σ_Δ = 0.000190，判定 |Δ| 用的分母）。
    """
    data = _noise_floor_json()
    derived = data["derived"]
    assert derived["threshold"] == 0.001, (
        f"显著性阈值变成了 {derived['threshold']} —— "
        "实测值是 0.001（σ_Δ = 0.000190，5.3σ）。"
        "如果确实要改，请同时更新 scratch/ablation.sh 和全部教程章节")
    # 保留旧值作为记录
    assert derived["old_unmeasured_threshold"] == 0.02, \
        "旧的未验证阈值应保留为 0.02 作为记录"


def test_ratio_claims_are_recomputable_from_their_denominators():
    """
    ★★ 「旧阈值比噪声大 N 倍」这个说法，N 必须能由分母**重算**出来。

    真实事故：0.02 ÷ 某个噪声值可以算出三个都合法的倍数 ——
        ÷ σ_Δ = 0.000190 → 105 倍  ← 判定 |Δ| 用这个（项目口径）
        ÷ σ   = 0.000134 → 149 倍  ← 单次跑本身的样本标准差
        ÷ 偏移 = 0.000437 →  46 倍  ← 换初始化实测到的最大系统性偏移

    而本仓库历史上同时出现过「46 倍」和「150 倍」，**都没写分母**，
    读者无从判断说的是哪个 —— 三者差了近 3 倍。

    所以这里不比对硬编码的字面量，而是**拿分母重算一遍**：
    改了任何一个分母、或手改了任何一个倍数，判据都会立刻报警。
    """
    d = _noise_floor_json()["derived"]
    r = d["ratios_vs_old_threshold"]
    old = d["old_unmeasured_threshold"]

    # σ_Δ 必须是 √2·σ（两个独立配置相减的合成噪声）
    assert d["sigma_delta"] == pytest.approx(d["same_seed_stdev"] * 2 ** 0.5, abs=1e-6), (
        f"σ_Δ={d['sigma_delta']} 不等于 √2·σ={d['same_seed_stdev'] * 2 ** 0.5:.6f}")

    for key, denom, what in [
        ("vs_sigma_delta", d["sigma_delta"], "判定 |Δ| 用的合成噪声"),
        ("vs_same_seed_stdev", d["same_seed_stdev"], "单次跑的样本标准差"),
        ("vs_init_shift", d["init_shift"], "换初始化的最大系统性偏移"),
    ]:
        assert key in r, f"ratios_vs_old_threshold 缺了 {key}"
        assert r[key] == pytest.approx(old / denom, abs=0.5), (
            f"{key} 记的是 {r[key]}，但 0.02 ÷ {denom:.6f}"
            f"（{what}）= {old / denom:.1f}。"
            "倍数必须与分母自洽 —— 请重算，不要手改字面量")

    # 阈值自述的 σ 倍数也要与 σ_Δ 自洽
    assert d["threshold_in_sigma"] == pytest.approx(d["threshold"] / d["sigma_delta"], abs=0.05), (
        f"threshold_in_sigma={d['threshold_in_sigma']}，"
        f"但 threshold÷σ_Δ={d['threshold'] / d['sigma_delta']:.2f}")

    # 权威口径必须是 σ_Δ，且文档里的头条数字要与之相符
    assert d["authoritative_ratio"] == "vs_sigma_delta", (
        "判定 |Δ| 的权威口径是 vs_sigma_delta（分母 σ_Δ）。"
        "若要改口径，请一并更新 README.md / doc/tutorial/README.md 的头条倍数")


def test_ratio_claims_in_docs_always_name_their_denominator():
    """
    ★ 文档里出现「N 倍」这个说法时，必须同时说明分母。

    「0.02 比噪声大 46 倍」单独出现是无意义的：读者无法判断分母是
    σ_Δ、σ 还是换初始化的偏移，而三者差了近 3 倍 —— 结论强度完全不同
    （105 倍是严格口径，46 倍是最保守口径）。
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    docs = [root / "README.md"] + sorted((root / "doc" / "tutorial").glob("*.md"))

    # 「N 倍」附近必须出现分母说明，或者这一行本来就在否定/更正它
    denominators = ("σ_Δ", "σ_Δ", "0.000190", "0.000134", "0.000437",
                    "分母", "sigma_delta")
    offenders = []
    for md in docs:
        for lineno, line in enumerate(md.read_text().splitlines(), 1):
            if not re.search(r"(46|105|149|150)\s*倍", line):
                continue
            if any(k in line for k in denominators):
                continue
            # 允许「历史上同时出现过 46 倍和 150 倍」这类元叙述
            if any(k in line for k in ("历史", "同时出现", "而不是", "而非")):
                continue
            offenders.append(f"{md.relative_to(root)}:{lineno}: {line.strip()[:88]}")
    assert not offenders, (
        "这些行报了倍数却没写分母（σ_Δ / σ / 偏移 三者差近 3 倍）：\n  "
        + "\n  ".join(offenders))


# ===========================================================================
# 4c. pytest 结果声明必须与单一事实源一致
#
# ★ 真实事故：三处「main 分支应该输出 69 failed, 116 passed, 7 skipped」
#   腐化了（实际是 71/125/8）却没人发现，两层原因：
#
#   (1) scratch/audit_docs.py 的正则只认「反引号 + N passed」一种形状，
#       骨架态声明的「N failed, M passed」结构上就匹配不到（已修，见该文件）。
#   (2) ★ 更根本：**骨架态的数字在 solution 分支上根本无法验证** ——
#       audit_docs.py 的 LEGAL 集合由「当前分支实测」构建，在答案分支上跑
#       永远验不到骨架态的数字。所以光修正则不够。
#
# 修法：把两个分支的预期数字登记进 scratch/pytest_expectations.json，
#       然后本判据要求**文档里所有声明必须与它逐字一致**。
#       这样在任何一个分支上都能抓到「某一处单独腐化」。
# ===========================================================================
def _pytest_expectations():
    import json as _json
    return _json.loads((pathlib.Path(__file__).resolve().parents[1]
                        / "scratch" / "pytest_expectations.json").read_text())


def test_skeleton_state_claims_are_consistent_across_docs():
    """文档里所有 pytest 结果声明，必须与 scratch/pytest_expectations.json 一致。

    单一事实源是那个 JSON；本判据保证文档不偏离它。
    ★ 注意它是**跨分支可用**的：不依赖当前分支实测，所以骨架态的
      数字在答案分支上同样受守护。
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    states = _pytest_expectations()["states"]
    docs = [root / "README.md"] + sorted((root / "doc" / "tutorial").glob("*.md"))

    # 骨架态声明：「N failed, M passed, K skipped」（failed 在前）
    skel_re = re.compile(r"(?<![\d.])(\d+)\s+failed,\s*(\d+)\s+passed"
                         r"(?:\s*,\s*(\d+)\s+skipped)?")
    # 答案态声明：「M passed, K skipped」
    ans_re = re.compile(r"(?<![\d.])(\d+)\s+passed(?:,\s*(\d+)\s+skipped)?")

    want_s = states["skeleton"]
    want_a = states["solution"]
    found_skel, found_ans = [], []

    for md in docs:
        text = md.read_text()
        # 先把骨架态的匹配挖掉，剩下的才算答案态候选
        skel_hits = list(skel_re.finditer(text))
        masked = text
        for m in reversed(skel_hits):
            masked = masked[:m.start()] + "\x00" * (m.end() - m.start()) + masked[m.end():]
        for m in skel_hits:
            got = (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
            want = (int(want_s["failed"]), int(want_s["passed"]), int(want_s["skipped"]))
            if got != want:
                line = text[:m.start()].count("\n") + 1
                found_skel.append(
                    f"{md.relative_to(root)}:{line}: 骨架态声明 {got[0]} failed, "
                    f"{got[1]} passed, {got[2]} skipped ≠ 登记值 "
                    f"{want[0]}/{want[1]}/{want[2]}")
        for m in ans_re.finditer(masked):
            cp, ck = int(m.group(1)), int(m.group(2) or 0)
            # 「全绿」的写法（不带 skipped）也常见，放行
            if (cp, ck) in ((int(want_a["passed"]), int(want_a["skipped"])),
                            (int(want_a["passed"]), 0)):
                continue
            line = text[:m.start()].count("\n") + 1
            found_ans.append(
                f"{md.relative_to(root)}:{line}: 答案态声明 {cp} passed"
                + (f", {ck} skipped" if ck else "")
                + f" ≠ 登记值 {want_a['passed']} passed, {want_a['skipped']} skipped")

    assert not found_skel and not found_ans, (
        "这些 pytest 结果声明与 scratch/pytest_expectations.json 不一致：\n  "
        + "\n  ".join(found_skel + found_ans)
        + "\n  ★ 单一事实源是那个 JSON。改判据数量后请在两个分支各重跑一次，"
          "然后同步 JSON 与 doc/、README.md。")


def test_noise_floor_numbers_match_the_actual_run_artifacts():
    """
    ★ `ablation_noise_floor.json` 里的每个数字必须与
       `runs/base_checkpoints/*/meta_*.json` 对得上。

    否则那份 JSON 会变成「看起来像实测的编造数据」——
    这正是本项目要消灭的东西。
    """
    import glob
    import json as _json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    data = _noise_floor_json()

    checked = 0
    for section in ("runs_same_seed", "run_different_seed"):
        for tag, recorded in data[section].items():
            if tag.startswith("_"):
                continue
            files = sorted(glob.glob(str(
                root / "runs" / "base_checkpoints" / tag / "meta_*.json")))
            assert files, (
                f"ablation_noise_floor.json 记了 {tag}，但 runs/ 下没有它的存档 —— "
                "要么跑批被删了，要么这个数字是编的")
            actual = _json.load(open(files[-1]))["best_val_bpb"]
            assert abs(actual - recorded) < 1e-12, (
                f"{tag} 的真实 val_bpb 是 {actual!r}，"
                f"而 ablation_noise_floor.json 记的是 {recorded!r}")
            checked += 1
    assert checked == 4, f"应核对 4 次跑批，实际核对了 {checked} 次"


def test_every_ablation_measurement_on_disk_is_listed_in_the_docs():
    """
    ★ 跑过的消融必须在文档里有一行，不能悄悄多跑一堆没人看的数据。

    反过来也成立：文档里出现的 tag 必须在磁盘上有对应存档。
    """
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    on_disk = {p.name for p in (root / "runs" / "base_checkpoints").iterdir()
               if p.is_dir()} if (root / "runs" / "base_checkpoints").is_dir() else set()
    doc = (root / "doc" / "tutorial" / "README.md").read_text()

    tags = [t for t in sorted(on_disk) if t.startswith("d6_")]
    assert tags, "磁盘上没有任何 d6_* 消融存档 —— 跑批结果被清掉了？"
    missing = [t for t in tags if t not in doc]
    assert not missing, (
        f"这些消融合跑过但文档里没有：{missing} —— "
        "实测数据必须出现在教程里，否则等于没跑")

    # 文档里提到的 d6_* tag 必须真有存档
    mentioned = set(re.findall(r"`(d6_[a-z0-9_]+)`", doc))
    phantom = sorted(t for t in mentioned if t not in on_disk)
    assert not phantom, f"文档里引用了磁盘上不存在的消融：{phantom}"


def test_ablation_sh_discipline_matches_the_measured_threshold():
    """
    `scratch/ablation.sh` 里印给用户看的纪律，必须是实测阈值 0.001。
    """
    from pathlib import Path
    sh = (Path(__file__).resolve().parents[1]
          / "scratch" / "ablation.sh").read_text()
    assert "|Δ| < 0.001" in sh, \
        "ablation.sh 的纪律第 3 条没有用实测阈值 0.001"
    assert "|Δ| < 0.02" not in sh, \
        "ablation.sh 还印着未验证的 0.02 阈值"


def test_tutorials_do_not_assert_the_unmeasured_threshold_as_fact():
    """
    ★ 教程里不得把 0.02 当成既成事实陈述。

    允许「早期文档写的是 0.02，那没测过」这类**更正说明**，
    但不允许「噪声带 ±0.02」这种断言式表述。
    """
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    # 这些章节整节都在「更正这句话」，允许出现
    excused = {
        "00-如何使用本教程.md", "06-bits-per-byte.md", "08-RMSNorm.md",
        "10-注意力三步曲.md", "12-QK-Norm与GQA.md", "README.md",
    }
    # ★ 逐行豁免：同一行里出现这些词，说明这一行是在**否定** 0.02，
    #   而不是断言它。缺了这个豁免，本判据会把自己写的更正文字也抓了
    #   —— 第一次就栽在这上面（ch13 的更正里写了「噪声 ±0.02」）。
    #    （刻意不含「实测」—— 那太宽，「实测噪声带 ±0.02」这种
    #     伪装成实测的断言也会被放过。已验证去掉它仍能通过。）
    disavow = ("拍的", "没测过", "未验证", "更正", "早期文档",
               "曾经", "而不是", "而非", "0.001")
    offenders = []
    for md in sorted((root / "doc" / "tutorial").glob("*.md")):
        if md.name in excused or md.name == "README.md":
            continue
        for lineno, line in enumerate(md.read_text().splitlines(), 1):
            if "0.02" not in line:
                continue
            if any(k in line for k in disavow):
                continue
            # 断言式表述：噪声带 / 落在噪声内 …… 0.02
            if re.search(r"噪声(带)?[^。]{0,12}0\.02|0\.02[^。]{0,12}噪声", line):
                offenders.append(f"{md.name}:{lineno}: {line.strip()[:90]}")
    assert not offenders, (
        "这些行把未验证的 0.02 当成事实：\n  " + "\n  ".join(offenders) +
        "\n  真实阈值是实测的 0.001（σ_Δ = 0.000190）。")


def test_interaction_check_is_stored_and_mathematically_consistent():
    """
    ★★ 「单变量边际效应 ≠ 从组合移除的效应」这条结论必须自洽。

    2026-10 实测发现：
        resid_lambdas 单独测        Δ = +0.0061  （32σ，显著有害）
        从 5-trick 组合里移除它    Δ = -0.0008  （4σ，方向相反且小 7.6 倍）

    这两个数**都在磁盘上**（runs/base_checkpoints/d6_resid 和
    d6_alltricks / d6_fulltricks）。本条验证：
      1. 存储的 interaction_check 数字与磁盘一致
      2. ratio 算得对
      3. ★ 文档里必须同时出现这两个数 ——
         只写「resid 有害 +0.0061」而不写「实际只赚 0.0008」
         就是**选择性引用**，会让读者以为删掉 resid 能赚 0.0061
    """
    import glob
    import json as _json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    data = _noise_floor_json()
    ic = data["derived"]["interaction_check"]

    def bpb(tag):
        f = sorted(glob.glob(str(root / "runs" / "base_checkpoints" / tag
                                / "meta_*.json")))
        assert f, f"{tag} 没有跑批存档"
        return _json.load(open(f[-1]))["best_val_bpb"]

    import statistics as st
    same = [bpb(t) for t in ("d6_base", "base_r1", "base_r2")]
    base = st.mean(same)

    resid_alone = bpb("d6_resid") - base
    removal = bpb("d6_fulltricks") - bpb("d6_alltricks")

    assert abs(ic["resid_alone_delta"] - resid_alone) < 1e-9, \
        f"interaction_check.resid_alone_delta={ic['resid_alone_delta']}，" \
        f"实算是 {resid_alone}"
    assert abs(ic["removing_resid_from_all5_delta"] - removal) < 1e-9, \
        f"interaction_check 的移除值={ic['removing_resid_from_all5_delta']}，" \
        f"实算是 {removal}"
    assert abs(ic["ratio"] - abs(resid_alone / removal)) < 0.2, \
        f"ratio={ic['ratio']}，按实算应为 {abs(resid_alone / removal):.1f}"

    # ★ 方向必须相反：单独测有害，从组合移除才是收益
    assert resid_alone > 0 > removal, (
        f"方向反了：单独测 resid_alone={resid_alone:+.4f}，"
        f"移除 removal={removal:+.4f} —— "
        "交互效应的前提是这两个符号相反")

    # 文档必须同时出现两个数
    for md in ("README.md", "doc/tutorial/README.md",
               "doc/tutorial/17-resid-lambdas与x0-lambdas.md"):
        text = (root / md).read_text()
        assert "0.0061" in text, f"{md} 没提 resid 单独测的 +0.0061"
        assert "0.0008" in text, (
            f"{md} 提到了 resid 单独测 +0.0061，却没提"
            "「从组合移除实际只赚 0.0008」—— "
            "这是选择性引用，读者会以为删掉 resid 能赚 0.0061")


def test_full_preset_no_longer_claims_all_five_tricks():
    """
    `full` 档现在是 4 个 trick。任何仍说它「trick 全开」的地方都是错的。
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for md in ("README.md", "doc/tutorial/README.md"):
        text = (root / md).read_text()
        bad = [ln.strip() for ln in text.splitlines()
               if "trick 全开" in ln or "全开 + Muon" in ln]
        # 允许出现在明确限定「那是旧的 5-trick 配置」的语境里
        bad = [ln for ln in bad
               if not any(k in ln for k in ("5 个", "旧", "曾经", "resid",
                                            "device_batch_size", "相比"))]
        assert not bad, (
            f"{md} 仍在把 full 档描述成「trick 全开」，但 resid_lambdas "
            f"已被去掉：\n  " + "\n  ".join(bad))
