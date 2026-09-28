"""
文档一致性审计。

检查教程/文档里所有「可验证的断言」是否与实际一致。
这个脚本本身也是给维护者用的：改了代码就跑一遍。
"""
import pathlib
import re
import subprocess
import sys

# 以本文件位置推导仓库根目录 —— 之前这里写死了作者机器上的绝对路径
# （/home/ben/Dev/Gpt_Base），导致这个专门用来查文档问题的工具在别人
# 机器上一跑就 FileNotFoundError。
ROOT = pathlib.Path(__file__).resolve().parent.parent
ISSUES = []


def note(ok, where, msg):
    if not ok:
        ISSUES.append(f"{where}: {msg}")


# ── 1. 测试数量 ──────────────────────────────────────────────
# test_judging_soundness.py 是判据自检（不测手抄目标），不计入章节判据。
CHAPTER_TESTS = {"test_core.py": 18, "test_data.py": 19,
                 "test_presets.py": 6, "test_optim.py": 12,
                 "test_metrics.py": 11}
SOUNDNESS = "test_judging_soundness.py"

counts = {}
for f in (ROOT / "tests").glob("test_*.py"):
    counts[f.name] = len(re.findall(r"^def test_", f.read_text(), re.M))

chapter_counts = {k: v for k, v in counts.items() if k in CHAPTER_TESTS}
total = sum(chapter_counts.values())
note(chapter_counts == CHAPTER_TESTS,
     "tests/", f"实际章节判据数 {chapter_counts}（应 {CHAPTER_TESTS}，合计 {total}）")
n_soundness = counts.get(SOUNDNESS, 0)
note(n_soundness > 0, "tests/", f"缺少 {SOUNDNESS}（判据自检，防回归的关键闸门）")

docs = list((ROOT / "doc").rglob("*.md")) + [ROOT / "README.md"]
for p in docs:
    s = p.read_text()
    for m in re.finditer(r"(\d+)\s*个测试", s):
        if int(m.group(1)) != total:
            note(False, str(p.relative_to(ROOT)),
                 f"写了「{m.group(1)} 个测试」，实际 {total}（章节判据，不含判据自检）")

# ── 2. pytest 结果声明 ────────────────────────────────────────
# pytest 的摘要在有失败时是 "44 failed, 7 passed in 1.29s"（failed 在前），
# 全通过时是 "51 passed in 1.2s"。两种顺序都要认。
chapter_files = [f"tests/{k}" for k in CHAPTER_TESTS]

def _run_pytest():
    """跑章节判据。优先 uv run，退回当前解释器；都不可用就跳过这项检查。"""
    for cmd in (["uv", "run", "pytest", *chapter_files, "-q"],
                [sys.executable, "-m", "pytest", *chapter_files, "-q"]):
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if r.returncode in (0, 1):      # 0=全绿 1=有失败，都是有效结果
            return r.stdout
    return None

_out = _run_pytest()
if _out is None:
    print("[warn] 跑不了 pytest，跳过「pytest 结果声明」这项检查\n")
else:
    tail = [l for l in _out.strip().split("\n") if "passed" in l or "failed" in l]
    actual = tail[-1] if tail else "?"
    np_ = sum(int(x) for x in re.findall(r"(\d+) passed", actual))
    nf_ = sum(int(x) for x in re.findall(r"(\d+) failed", actual))
    # 合法状态：main 骨架态（少数 passed + 多数 failed）；
    #          solution 完整实现（全通过）；solution 上只有 test_core 的旧状态。
    LEGAL = {(np_, nf_), (18, 0), (total, 0)}
    for p in docs:
        s = p.read_text()
        for m in re.finditer(r"`(\d+) passed(?:, (\d+) failed)?`", s):
            cp, cf = int(m.group(1)), int(m.group(2) or 0)
            if (cp, cf) not in LEGAL:
                note(False, str(p.relative_to(ROOT)),
                     f"写了 `{cp} passed{', ' + str(cf) + ' failed' if cf else ''}`，"
                     f"main 实际 `{' '.join(actual.split(' in ')[0].split())}`"
                     f"（solution 分支是 {total} passed）")

# ── 3. 教程里引用的文件是否存在 ───────────────────────────────
for p in docs:
    for m in re.finditer(r"`(scratch/[\w./]+\.py)`", p.read_text()):
        if not (ROOT / m.group(1)).exists():
            note(False, str(p.relative_to(ROOT)),
                 f"引用了不存在的文件 {m.group(1)}")
    for m in re.finditer(r"`(script/[\w./]+\.sh)`", p.read_text()):
        if not (ROOT / m.group(1)).exists():
            note(False, str(p.relative_to(ROOT)),
                 f"引用了不存在的脚本 {m.group(1)}")
    for m in re.finditer(r"`(tests/[\w./]+\.py)`", p.read_text()):
        if not (ROOT / m.group(1)).exists():
            note(False, str(p.relative_to(ROOT)),
                 f"引用了不存在的测试文件 {m.group(1)}")

# ── 4. 教程里引用的测试函数是否真实存在 ──────────────────────
all_tests = set()
for f in (ROOT / "tests").glob("test_*.py"):
    all_tests |= set(re.findall(r"^def (test_\w+)", f.read_text(), re.M))
for p in docs:
    for m in re.finditer(r"`(test_\w+)`", p.read_text()):
        if m.group(1) not in all_tests:
            note(False, str(p.relative_to(ROOT)),
                 f"提到了不存在的测试函数 {m.group(1)}")

# ── 5. 教程内部链接 ──────────────────────────────────────────
# 待写章节在目录里带 🚧 标记，其链接指向尚未创建的文件 —— 合法
PENDING = re.compile(r"🚧\s*\[\d\d\]\(")
for p in docs:
    txt = p.read_text()
    pending = set(PENDING.findall(txt))          # 已被标记为待写
    for m in re.finditer(r"(\[\d\d\]|\]\()?\(?(\d\d-[^)]+\.md)\)", txt):
        pass
    for m in re.finditer(r"\]\((\d\d-[^)]+\.md)\)", txt):
        f = m.group(1)
        if (p.parent / f).exists():
            continue
        # 往前看 10 个字符有没有 🚧
        if m.start() >= 0 and "🚧" in txt[max(0, m.start() - 8):m.start()]:
            continue
        note(False, str(p.relative_to(ROOT)), f"断链 -> {f}")

# ── 6. 文档里提到的章节文件是否存在 ──────────────────────────
index = ROOT / "doc/tutorial/README.md"
_itxt = index.read_text()
for m in re.finditer(r"\]\((\d\d-[^)]+\.md)\)", _itxt):
    f = m.group(1)
    if (index.parent / f).exists():
        continue
    if "🚧" in _itxt[max(0, m.start() - 8):m.start()]:
        continue    # 已标记为待写
    note(False, "doc/tutorial/README.md", f"目录里列了但文件不存在且未标 🚧: {f}")

# ── 报告 ─────────────────────────────────────────────────────
print("测试文件分布:", counts, "合计", total)
print("main 分支 pytest:", actual)
print()
if ISSUES:
    print(f"发现 {len(ISSUES)} 个问题：")
    for i in ISSUES:
        print("  ✗", i)
    sys.exit(1)
else:
    print("✓ 文档与代码一致，未发现问题")
