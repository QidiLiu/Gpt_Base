"""
文档一致性审计。

检查教程/文档里所有「可验证的断言」是否与实际一致。
这个脚本本身也是给维护者用的：改了代码就跑一遍。
"""
import ast
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
#
# 注意：用 pytest 实际 collect 的数量，而不是 `^def test_` 的行数 ——
# 参数化（@pytest.mark.parametrize）会让 1 个函数收集出多个用例，
# 两者对不上会让文档里的数字失真。
CHAPTER_TESTS = {"test_core.py": 43, "test_data.py": 21,
                 "test_presets.py": 24, "test_optim.py": 16,
                 "test_metrics.py": 11, "test_checkpoint.py": 15,
                 "test_dataloader_resume.py": 10,
                 "test_engine.py": 53, "test_tasks.py": 27}
SOUNDNESS = "test_judging_soundness.py"


def _collected(paths):
    """用 pytest 实际 collect 出来的用例数。跑不起来就退回数函数。"""
    r = subprocess.run([sys.executable, "-m", "pytest", *paths,
                        "--collect-only", "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    if m:
        return int(m.group(1))
    total = 0
    for p in paths:
        total += len(re.findall(r"^def test_", pathlib.Path(p).read_text(), re.M))
    return total


counts = {}
for f in sorted((ROOT / "tests").glob("test_*.py")):
    counts[f.name] = _collected([str(f)])

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
# pytest 的摘要在有失败时是 "58 failed, 111 passed, 7 skipped in 1.29s"
# （failed 在前），全通过时是 "176 passed, 2 skipped in 12.4s"。
# 两种顺序都要认。
chapter_files = [f"tests/{k}" for k in CHAPTER_TESTS]

def _run(files):
    """跑一组测试文件，返回摘要行。优先 uv run，退回当前解释器。"""
    for cmd in (["uv", "run", "pytest", *files, "-q"],
                [sys.executable, "-m", "pytest", *files, "-q"]):
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if r.returncode in (0, 1):      # 0=全绿 1=有失败，都是有效结果
            tail = [l for l in r.stdout.strip().split("\n")
                    if "passed" in l or "failed" in l]
            return tail[-1] if tail else None
    return None


_out = _run(chapter_files)
_full = _run(["tests"])            # 全量：多出判据自检，数字与章节口径不同
if _out is None:
    print("[warn] 跑不了 pytest，跳过「pytest 结果声明」这项检查\n")
else:
    actual = _out
    branch = subprocess.run(["git", "-C", str(ROOT), "rev-parse",
                             "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "?"

    def _counts(summary):
        return (sum(int(x) for x in re.findall(r"(\d+) passed", summary)),
                sum(int(x) for x in re.findall(r"(\d+) failed", summary)))

    np_, nf_ = _counts(actual)
    # 合法状态有两套口径，文档里两种写法都算对：
    #   章节口径（只跑 9 个判据文件）：骨架态有 failed，答案态全通过
    #   全量口径（pytest tests/）：额外包含 test_judging_soundness 的 10 个，
    #                            它们在骨架态/答案态之间会自动 skip
    LEGAL = {(np_, nf_), (total, 0)}
    if _full:
        LEGAL.add(_counts(_full))
    for p in docs:
        s = p.read_text()
        for m in re.finditer(r"`(\d+) passed(?:, (\d+) failed)?`", s):
            cp, cf = int(m.group(1)), int(m.group(2) or 0)
            if (cp, cf) not in LEGAL:
                other = "solution" if branch == "main" else "main"
                note(False, str(p.relative_to(ROOT)),
                     f"写了 `{cp} passed{', ' + str(cf) + ' failed' if cf else ''}`，"
                     f"当前分支（{branch}）实际 "
                     f"`{' '.join(actual.split(' in ')[0].split())}`"
                     f"（章节口径 {np_} passed / {nf_} failed，"
                     f"全量口径 {_full or '跑不了'}）"
                     f"（{other} 分支是 {total} passed）")

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

# ── 5b. 教程代码块 vs 仓库文件 ───────────────────────────────
# 教程里的代码块是给人**照抄**的。一旦它和仓库里真正能跑的文件漂移，
# 读者抄下来就是错的 —— 而且往往不报错，只是数字不对。
#
# 为什么用「登记式」而不是「全文逐字比对」？
#   教程里有很多块是**故意不完整**的：类的方法续写（靠缩进挂在上面的类里）、
#   夹在散文里的说明性片段、手抄版本（比 src/ 的成品更短）。
#   全文比对会产生几十条误报，不如只检查少数几个「必须一致」的对照。
#
# 下面每一对表示：md 里标了 ✍️/📦 的那些块，应该和仓库文件讲同一件事。
# 只检查能安全检查的维度（符号名、死代码），不做逐字 diff —— 逐字 diff
# 一旦有人为了教学可读性调整注释就会误报，得不偿失。
#
# 为什么不按「文件」对照，而要精确到**代码块**？
#   一章里往往有好几段互不相干的代码：BPETokenizer 的类定义、toy_bpe 的
#   完整脚本、minibpe 的实现、mask 演示的三个验证…它们各自对应不同的文件。
#   拿整章去对一个文件，必然误报。
#   所以下面每一条只匹配**一个特定代码块**：靠「块内必须出现的内容」定位，
#   命中之后才拿它去和仓库文件比。
CODE_BLOCK_CHECKS = [
    # (教程文件, 仓库文件, 块内必须出现的字符串, 说明)
    ("doc/tutorial/02-为什么需要分词器.md", "scratch/vocab_math.py",
     "build_model_config", "验证 3 的 vocab_math 脚本"),
    ("doc/tutorial/02-为什么需要分词器.md", "scratch/toy_bpe.py",
     "def toy_bpe", "验证 2 的 toy_bpe"),
    ("doc/tutorial/03-手写一个玩具BPE.md", "scratch/mini_bpe.py",
     "class MiniBPE", "手抄代码的 MiniBPE"),
    ("doc/tutorial/04-对话模板与特殊token.md", "scratch/mask_demo.py",
     "tok = get_tokenizer()", "验证 1 的 mask 演示"),
    ("doc/tutorial/01-环境与全景图.md", "scratch/check_data.py",
     "make_dataloader", "验证 2 的数据不变量检查"),
    ("doc/tutorial/01-环境与全景图.md", "scratch/hardware.py",
     "get_device_capability", "硬件参数表 + SDPA 布局陷阱"),
]


def _py_fences(txt):
    """抽出 markdown 里所有 ```python 代码块。"""
    out, cur = [], None
    for line in txt.split("\n"):
        s = line.strip()
        if cur is None and s.startswith("```python"):
            cur = []
        elif cur is not None and s == "```":
            out.append("\n".join(cur))
            cur = None
        elif cur is not None:
            cur.append(line)
    return out


def _parse(block):
    """能 parse 就返回 AST，不能就返回 None（教程里故意不完整的块）。"""
    try:
        return ast.parse(block)
    except SyntaxError:
        return None


def _symbols(tree):
    """AST 里出现的所有 def/class 名。"""
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def _called_names(tree, block):
    """
    块里「调用了但本块没定义、也没 import」的函数名。

    这类名字必须来自仓库文件 —— 函数改了名而文档没跟，就是最典型的漂移。
    排除内置和标准库，避免把 print/len 之类也当成待核对项。
    """
    import builtins
    defined, imported = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                imported.add((a.asname or a.name).split(".")[0])
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    skip = defined | imported | set(dir(builtins)) | {"self", "cls", "__name__"}
    return {c for c in called if c not in skip}


def _assigned_names(stmt):
    """一条语句里被赋值的变量名（只看最外层的 = / += / 注解赋值）。"""
    out = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(node.target, ast.Name):
                out.add(node.target.id)
    return out


def _is_read(stmt, name):
    """这条语句是否读取了 name（右侧出现即为读取）。"""
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name) and node.id == name:
            if not isinstance(node.ctx, ast.Store):
                return True
        # f-string 里的 {name} 也算读取
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.FormattedValue) and \
                        isinstance(v.value, ast.Name) and v.value.id == name:
                    return True
    return False


def _useless_stores(tree, block):
    """
    找出「白算了」的赋值 —— 本教程真实踩过的坑。

    两类：
      1) 覆盖型：X = A 之后，下一条语句就是 X = B，中间没人读 X。
         （02 章的 `total = ...` 写了两遍，第一遍完全无效）
      2) 死存型：X = A 之后，本块内再也没读过 X。
         （`make_run_config` 这种 import 了没用上的情况靠它兜不住，
           那个交给未定义名/未使用名检查）

    保守做法：只看**同一个语句列表**里相邻的两条，
    跨作用域（if/for/try 内部）一律不管 —— 宁可漏报不要误报。
    """
    lines = block.split("\n")

    def _loc(lineno):
        return lines[lineno - 1].strip() if 0 < lineno <= len(lines) else "?"

    findings = []
    # 覆盖型：相邻两条语句赋同一个名字，前一条的结果没被用到
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for a, b in zip(body, body[1:]):
            if not isinstance(a, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                continue
            names = _assigned_names(a) & _assigned_names(b)
            for name in names:
                if not _is_read(b, name):
                    findings.append(("覆盖", name, a.lineno, _loc(a.lineno)))

    # 死存型：整个块里从没被读过的赋值
    stored, read = {}, set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Store):
                stored.setdefault(n.id, []).append(n.lineno)
            else:
                read.add(n.id)
    for name, lns in stored.items():
        if name in read or name.startswith("_") or name.isupper():
            continue
        # 排除循环变量等常见合法用法：只报顶层语句位置
        top = {s.lineno for s in getattr(tree, "body", [])}
        for ln in lns:
            if ln in top:
                findings.append(("未使用", name, ln, _loc(ln)))
    return findings


for md_rel, py_rel, anchor, what in CODE_BLOCK_CHECKS:
    md_path, py_path = ROOT / md_rel, ROOT / py_rel
    if not (md_path.exists() and py_path.exists()):
        note(False, "audit_docs.py", f"代码块对照表里的文件不存在: {md_rel} / {py_rel}")
        continue

    # 只取包含 anchor 的块 —— 一章里的其它代码块不属于这个对照
    blocks = [b for b in _py_fences(md_path.read_text())
              if anchor in b and _parse(b)]
    if not blocks:
        note(False, md_rel,
             f"{what}：md 里找不到含 `{anchor}` 的可解析 ```python 块"
             f"（教程结构改了？对照表需要更新）")
        continue

    repo_tree = ast.parse(py_path.read_text())
    repo_syms = _symbols(repo_tree)

    for b in blocks:
        t = _parse(b)
        # (a) 块里的 def/class 必须都在仓库文件里存在 —— 函数改名了文档没跟
        missing = _symbols(t) - repo_syms
        if missing:
            note(False, md_rel,
                 f"{what}：块里的 {sorted(missing)} 在 {py_rel} 里不存在（文档与代码漂移）")

        # (a2) 块里调用的「本项目函数」也必须在仓库文件里有同名 def/class
        unknown = _called_names(t, b) - repo_syms
        if unknown:
            note(False, md_rel,
                 f"{what}：块里调用了 {sorted(unknown)}，"
                 f"但 {py_rel} 里没有同名函数（文档与代码漂移）")

        # (b) 白算了的赋值 —— 会让读者照抄一段永远不生效的代码
        for kind, name, ln, src in _useless_stores(_parse(b), b):
            if src.startswith(("for ", "import ", "from ", "@")):
                continue
            note(False, md_rel,
                 f"{what}：md 代码块第 {ln} 行 `{src}` 是无效赋值"
                 f"（{kind}：{name} 的这个值没被用到）")

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
_branch = subprocess.run(["git", "-C", str(ROOT), "rev-parse",
                          "--abbrev-ref", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "?"
print(f"{_branch} 分支 pytest:", actual)
print("（判据只在两种状态下成立：骨架态有 failed，答案态全通过）")
print()
if ISSUES:
    print(f"发现 {len(ISSUES)} 个问题：")
    for i in ISSUES:
        print("  ✗", i)
    sys.exit(1)
else:
    print("✓ 文档与代码一致，未发现问题")
