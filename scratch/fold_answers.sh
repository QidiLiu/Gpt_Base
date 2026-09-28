#!/usr/bin/env bash
# 把教程里「手抄代码」小节下的代码块包进 <details>，做成「先自己想，再展开」。
# 只处理已经存在的章节（卷 0-1），后续章节用同样的格式手写。
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
import pathlib, re

TUT = pathlib.Path("doc/tutorial")

# 每个文件里，标记「从这里开始的代码块要折叠」
SECTIONS = ("### 第 1 块", "### 第 2 块", "### 第 3 块", "## 手抄代码")

def wrap_sections(md: str) -> str:
    lines = md.split("\n")
    out, i = [], 0
    n_wrapped = 0
    while i < len(lines):
        line = lines[i]
        if any(line.startswith(s) for s in SECTIONS):
            out.append(line)
            # 找到这一节里的 ```python ... ``` 块
            i += 1
            while i < len(lines):
                if lines[i].startswith("```python"):
                    # 收集整块
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith("```"):
                        j += 1
                    block = lines[i:j + 1]
                    # 检查块内是否已有 details
                    if any("details" in b for b in block):
                        out.extend(block)
                    else:
                        n_wrapped += 1
                        out.append("")
                        out.append("<details>")
                        out.append(f"<summary><b>👀 展开参考答案（先自己想 20 分钟）</b></summary>")
                        out.append("")
                        out.extend(block)
                        out.append("")
                        out.append("</details>")
                    i = j + 1
                    continue
                # 非代码块，遇到下一个 ## 就退出
                if lines[i].startswith("## ") and not lines[i].startswith("### 第"):
                    break
                out.append(lines[i])
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out), n_wrapped

total = 0
for f in sorted(TUT.glob("*.md")):
    if f.name == "README.md":
        continue
    md = f.read_text()
    if "<details>" in md:
        print(f"  跳过 {f.name}（已折叠）")
        continue
    new, n = wrap_sections(md)
    if n:
        f.write_text(new)
        total += n
        print(f"  {f.name}: 折叠 {n} 个代码块")
print(f"共折叠 {total} 个代码块")
PY
