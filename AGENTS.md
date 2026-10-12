# AGENTS.md

从零手抄一个 GPT 的教学仓库。架构对齐 nanochat，可读性对齐 nanoGPT。

## ⚠ 先看分支 —— 这是本仓库最容易踩的坑

`main` 与 `solution` **已分叉**（不是「同一份代码的两种视图」）：

| | `main` | `solution` |
|---|---|---|
| 性质 | 骨架态：`src/` 里待实现函数体是 `raise NotImplementedError` | 完整答案 |
| 教程章节 | 7 章（卷0+卷1） | 38 个编号文件 + README |
| 章节判据 | 194 个 | 224 个 |
| 判据自检 | 11 个 | 21 个 |
| 实测 | `71 failed, 126 passed, 8 skipped` | `243 passed, 2 skipped` |

- **在 `main` 上看到一堆 `failed` 是正确的**，不是环境坏了 —— 每一个 failed 就是接下来要写的一个函数。
- **在 `main` 上看到 `243 passed` 才说明走错了分支。**
- 两分支的**判据数量不同**，所以 README 里的「N 个测试」在两边本来就该不同。别跨分支搬运数字（实测踩过：把 solution 的 224 搬到 main 会报「判据被掏空」）。
- 分支约定：`git checkout solution` 看答案，`git checkout main` 回来。commit 用中文主题行 + 详细正文。

## 命令

```bash
uv sync                          # 建 .venv（torch 走 cu132 索引）
uv run pytest tests/ -q          # 全量判据
uv run pytest tests/test_core.py -k 'rmsnorm' -v   # 单条判据
uv run python scratch/audit_docs.py    # 文档审计（退出码 0 = 干净）
uv run python scratch/measure_noise_floor.py   # 重算消融显著性
bash script/progress.sh          # 逐章看「我该做哪一章」（solution 上 37 章全通过）
uv run ruff check .
```

无 lint/typecheck/CI/pre-commit 配置 —— `ruff check .` 是唯一的静态检查。

## 🔴 `runs/` 不在版本库里，但有判据依赖它

`.gitignore` 忽略 `runs/`，而 `tests/test_judging_soundness.py` 有 3 条判据要读
`runs/base_checkpoints/*/meta_*.json` 去核对实测数字。

**新克隆的仓库上这 3 条会失败**，这是环境缺数据、不是代码坏了：

```
test_noise_floor_numbers_match_the_actual_run_artifacts
test_every_ablation_measurement_on_disk_is_listed_in_the_docs
test_interaction_check_is_stored_and_mathematically_consistent
```

它们**没有 skip 分支**（这是有意的：判据的立场是「JSON 里的数字必须能在磁盘上
找到出处，否则就是编造的数据」）。本地有 `runs/` 时全绿。跑消融（每组约 39 分钟）
或从作者的 `runs/` 恢复即可。`scratch/audit_docs.py` 不受影响。

## 判据自检是这个仓库的核心机制

`tests/test_judging_soundness.py`（solution 上 21 条）**不测待实现函数，测判据本身**。
改判据前先读它的 docstring —— 那里记录了真实事故：

- `HyperParams(**kw)` **只登记 key、值恒为 0**，赋值只能走 `.set()`。传 kwargs 期待它生效 → 所有超参为 0 → NaN，且不报错。
- `assert (p != 1.0).any()` 在 `p` 全 NaN 时恒真，会长期充当假护栏。
- `progress.sh` 的 `-k` 表达式必须真的能选中 ≥1 个用例（曾经只检查「引用的文件存在」，`-k '不存在的关键字'` 照样通过）。

新增/删除判据后**必须同步 4 处**（`test_tutorial_claims_match_reality` 用精确相等）：

1. `scratch/audit_docs.py` 的 `CHAPTER_TESTS`
2. `tests/test_judging_soundness.py` 的 `minimum`
3. 文档里的「N 个测试」
4. 文档里的「M passed, K skipped」

顺序：改测试 → 跑 `audit_docs.py` → 按报错同步 → 再跑。

## 文档数字必须可证伪

本项目的纪律是「可证伪的声明就用可证伪的方式守住」。改任何写进文档的数字时：

- **倍数必须标分母。** 「0.02 比噪声大 N 倍」有三个合法分母，差近 3 倍：
  σ_Δ=0.000190 → **105 倍**（判定 `|Δ|`，项目口径）/ σ=0.000134 → 149 倍 /
  偏移=0.000437 → 46 倍。有判据逐行检查这件事。
- **实测数字与 `scratch/ablation_noise_floor.json` 对账**，该 JSON 的倍数由判据**重算**核对，不是比对硬编码字面量。
- 不要把某个档位的数字挪到另一个档位。`smoke` 档的噪声**至今没测过** —— 已知结论是「大于效应本身」，不要给它编一个数值。
- **「没测过」≠「测不出」**，两者在文档里必须分开写。

## 代码里几个必须照做的约定

- **静默失效是最危险的 bug 形状。** 手抄卷的错误会立刻 assert，只读卷的错误往往不报错只是没效果（本项目已踩：`ns_steps>5` 失效、`use_muon_plus` 名字盖着差 6 个数量级的行为、torch.compile 静默回落、KV cache 指针不推进）。改动后要**主动验证真的生效**，不要只看「测试没报错」。
- **名字不能盖着两种行为。** 已因此把 `use_muon_plus` 拆成 `use_frobenius_snap` / `use_muon_plus`（见 `src/optim/orthogonalize.py`），把 `weight_decay_final` 改名 `weight_decay_peak`。
- **`__init__` 只能在 meta device 上算形状和 dtype**，不能有任何真实数据操作；数据依赖的量（如 RoPE 表）必须在 `init_weights()` 里重算 —— `to_empty()` 只给未初始化的垃圾内存。
- **`GPT.init_weights(seed=None)` 默认按模型形状派生种子**（历史行为，所有已发布 val_bpb 基于它）。传 `--seed N` 才会换初始化。
- **`LayerNormAffine` 必须是真的 `nn.Module`**，不能用闭包捕获 γ/β —— 闭包里的参数不在 `model.parameters()` 里，optimizer 永远看不到。
- **KV cache 的写指针只在最后一层 `advance(T)` 一次**（`layers.py`）。漏了会「生成像胡言乱语但不报错」。
- 精度控制点只有一处：`model.layers.Linear.forward` 显式 cast，不用 `torch.amp.autocast`。

## 改代码前后

```bash
uv run pytest tests/ -q && uv run python scratch/audit_docs.py && uv run ruff check .
```

`audit_docs.py` 会核对：章节判据数、`N 个测试`/`N passed` 声明、文档引用的文件与
测试函数是否存在、断链、教程代码块与仓库文件的漂移。它是文档正确性的主要闸门。

**改判据/文档时不要动 `runs/` 下的实测产物** —— 那是 18 组真实跑批的 `val_bpb`
（15 组 `d6_*` 消融 + 3 次基线复跑），多个判据按精确相等核对它们。

⚠ **本文件（`AGENTS.md`）不在文档审计范围内** —— `audit_docs.py` 只扫
`doc/**/*.md` + `README.md`，`test_tutorial_claims_match_reality` 也只读那两个。
所以下面这些数字**没有机器守护**，改动判据数量时要手动回来核对：
`243 passed` / `224 个章节判据` / `21 个判据自检` / `71 failed, 126 passed, 8 skipped`。
（本文件只有 mojibake 那条守护会扫到，因为它的 glob 含根目录 `*.md`。）

## 环境

- Python ≥3.13，torch 走 `pytorch-cu132` 索引；`uv.lock` 被 gitignore（不保证可复现）。
- 本仓库的网络实测（2026-09）：`huggingface.co` / `raw.githubusercontent.com` / `pypi.org` 直连不可达。`script/_common.sh` 统一走 `hf-mirror.com` 与 `ghproxy.net`，并设 `OMP_NUM_THREADS=1`、`PYTORCH_ALLOC_CONF=expandable_segments:True`。**绕过 `_common.sh` 直接跑 python 会走不到镜像。**
- `pyproject.toml` 设了 `pythonpath = ["."]`，让 `from scratch.packing_demo import ...` 在 `pytest` 与 `python -m pytest` 两种调用方式下都成立。改判据时别绕过它。
- 实测硬件是 RTX 4060 Ti 16GB（单卡）。MFU、消融档耗时、显存数字都只对该卡成立，文档已标注不可外推。