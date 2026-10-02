# 36 · checkpoint 与续训

第 29 章第一次提到「续训要恢复四样东西」。
本章讲**那四样东西是怎么存和读的**，以及两个 JSON 的坑：

- **`inf` 存不进标准 JSON**（会变成 `null`）
- **`NaN` 也不能**（而且 `allow_nan=False` 会直接报错）

---

## 本章目标

- 说清 checkpoint 的文件布局和「为什么用 `.tmp` + rename」
- 理解 `inf ↔ null` 的往返，以及忘了还原会发生什么
- 知道 `find_latest` 的正则挡住了哪些干扰文件

---

## 前置回顾

[第 29 章](29-训练循环.md)：续训要恢复四样东西。
[第 16 章](16-meta-device三步建模型.md)：`load_latest` 复用 meta 三步法。

> **本章是只读教程。** 代码在 `src/common/checkpoint.py`。

---

## 文件布局

```python
def _model_path(ckpt_dir: str, step: int) -> str:
    return os.path.join(ckpt_dir, f"model_{step:06d}.pt")

def _meta_path(ckpt_dir: str, step: int) -> str:
    return os.path.join(ckpt_dir, f"meta_{step:06d}.json")
```

**每个 step 两个文件**：

```
runs/base_checkpoints/d6_smoke/
├── model_000003.pt      ← 权重（torch.save）
├── meta_000003.json     ← 元信息（json.dump）
├── history.csv          ← 第 29 章 write_history 生成的
├── curves.png
└── ...
```

模块 docstring 解释了三个设计选择：

> 刻意做成「**扁平的多文件**」而不是 nanochat 那种分 rank 的 ZeRO 存档：
> 单卡项目不需要优化器状态分片。但保留了几个 nanochat 的好设计：
> - 存档目录按 `model_tag` 分（d4 / d6 / d12 ...），不同实验互不干扰
> - **metadata 存成 JSON 而不是 pickle** —— 可以直接 `cat` 出来看
> - 记录完整 `run_config`，这样「这个模型是怎么训出来的」永远不会丢

⚠ **但优化器状态不在 checkpoint 里** —— `muon_step` 的 docstring 记着：

> 注意优化器状态不在 `state_dict` 里！这是本实现的一个刻意的简化：
> `state_dict` 只存参数，AdamW 的 `exp_avg`/`exp_avg_sq` 和 Muon 的
> `momentum_buffer` 全部在内存里，**不参与 checkpoint**。
> 后果是断点续训会丢失优化器动量，头几十步会有一段小抖动。
> 对本项目（几百到几千步）可接受。

**这是第 25 章那 647 MiB 的直接后果** ——
省了显存，代价是续训要重新暖动量。

---

## 概念：`.tmp` + `os.replace` —— 原子写入

```python
mp = _model_path(ckpt_dir, step)
torch.save(model_state, mp + ".tmp")
os.replace(mp + ".tmp", mp)
```

docstring：

> 为什么要 `.tmp` + rename？
> 断电/中断时如果直接写目标文件，会留下一个半截的 `.pt`，
> 下次 resume 就 load 不出来。**rename 是同文件系统内的原子操作。**

**这是 POSIX 的保证**：`rename(2)` 在同一文件系统内是原子的 ——
要么文件完整地出现在目标名，要么完全不存在。**不会出现「半个文件」。**

### ⚠ 但 meta 文件没有这个保护

```python
with open(_meta_path(ckpt_dir, step), "w", encoding="utf-8") as f:
    json.dump(_json_safe(meta), f, indent=2, default=str, allow_nan=False)
```

**直接写目标文件，没有 `.tmp`。**

所以存在一个理论上的窗口：`.pt` 写完了，`json` 写到一半断电
→ `.pt` 完整但 `.json` 损坏 → `json.load` 抛异常 → 续训崩溃。

⚠ **这不会导致「读到半截权重」**（`.pt` 是原子的），
但会导致「续训时 meta 读不出来」。

**判据 `test_save_leaves_no_tmp_files` 守的是「不留 `.tmp`」**，
没管 meta 的原子性。

**这是本章找到的一个真实的（非致命的）不一致。**

---

## 概念：★ `_json_safe` —— 为什么需要它

```python
def _json_safe(obj):
    """
    把 meta 里的非有限浮点（inf / nan）换成 None。

    为什么要这一步：`best_val_bpb` 在「还没验证过」时是 `float('inf')`。
    `json.dump` 默认会写成 `Infinity` —— 那是**非标准 JSON**，
    Python 自己读得回来，但 jq 和其它语言的解析器会直接报错。
    而这个文件的卖点正是「存成 JSON 可以直接 cat 出来看」。
    """
```

### 实测

```python
import sys, json, math
sys.path.insert(0, "src")
from common.checkpoint import _json_safe

obj = {"best_val_bpb": float("inf"), "loss": float("nan"),
       "nested": {"a": float("-inf"), "b": 1.5},
       "list": [float("inf"), 2.0, float("nan")]}
print("输入", obj)
print("输出", _json_safe(obj))
```

实测：

```
输入 {'best_val_bpb': inf, 'loss': nan, 'nested': {'a': -inf, 'b': 1.5}, 'list': [inf, 2.0, nan]}
输出 {'best_val_bpb': None, 'loss': None, 'nested': {'a': None, 'b': 1.5}, 'list': [None, 2.0, None]}
```

**递归处理 dict / list / tuple。**

⚠ **tuple 变成 list** —— JSON 没有 tuple 类型。
所以 `run_config` 里的 tuple（如果用 tuple 而不是 list）
往返后会变成 list。**对当前项目无害**（config 里没有 tuple 字段）。

### 不做这一步会怎样

```
不加 _json_safe 直接 json.dumps(obj, allow_nan=False):
    ValueError: Out of range float values are not JSON compliant
    （或 allow_nan=True 时写出 "Infinity"）
```

而 `allow_nan=True`（默认）时写出的内容：

```
{"best_val_bpb": Infinity, "loss": NaN, ...}
```

**`Infinity` 和 `NaN` 是 JavaScript 的扩展，不是标准 JSON。**
Python 的 `json.load` 读得回来（宽松模式），
但 `jq`、`JSON.parse`、大多数语言的解析器会**直接报错**。

实测严格解析：

```
严格解析失败: 非标准常量 Infinity
```

### `allow_nan=False` 是第二道防线

```python
json.dump(_json_safe(meta), f, indent=2, default=str, allow_nan=False)
```

注释：

> `allow_nan=False`：万一还有漏网的非有限值，宁可报错也不写脏 JSON

**「宁可报错也不写脏 JSON」** —— 训练会因为一次存档失败而崩，
但那比存下一个别的工具读不了的文件好。

⚠ `default=str` 是另一个兜底：**遇到不认识的对象就 `str()` 化**。
所以 `ModelConfig` 之类的 dataclass 也能存（变成它的 repr 字符串）。

**代价**：读回来那个字段是字符串，不是对象。
所以 `load_latest` 里要 `ModelConfig(**cfg_dict)` ——
如果 `model_config` 被 `str()` 化了那就会失败。

**实测 `model_config` 是正常存的 dict**（因为第 29 章用了
`asdict(cfg.model)`），所以走的是 JSON 原生路径。

---

## 概念：★ `inf ↔ null` 的往返

```python
# 存
save_checkpoint(ckpt_dir, step, state, {"best_val_bpb": float("inf"), ...})

# 取（第 29 章）
_bpb = meta.get("best_val_bpb")
best_bpb = float("inf") if _bpb is None else float(_bpb)
```

实测存档文件内容：

```json
{
  "model_config": { "n_layer": 6, "n_embd": 384 },
  "run_config": { "mode": "ablation" },
  "step": 3,
  "best_val_bpb": null,
  "dataloader_state": { "pq_idx": 0, "rg_idx": 1, "epoch": 1 },
  "history": [...]
}
```

**`"best_val_bpb": null` 就是「还没验证过」。**

### 忘了还原会怎样

```python
best_bpb = meta.get("best_val_bpb")      # None
...
best_bpb = min(best_bpb, bpb)            # min(None, 1.2) -> TypeError
```

**第一次验证时崩溃。**

判据：`test_resume_with_unset_best_bpb_reads_back_as_none`
—— 它守的是**存读往返后是 `None`**（不是 `inf`），
所以调用方**必须**自己还原。

**这是一个「类型契约」**：
`_json_safe` 保证存进去的是 `None`，
`train_base` 保证读出来之后转回 `float("inf")`。
**两端都要写对才算完整。**

---

## 概念：`find_latest` 的正则挡住了什么

```python
def find_latest(ckpt_dir: str):
    """找出最大的 step 存档号；没有则返回 None。"""
    if not os.path.isdir(ckpt_dir):
        return None
    steps = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"model_(\d{6})\.pt", f)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None
```

### 实测：干扰文件

放 6 个干扰文件：

```
model_abc.pt           非数字
model_123.pt           3 位（不是 6 位）
model_1234567.pt       7 位
model_000005.pt.tmp    ★ 半截文件
meta_000005.json       ★ meta 不是 model
history.csv            无关
curves.png             无关
```

实测：

```
放了 step 1/10/100/999999 + 6 个干扰文件
find_latest -> 999999  （应为 999999）
```

**全部被正确排除。**

### 四个设计点

| 设计 | 挡住什么 |
|---|---|
| `\d{6}` | 非 6 位数字的（`model_abc.pt`、`model_123.pt`、`model_1234567.pt`） |
| `fullmatch` 而非 `match` | `xxmodel_000005.pt`（前缀污染） |
| 只匹配 `model_` 前缀 | `meta_000005.json` |
| **不匹配 `.tmp`** | **断电残留的半截文件** |

**最后一条最重要。** 如果正则写成 `r"model_(\d{6})\.pt"`（不 fullmatch）
或者用 `startswith("model_")`，那么 `model_000005.pt.tmp` 会被匹配 ——
然后 `load_checkpoint` 去读一个不存在的 `model_000005.pt`，
或者更糟：读到上次的半截文件。

**实测确认**：目录里存在 `model_000005.pt.tmp` 时
`find_latest` 仍返回 999999 ✓

### `find_latest` 的两个判据

```bash
uv run pytest tests/test_checkpoint.py -k find_latest -v
```

- `test_find_latest_empty_dir_returns_none`
- `test_find_latest_ignores_non_conforming_names`
- `test_find_latest_picks_max_step`

---

## 概念：`load_latest` —— 从 meta 重建模型

```python
def load_latest(run_root: str, tag: str, device):
    ckpt_dir = os.path.join(run_root, "base_checkpoints", tag)
    step = find_latest(ckpt_dir)
    assert step is not None, (
        f"{ckpt_dir} 里没有存档。先跑 `bash script/train_base.sh <mode>`。")
    meta = load_checkpoint(ckpt_dir, step, "cpu", "meta")
    # ModelConfig 有嵌套的默认值，用 filter 丢掉不认识的新字段
    fields = set(ModelConfig.__dataclass_fields__)
    cfg_dict = {k: v for k, v in meta["model_config"].items() if k in fields}
    cfg = ModelConfig(**cfg_dict)

    # 复用 build_model 的 meta device 流程，然后用存档覆盖
    from model.gpt import build_model
    model = build_model(cfg, device=device)
    model.load_state_dict(load_checkpoint(ckpt_dir, step, device, "model"))
    return model, get_tokenizer(), meta
```

**三步**：

1. 从 meta 恢复 `ModelConfig`
2. `build_model` 建出**正确形状**的模型
3. `load_state_dict` 用存档权重覆盖

⚠ **第 2 步会先随机初始化一遍**（第 16 章的 `init_weights`），
然后第 3 步全部覆盖。**所以那次初始化是纯浪费** ——
但这是 meta 三步法的代价（它必须先有形状才能分配内存）。

### 那个 filter

```python
cfg_dict = {k: v for k, v in meta["model_config"].items() if k in fields}
```

**前向兼容**：老 checkpoint 的 meta 里如果有后来新增的字段，
`ModelConfig(**...)` 会因为「不认识的参数」而 `TypeError`。
过滤掉就没事。

⚠ **但它是静默的** —— 一个拼错的字段名会被丢掉，
然后 `ModelConfig` 用默认值。

**后果**：模型形状可能不匹配，`load_state_dict` 报
`size mismatch` —— **还算好**（至少会崩）。
如果形状碰巧一致（比如两个字段都是 384），
**就完全静默地用了错的配置**。

第 34 章的 `load_sft_model` 里也有一份同样的逻辑 ——
**两处重复，都有同样的风险**。

---

## 动手验证

### 验证 1：`_json_safe` 和严格解析

见上面「实测」。**关键是 `Infinity` 不是标准 JSON。**

### 验证 2：往返

```python
import sys, shutil, torch
sys.path.insert(0, "src")
from common.checkpoint import save_checkpoint, load_checkpoint

D = "/tmp/opencode/ckpt_probe"
shutil.rmtree(D, ignore_errors=True)
save_checkpoint(D, 3, {"w": torch.randn(10, 10)},
                {"model_config": {"n_layer": 6}, "step": 3,
                 "best_val_bpb": float("inf"),
                 "history": [{"step": 1}, {"step": 2}]})
back = load_checkpoint(D, 3, "cpu", "meta")
print(f"best_val_bpb 读回来是 {back['best_val_bpb']!r}")
print(f"history 有 {len(back['history'])} 条")
shutil.rmtree(D, ignore_errors=True)
```

**期望**：`best_val_bpb` 是 `None`（不是 `inf`）。

### 验证 3：`find_latest` 的干扰文件

见上面。**关键：`model_000005.pt.tmp` 不被匹配。**

```bash
uv run pytest tests/test_checkpoint.py -v
```

### 验证 4：真的存一个 checkpoint 看看

```bash
uv run python -m training.train_base --mode smoke --num-iterations 3 \
    --no-resume --save-every 3 --model-tag ckpt_demo

ls -la runs/base_checkpoints/ckpt_demo/
cat runs/base_checkpoints/ckpt_demo/meta_000003.json | head -30
```

⚠ **直接 `cat` JSON 是这个设计的卖点**（docstring 明说
「可以直接 cat 出来看」）。

---

## 常见坑

### 坑 1：以为 checkpoint 含优化器状态

**不含。** 续训会丢动量，第 25 章那 647 MiB 的代价。

### 坑 2：忘了把 `None` 还原成 `inf`

**第一次验证就 `TypeError`。**

### 坑 3：用 `startswith` 而不是 `re.fullmatch` 找存档

**会匹配到 `.tmp` 半截文件。**

### 坑 4：meta 的写入没有原子保护

`.pt` 有（`.tmp` + `replace`），**`.json` 没有**。
断电窗口里可能出现「权重完整 + meta 损坏」。

### 坑 5：`json.dump` 不加 `allow_nan=False`

**「宁可报错也不写脏 JSON」** 这条防线就没了。

### 坑 6：`load_latest` 的字段过滤是静默的

拼错的字段名被丢掉 → 可能用了错的默认值。

### 坑 7：`_json_safe` 把 tuple 变成 list

当前无害（config 无 tuple 字段），但往返后类型变了。

---

## 延伸

**为什么不用 pickle 存 meta**

module docstring 说「存成 JSON 而不是 pickle —— 可以直接 cat 出来看」。

**更重要的原因是安全**：pickle 可以执行任意代码。
从一个不可信的 checkpoint 恢复 = 任意代码执行。

⚠ **权重用 `torch.save` 也是 pickle 格式** ——
所以有 `weights_only=True`：

```python
return torch.load(path, map_location=device, weights_only=True)
```

实测确认：`{"w": Tensor, "obj": dict}` 能读（都是白名单类型）。

**如果存了自定义类，`weights_only=True` 会直接拒绝** ——
这意味着**你不能把自己的 nn.Module 类 pickle 进 checkpoint**。

实测（本项目）：

```
读出来: {'w': 'Tensor', 'obj': 'dict'}
weights_only=True 允许 dict / tensor / 基本类型
```

**所以第 29 章存的 `model.state_dict()` 只能是
「str -> tensor」的字典** —— 这也是 PyTorch 的约定。

**ZeRO / FSDP 的分片存档**

nanochat 那种「每个 rank 一个文件」的布局：
读的时候要按 rank 拼回来。

本项目单卡，所以是「一个文件」。

**history.csv 与 curves.png**

第 29 章的 `write_history` 生成的：

> 把训练历史存成 CSV + 画曲线。**matplotlib 失败不影响训练结果。**

⚠ 「matplotlib 失败不影响」这个设计很重要 ——
教学项目不应该因为缺一个可视化库就训不了模型。

**这四样东西里哪个最重要**

| 东西 | 丢了会怎样 |
|---|---|
| `model_*.pt` | 模型没了 |
| `meta_*.json` | **不知道这个模型是怎么训出来的** |
| `history.csv` | 曲线没了（但不影响模型质量） |
| `curves.png` | 同上 |

**meta 是 nanochat 那个「记录完整 run_config，这样
『这个模型是怎么训出来的』永远不会丢」的价值所在。**

---

## 下一章

[第 37 章：全流程与排错](37-全流程与排错.md) ——

**卷 5-8 的最后一章。** 把 36 章的东西串成一条可执行的命令链，
外加一份「出问题了先看哪里」的排查清单。

判据：`tests/test_judging_soundness.py` 的 10 个自检 ——
它们检查的就是「判据本身有没有被改坏」。
