# 从零手抄一个 GPT

> **目标**：不是训出一个好模型，而是让你**亲手把 GPT 的每个组件写一遍**，
> 并且用数字看到每个设计到底值多少钱。

---

## ⚠️ 先搞清楚：现在 `src/` 里是**骨架**，不是答案

```
src/data/tokenizer.py       每个函数体是 raise NotImplementedError
                            + 一段「这一步在干什么」的注释 + 验证方法
src/model/layers.py         同上
src/model/gpt.py            同上
src/optim/*.py              同上
```

`raise NotImplementedError` 的消息里写清楚了**要做哪几步**和**怎么验证**。
比如：

```
NotImplementedError: 待实现：attend —— 见 docstring 的三步
  1) 算 gqa = (q.size(2) != k.size(2))   # 头数不同即 GQA
  2) 如果窗口是全上下文 -> sdpa_bt(q,k,v, causal=True, gqa=gqa)
  3) 否则构造 mask：...
验证：uv run pytest -k "attend or causal" -v
```

**但只读代码不敲是过不了判据的** —— 每个函数都有一组 pytest 用例守着。
这不是建议，是硬约束。

---

## 两条推进路线

| 路线 | 做法 | 适合 |
|---|---|---|
| **A. 跟敲（推荐）** | 照着 `raise` 消息 + 教程概念部分自己实现，跑通判据再往下走 | 想真正理解，卷1-4 约 8-10 小时 |
| **B. 通读** | `git checkout solution` 拿到完整实现，直接读 | 只想建立全貌，3-4 小时 |

**「手抄」的适用范围是有原则的**，不是「全部都抄」：

| 范围 | 手抄？ | 理由 |
|---|---|---|
| 卷 1 数据（680 行） | ✅ 抄 | best-fit 装箱值得亲手写出来，这是全项目最有价值的一段 |
| 卷 2-3 模型（810 行） | ✅ 抄 | 组件小、边界清晰，逐个敲能建立对 shape 的手感 |
| 卷 4 优化器（630 行） | ✅ 抄 | `orthogonalize_simple` 那 20 行必须懂 SVD 才写得出来 |
| 卷 5 训练循环（1500 行） | 📖 **只读** | 纯工程（日志、存档、调度循环），抄一遍不产生理解 |
| 卷 6-8 分布式/对齐/收尾 | 📖 **只读** | 同上 |

判据很简单：**抄的时候你需不需要做决策？**
需要 → 抄（best-fit 选最长？mask 给多少？），不需要 → 读
（`log0(f"训练步数 {n}")` 这种）。

---

## 环境自检 + 进度追踪

```bash
cd /path/to/Gpt_Base          # 换成你自己的仓库路径
uv sync
bash script/progress.sh        # 告诉你哪一章还没做完
```

`progress.sh` 逐章列出完成判据：

```
章节   状态     完成判据
------------------------------------------------------------------
00-01  通过     tests/test_presets.py -k 'presets'
02     未完成   tests/test_data.py -k 'tokenizer or special or decode_bytes'
03     未完成   tests/test_data.py -k 'tokenizer or decode_bytes'
04     未完成   tests/test_data.py -k 'mask or python_output or truncate or alternation'
05     未完成   tests/test_data.py -k 'dataloader or targets_are or bos or padding or bestfit or split'
...
```

只看某一章：`bash script/progress.sh 05`

---

## 你的硬件够用吗？

项目按 16 GB 单卡（RTX 4060 Ti）设计。三档规模：

| 档位 | 模型 | 步数 | 耗时 | 峰值显存 |
|---|---|---|---|---|
| `debug` | d2, 32 维, 1 头 | 20 | **秒级** | < 100 MB |
| `smoke` | d4, 128 维, 4 头 | 896 | **2-3 分钟** | 650 MB |
| `full` | d6, 384 维, 6 头 | 3096 | **20-25 分钟** | ~3 GB |

> **一个必须先接受的事实**
>
> 16 GB 消费卡比 nanochat 的目标硬件（8×H100 80GB）弱约 **1250 倍（FLOPs）**。
> 所以 `full` 档训出来的模型很弱，生成的文本只是「有点像英文」。
> 这不影响本教程的价值 —— 价值在**消融实验**：
> 同一个模型，开/关某个 trick，val bpb 差 0.01-0.09。
> 那些小数字才是你要亲手测出来的东西。

---

## 目录与卷的对应关系

```
src/
├── common/       设备、精度、路径、网络镜像、单一旋钮 config      → 卷0  📖只读
├── data/         tokenizer / dataset / dataloader / tasks         → 卷1  ✍手抄
├── model/        layers（零件） + gpt（组装与 trick）              → 卷2-3 ✍手抄
├── optim/        orthogonalize（正交化） + muon（优化器类）       → 卷4  ✍手抄
├── training/     train_base / train_sft / eval_* / chat          → 卷5  📖只读
├── inference/    engine（KV cache 生成 + 工具调用状态机）         → 卷7  📖只读
└── evaluation/   metrics（bpb / 多选 / pass@k）                  → 卷7  📖只读
```

> ⚠ **两个例外**：`training/train_tokenizer.py` 的 `iter_parquet_text`（第 02 章）与
> `compute_token_bytes`（第 06 章）、以及 `evaluation/metrics.py` 的 `evaluate_bpb`
> （第 06 章）是**卷1 的手抄目标**，虽然文件本身归属卷5/卷7。
> 判据：`pytest tests/test_data.py -k 'parquet or token_bytes or bpb'`。

---

## 教程目录

> 🚧 = 教程正文还没写（代码骨架已就位，第二批交付）。
> **判据测试和 `src/` 里的骨架提示已经可以用了**，所以你可以先敲代码，
> 教程正文随后补上。（未写的章节不做超链接，避免点进去是死链。）

### 卷 0 · 准备　📖 只读
| 章 | 标题 | 核心问题 |
|---|---|---|
| [00](00-如何使用本教程.md) | 如何使用本教程 | 怎么安排学习顺序、怎么查进度 |
| [01](01-环境与全景图.md) | 环境与全景图 | 一次训练的数据流长什么样 |

### 卷 1 · 数据：文本怎么变成 tensor　✍ 手抄
| 章 | 标题 | 完成判据指向 |
|---|---|---|
| [02](02-为什么需要分词器.md) | 为什么需要分词器 | `-k tokenizer` |
| [03](03-手写一个玩具BPE.md) | 手写一个玩具 BPE | `-k decode_bytes` |
| [04](04-对话模板与特殊token.md) | 对话模板与特殊 token | `-k mask or python_output or truncate` |
| [05](05-文档打包BOS-aligned-bestfit.md) | **文档打包：BOS-aligned best-fit** | `-k dataloader or bos or bestfit` |
| [06](06-bits-per-byte.md) | bits per byte | `scratch/bpb_demo.py` |

### 卷 2 · 模型：tensor 怎么变成 logits　✍ 手抄
| 章 | 标题 | 完成判据指向 |
|---|---|---|
| 🚧 07 | 先跑通一个最小 GPT | `-k uniform or causal` |
| 🚧 08 | RMSNorm | `-k uniform` |
| 🚧 09 | **RoPE 旋转位置编码** | `-k rope` |
| 🚧 10 | 注意力三步曲 | `-k attend or sliding` |
| 🚧 11 | SDPA 与 FlashAttention | `-k sdpa` ★ 有个大坑 |
| 🚧 12 | QK Norm 与 GQA | `-k attend` |
| 🚧 13 | MLP 与激活函数 | `-k uniform` |
| 🚧 14 | 残差流与 Pre-LN | `-k causal` |
| 🚧 15 | 权重绑定与 logit softcap | `-k sampling or topk` |

### 卷 3 · nanochat 的架构 trick　✍ 手抄
| 章 | 标题 | 完成判据指向 |
|---|---|---|
| 🚧 16 | **meta device 三步建模型** | `-k uniform` ★ 有个 NaN 坑 |
| 🚧 17 | resid_lambdas 与 x0_lambdas | `-k all_tricks` |
| 🚧 18 | Value Embeddings 与门控 | `-k all_tricks` |
| 🚧 19 | Smear 与 Backout | `-k all_tricks` |
| 🚧 20 | 滑动窗口注意力 | `-k sliding` |

### 卷 4 · 优化器（最硬核的一卷）　✍ 手抄
| 章 | 标题 | 完成判据指向 |
|---|---|---|
| 🚧 21 | 从 SGD 到 AdamW | `-k adamw` |
| 🚧 22 | 为什么矩阵参数适合 Muon | `-k adamw` |
| 🚧 23 | **手写 Newton-Schulz 正交化** | `-k orthogonalize` ★ 20 行 |
| 🚧 24 | Polar Express | `-k polar` |
| 🚧 25 | MuonEq / Muon+ / NorMuon | `-k muon_plus or nor_muon` |
| 🚧 26 | 谨慎权重衰减 | `-k cautious` |
| 🚧 27 | 混合优化器与参数分组 | `-k grouping` |
| 🚧 28 | torch.compile 融合与 0-D tensor 技巧 | `-k hyperparams` |

### 卷 5-8 · 训练循环、分布式、对齐、收尾　📖 只读
*（第二批交付：代码已在 `src/` 里，是只读的。教程随第二批补上）*

卷 5 训练循环 · 卷 6 分布式 · 卷 7 推理与 SFT · 卷 8 全流程与排错

---

## 每章的固定结构

```
## 本章目标         学完能回答什么问题
## 前置回顾         链接上一章
## 概念             why，配小例子
## ❗ 你要敲的        列出这一章要实现哪几个函数（src/ 里已留骨架）
## 动手验证         1 分钟内能跑完的实验 + 实测输出
## ★ 消融实验       开关关掉，数字变化多少
## 常见坑
## 延伸            nanochat 完整版怎么做的，差异在哪
```

**★ 消融实验是本教程的核心。** 每章的「延伸」告诉你 nanochat 怎么做的，
「消融实验」让你亲手验证它到底值多少。全部做完，你会得到一张表：

| 变体 | val_bpb | Δ | 解读 |
|---|---|---|---|
| 基线（RoPE + QK-Norm + ReLU²） | 1.7136 | — | |
| **去掉 RoPE** | 1.8013 | **+0.0877** | 最大的单项影响 |
| **GELU 替代 ReLU²** | 1.7240 | **+0.0104** | relu² 略好 |
| 去掉 QK Norm | 1.7098 | −0.0038 | 噪声（\|Δ\|<0.02，要重跑） |

（这是 smoke 档实测。`full` 档数字会不同，量级关系应该一致。）

---

## scratch 脚本什么时候能跑

`scratch/` 里的实验脚本有些依赖你还没实现的部分。**先看这张表再跑**，
免得看到 `NotImplementedError` 以为环境坏了。

| 脚本 | 依赖 | 状态 | 解锁条件 |
|---|---|---|---|
| `scratch/hardware.py` | `common` | ✅ **现在就能跑** | 卷 0 |
| `scratch/toy_bpe.py` | 纯 Python | ✍ **手抄目标** | 敲完 `toy_bpe()` 再跑（第 02 章验证 2） |
| `scratch/vocab_math.py` | `common.config` | ✅ **现在就能跑** | 卷 0 |
| `scratch/bpb_uncomparable.py` | 纯 math | ✅ **现在就能跑** | 卷 0（它是卷 1 的材料） |
| `scratch/mini_bpe.py` | 纯 Python + `data.dataset` | ✍ **手抄目标** | 敲完 `MiniBPE` 的 4 个方法（第 03 章） |
| `scratch/mask_demo.py` | `tokenizer.load` → `render_conversation` | 🔒 | 卷1 第 04 章 |
| `scratch/check_data.py` | `tokenizer.load` → `dataloader` | 🔒 | 卷1 第 05 章（只有断言，无可抄算法） |
| `scratch/naive_dataloader.py` | `tokenizer.load` → `list_parquet_files` | ✍ **手抄目标** | 敲完两个函数再跑（第 05 章消融基线） |
| `scratch/packing_demo.py` | `tokenizer.load` → 三个装箱函数 | ✍ **手抄目标** | 敲完三个装箱函数再跑（第 05 章） |
| `scratch/bpb_demo.py` | `tokenizer.load` → checkpoint + `dataloader` | 🔒 | 卷1 第 06 章 |
| `bash script/train_base.sh` | 上面全部 + 模型 + 优化器 | 🔒 | 卷 1-4 全部完成 |
| `bash script/eval_base.sh` | checkpoint | 🔒 | 至少跑过一次 `train_base` |
| `bash script/train_sft.sh` | `data.tasks`（只读）+ 预训练 ckpt | 🔒 | 卷1 + 卷2-4 |
| `bash script/eval_sft.sh` | 同上 | 🔒 | 跑过一次 `train_sft` |
| `bash script/chat.sh` | 有 checkpoint 就能跑 | 🔒 | 跑过一次 `train_base`（无 SFT 时用基座） |
| `bash script/progress.sh` | 无 | ✅ **现在就能跑** | 随时 |

> 🔒 / ✍ 意味着跑到那一步会报 `NotImplementedError: 待实现：xxx`。
> **那不是 bug，是你还没写那一块。**
>
> 「依赖」列写的是**第一个真正卡住的地方**，不是这个脚本想演示的功能。
> 实测：除 3 个纯 Python 脚本外，其余全部先卡在 `tokenizer.load` ——
> 也就是说在卷1 第 02-03 章完成前，它们报的都不是自己想演示的那个待实现。

想跳过等待看效果：

```bash
git checkout solution          # 完整实现，scratch 和脚本全部可跑
git checkout main              # 看完切回来
```

---

## 五个入口脚本　📖 只读，直接用

```bash
bash script/train_base.sh smoke   # 预训练（自动串好 下载数据→训tokenizer→训练）
bash script/eval_base.sh  smoke   # val bpb + 多选题 + 采样
bash script/train_sft.sh  smoke   # 对话微调
bash script/eval_sft.sh   smoke   # 多选准确率 + GSM8K pass@1 + 多轮展示
bash script/chat.sh       smoke   # 交互式聊天
```

档位换成 `debug`（秒级，适合单步调试）或 `full`（20-25 分钟）。

> ⚠️ **手敲完卷 1-4 之前，这些脚本跑不通是正常的** ——
> 它们依赖你还没实现的那部分。等 `bash script/progress.sh` 全绿，它们就能跑。

---

## 卡住了怎么办

**第一步永远是跑测试**：

```bash
uv run pytest tests/ -v
```

66 个测试（卷0 6 个 + 卷1 19 个 + 卷2-3 18 个 + 卷4 12 个 + 卷7 只读护栏 11 个）
专门覆盖「看起来对其实错了」的 bug。你在敲的过程中改坏东西，
测试会立刻告诉你哪里坏了。几个重点：

| 测试 | 锁住的坑 |
|---|---|
| `test_sdpa_layout_is_bhtd_not_bthd` | SDPA 严格按 `(B,H,L,E)` 解读，`T==1` 时**静默算错** |
| `test_causality_no_future_leak` | attention 的 causal mask 漏了（最严重的 bug） |
| `test_rope_relative_position` | RoPE 的相对位置性质 |
| `test_every_row_starts_with_bos` | 手动 `row[0]=bos` 导致开头两个 BOS |
| `test_truncate_left_keeps_supervision` | SFT 截断把监督信号全切掉 → NaN |
| `test_muon_plus_rescues_low_rank` | Newton-Schulz 推不动近零奇异值 |
| `test_forward_loss_starts_uniform` | 初始 loss 应 ≈ `ln(vocab_size)`，nan 说明初始化坏了 |
| `test_param_grouping_covers_all_params` | 漏掉参数（症状隐蔽：训练正常但某模块一直是随机初始化） |

**loss 变 nan 的三个常见原因**：

1. 某个张量没被正确初始化（检查是不是在 meta device 上创建了却没重算 —— 卷3 第 16 章）
2. 整批 `targets` 都是 `-1`（`F.cross_entropy` 算 0/0 —— 卷1 第 04 章）
3. 某个除法/范数分母为 0

---

## 想看答案的时候

```bash
# 只看某一个文件的完整实现
git show solution:src/model/layers.py

# 临时切到完整版本跑一遍（记得先 commit 你的进度）
git checkout solution && bash script/train_base.sh smoke
git checkout main

# 看你写了多少
git diff --stat
```

**建议**：卡住超过 20 分钟再看答案。看完之后**关掉答案重新写一遍** ——
「看懂了」和「写得出」之间差着整个理解。
