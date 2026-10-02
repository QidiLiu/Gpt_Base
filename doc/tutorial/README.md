# 从零手抄一个 GPT

> **目标**：不是训出一个好模型，而是让你**亲手把 GPT 的每个组件写一遍**，
> 并且用数字看到每个设计到底值多少钱。

本教程假设你从空目录开始，跟着敲。每一章的代码块都可以直接复制，
敲进 `src/` 下对应的文件。

---

## 两条推进路线

| 路线 | 做法 | 适合 |
|---|---|---|
| **A. 跟敲（推荐）** | 每章的「手抄代码」块自己敲进 `src/`，不看现成代码 | 想真正理解，预算 8-11 小时（+4-8 小时消融） |
| **B. 通读** | 直接读 `src/` 里的成品代码，本教程当注释读 | 想快速建立全貌，预算 3-4 小时 |

如果你在 `solution` 分支（`git checkout solution`），说明路线 B 的代码已经在你机器上了，
可以直接从「动手验证」小节开始。逐章时间预算见 [第 00 章](00-如何使用本教程.md)。

---

## 环境自检

```bash
cd ~/Dev/Gpt_Base
uv sync                      # 建 .venv，torch 走 cu132
uv run pytest tests/ -q      # main 分支：69 failed, 116 passed, 7 skipped
bash script/progress.sh      # 逐章告诉你「我该做哪一章」
```

> ### ⚠️ 看到一堆 `failed` 是**正确**的，不是环境坏了
>
> `main` 分支（也就是你克隆下来的样子）的 `src/` 是**骨架**：
> 每个要你实现的函数体都是 `raise NotImplementedError`。
> 所以 `pytest` 的**预期结果就是一堆 failed** ——
> **每一个 failed 就是你接下来要写的一个函数。**
>
> ```
> FAILED tests/test_data.py::test_every_row_starts_with_bos - NotImplementedError: 待实现：make_dataloader ...
> ```
>
> 想知道「我该做哪一章」，跑 `bash script/progress.sh`；
> 想知道这些 failed 到底是什么意思，见 [第 01 章](01-环境与全景图.md) 的
> 「⚠️ 先搞清楚：你现在看到的『一堆失败』是**正确**的」。
>
> `git checkout solution` 后再跑，会看到 `220 passed, 2 skipped` —— 那才是全绿。
> **在 `main` 上看到 `220 passed` 才说明你走错了分支。**

`bash script/train_base.sh smoke` 在卷 1-4 敲完之前**跑不通是正常的**
（它依赖你还没实现的那部分），不用 troubleshoot。

**你的硬件够用吗？** 本项目按 16 GB 单卡（RTX 4060 Ti）设计。
四档规模：

| 档位 | 模型 | 步数 | 耗时 | 峰值显存 |
|---|---|---|---|---|
| `debug` | d2, 32 维, 1 头 | 20 | **秒级** | < 100 MB |
| `smoke` | d4, 128 维, 4 头 | 896 | **2-3 分钟** | 650 MB |
| `ablation` | d6, 384 维, 6 头 | 3096 | **约 44 分钟** | 6.2 GB |
| `full` | d24, 768 维, 12 头 | 2088 | **约 41 小时** | 12.8 GB |

`ablation` 是**消融专用**档，基线保持中性（Muon `simple` + trick 全关）；
`full` 是**消融后的最佳组合**（trick 全开 + Muon `advanced`），只训一次。
两者回答的问题不同，所以超参数取法正好相反。

> **一个必须先接受的事实**
>
> 16 GB 消费卡比 nanochat 的目标硬件（8×H100 80GB）弱约 **1250 倍（FLOPs）**。
> 所以 `full` 档训出来的模型很弱，生成的文本只是「有点像英文」。
> 这不影响本教程的价值 —— 本教程的价值在**消融实验**：
> 同一个模型，开/关某个 trick，val bpb 差 0.01-0.05。
> 那些小数字才是你要亲手测出来的东西。

---

## 目录与卷的对应关系

`src/` 按功能分包，**一个包 = 一卷**：

```
src/
├── common/       设备、精度、路径、网络镜像、单一旋钮 config      → 卷0
├── data/         tokenizer / dataset / dataloader / tasks         → 卷1
├── model/        layers（零件） + gpt（组装与 trick）              → 卷2 卷3
├── optim/        orthogonalize（正交化） + muon（优化器类）       → 卷4
├── training/     train_base / train_sft / eval_* / chat          → 卷5
├── inference/    engine（KV cache 生成 + 工具调用状态机）         → 卷7
└── evaluation/   metrics（bpb / 多选 / pass@k）                  → 卷7
```

---

## 教程目录

### 卷 0 · 准备
| 章 | 标题 | 核心问题 |
|---|---|---|
| [00](00-如何使用本教程.md) | 如何使用本教程 | 怎么安排学习顺序 |
| [01](01-环境与全景图.md) | 环境与全景图 | 一次训练的数据流长什么样 |

### 卷 1 · 数据：文本怎么变成 tensor
| 章 | 标题 | 核心问题 |
|---|---|---|
| [02](02-为什么需要分词器.md) | 为什么需要分词器 | 词表大小和参数量、序列长度的三角关系 |
| [03](03-手写一个玩具BPE.md) | 手写一个玩具 BPE | BPE 到底在做什么 |
| [04](04-对话模板与特殊token.md) | 对话模板与特殊 token | SFT 的 loss mask 从哪来 |
| [05](05-文档打包BOS-aligned-bestfit.md) | **文档打包：BOS-aligned best-fit** | 变长文档怎么不浪费地拼成矩形 |
| [06](06-bits-per-byte.md) | bits per byte | 为什么 loss 不可比，bpb 可比 |

### 卷 2 · 模型：tensor 怎么变成 logits
| 章 | 标题 | 核心问题 |
|---|---|---|
| [07](07-先跑通一个最小GPT.md) | 先跑通一个最小 GPT | 最小可用的 GPT 长什么样 |
| [08](08-RMSNorm.md) | RMSNorm | 为什么可以砍掉偏置和减均值 |
| [09](09-RoPE旋转位置编码.md) | **RoPE 旋转位置编码** | 位置信息怎么变成旋转 |
| [10](10-注意力三步曲.md) | 注意力三步曲 | 逐行读懂 self-attention |
| [11](11-SDPA与FlashAttention.md) | SDPA 与 FlashAttention | 为什么快（IO bound 视角） |
| [12](12-QK-Norm与GQA.md) | QK Norm 与 GQA | 稳定训练 + 省 KV cache |
| [13](13-MLP与激活函数.md) | MLP 与激活函数 | ReLU 平方 vs GELU |
| [14](14-残差流与Pre-LN.md) | 残差流与 Pre-LN | 梯度为什么要走旁路 |
| [15](15-权重绑定与logit-softcap.md) | 权重绑定与 logit softcap | 省参数 vs 防溢出 |

### 卷 3 · nanochat 的架构 trick
| 章 | 标题 | 核心问题 |
|---|---|---|
| [16](16-meta-device三步建模型.md) | **meta device 三步建模型** | 一个比 NaN 更阴的性能陷阱 |
| [17](17-resid-lambdas与x0-lambdas.md) | resid_lambdas 与 x0_lambdas | 逐层缩放残差流 |
| [18](18-Value-Embeddings与门控.md) | Value Embeddings 与门控 | 占 43.6% 参数的那一路 |
| [19](19-Smear与Backout.md) | Smear 与 Backout | 26 个参数，以及唯一做减法的 trick |
| [20](20-滑动窗口注意力.md) | 滑动窗口注意力 | 为什么在 T=1024 上省不下时间 |

### 卷 4 · 优化器（最硬核的一卷）
| 章 | 标题 | 核心问题 |
|---|---|---|
| [21](21-从SGD到AdamW.md) | 从 SGD 到 AdamW | 每个超参在干什么；解耦 WD 的可测量含义 |
| [22](22-为什么矩阵参数适合Muon.md) | 为什么矩阵参数适合 Muon | 条件数 2042 → 6.36 |
| [23](23-手写Newton-Schulz正交化.md) | **手写 Newton-Schulz 正交化** | 20 行；5 步落在 `[0.5,1.5]`，8 步饱和 |
| [24](24-Polar-Express.md) | Polar Express | 斜率 3.44 → 8.16；**`ns_steps>5` 静默失效** |
| [25](25-MuonEq-Muon+与NorMuon.md) | MuonEq / Muon+ / NorMuon | **`use_muon_plus` 不是论文的**；因子化省 647 MiB |
| [26](26-谨慎权重衰减.md) | 谨慎权重衰减 | `g` 是正交化后的量；真实配置衰减只占 3.1% |
| [27](27-混合优化器与参数分组.md) | 混合优化器与参数分组 | `ndim==2` 划错一处；形状补偿与论文不一致 |
| [28](28-torch.compile融合与0-D-tensor技巧.md) | torch.compile 融合与 0-D tensor | **一个 if 值 9.9 小时**；静默回落 |

### 卷 5-8 · 引擎与工程（只读卷）
| 章 | 标题 | 核心问题 |
|---|---|---|
| [29](29-训练循环.md) | 训练循环 | **续训要恢复四样东西**；少了 dataloader_state 就偷偷重复训练 |
| [30](30-三条调度曲线.md) | 三条调度曲线 | **momentum warmup 硬编码 400**（lr 只有 40，差 10.3 倍） |
| [31](31-分布式.md) | 分布式 | **为什么 Muon 装不进 ZeRO-2/3**；本章专属判据数为 0 |
| [32](32-推理引擎与KV-cache.md) | 推理引擎与 KV cache | **理论 O(L²)→O(L)，实测只快 1.09 倍**；30% 时间花在反复转权重 |
| [33](33-生成循环与终止控制.md) | 生成循环与终止控制 | 两个 sampler **分布相同但 seed 取样不同** |
| [34](34-对话模板与SFT训练.md) | 对话模板与 SFT | `truncate` 方向选错 → 整批 loss 变 NaN |
| [35](35-评测与指标.md) | 评测与指标 | pass@k 的 `c` 是**总数**不是前缀（写错直接归零） |
| [36](36-checkpoint与续训.md) | checkpoint 与续训 | `inf` 存进 JSON 变 `null`；`.pt` 有原子保护而 `.json` 没有 |
| [37](37-全流程与排错.md) | 全流程与排错 | 完整命令链 + 7 条按症状组织的排查清单 |

**只读卷的三点不同：**

1. **没有「手抄代码」段** —— 改成「读这段代码时注意什么」
2. **没有「★消融实验」段** —— 改成「动手验证」（跑已有判据 + 读代码找陷阱）
3. **风险从「敲错」变成「改坏」** —— 所以判据设计原则是
   「关掉一个东西之后，两个版本必须真的不同」

**这一卷最值得带走的一件事**：**静默失效是最危险的 bug 形状。**
第 24 章的 `ns_steps>5`、第 25 章的 `use_muon_plus`、
第 28 章的 compile 回落、第 32 章的 KV cache —— **全都不报错，只是没效果。**
而手抄卷的 bug（形状对不上）会立刻 assert。

---

## 每章的固定结构

**卷 1-4（手抄卷）：**

```
## 本章目标         学完能回答什么问题
## 前置回顾         链接上一章
## 概念             why，配小例子
## 手抄代码         写进 src/ 哪个文件的哪一段
## 动手验证         1 分钟内能跑完的实验 + 预期输出
## ★ 消融实验       开关关掉，数字变化多少
## 常见坑
## 延伸            nanochat 完整版怎么做的，差异在哪
```

**卷 5-8（只读卷）：** 代码已在 `src/` 里，不需要敲。
两段替换掉：

```
## 读这段代码时注意什么   ★ 逐行读，指出「哪里容易看错」
## 动手验证         跑已有判据 + 读代码找陷阱
```

**★ 消融实验是手抄卷的核心。** 每章的「延伸」告诉你 nanochat 怎么做的，
「消融实验」让你亲手验证它到底值多少。全部做完，你会得到一张表：
每个组件对 val bpb 的贡献。

⚠ **只读卷学的是另一种能力。** 卷 3-4 已经出现了三个
「不报错、只是没效果」的 bug（`ns_steps>5`、`use_muon_plus`、compile 回落）。
手抄卷的错误会立刻 assert，只读卷的错误往往是静默的 ——
所以要练的是**「主动验证改动真的生效了」**，而不是「测试没报错」。

---

## ★ 消融总表

**这是本教程承诺的核心产出物。** 卷 1-4 每章的「★消融实验」都是在往这张表里填一行。

### 怎么跑

```bash
# 1) 先跑中性基线（必须 --no-resume，否则会被 auto-resume 吞掉，bpb 一模一样）
bash script/train_base.sh ablation --model-tag d6_base --no-resume

# 2) 一次只改一个变量，每组换个 tag
bash script/train_base.sh ablation --model-tag d6_norope --no-rope     --no-resume
bash script/train_base.sh ablation --model-tag d6_nosmear --no-smear   --no-resume
bash script/train_base.sh ablation --model-tag d6_adv    --muon-advanced --no-resume

# 3) 汇总
bash scratch/ablation.sh d6_base
```

**每组实测 39 分钟**（RTX 4060 Ti，3,096 步 / 2.03 亿 token）。

### 中性基线长什么样

基线必须保持**中性**（Muon `simple` + trick 全关），
否则测出来的是「A 相对 A+B」，单项贡献被稀释：

```
残差流 trick : 全部关闭（基础架构）
Muon flavor  : simple（5 步 Newton-Schulz）
```

### ★ 开关方向：两类实验方向相反

基线是「**架构全开 + trick 全关 + Muon simple**」。所以：

| 要测什么 | 方向 | 例子 |
|---|---|---|
| **架构特征**的贡献 | 基线里**是开的** → 关掉它 | `--no-rope` |
| **残差流 trick** 的贡献 | 基线里**是关的** → 打开它 | `--use-smear` |

⚠ **写成 `--no-smear` 会得到一个和基线完全一样的配置** ——
开关是 `no_*` 但那个 trick 本来就没开。Δ 会是 0.0000，
看起来像「这个 trick 毫无影响」，实际是**命令打错了**。

### 表模板

| 实验 | 方向 | 开关 | val_bpb | Δ vs base |
|---|---|---|---|---|
| `d6_base` | — | （中性基线） | **1.0615** | （基线）|
| **残差流 trick（基线里是关的 → 打开）** ||||
| `d6_smear` | 开 | `--use-smear` | | |
| `d6_backout` | 开 | `--use-backout` | | |
| `d6_resid` | 开 | `--use-resid-lambdas` | | |
| `d6_x0` | 开 | `--use-x0-lambdas` | | |
| `d6_ve` | 开 | `--use-value-embeds` | | |
| `d6_alltricks` | 开 | `--all-tricks` | | |
| **架构特征（基线里是开的 → 关掉）** ||||
| `d6_norope` | 关 | `--no-rope` | | |
| `d6_noqk` | 关 | `--no-qk-norm` | | |
| `d6_nosoftcap` | 关 | `--no-softcap` | | |
| `d6_tie` | 关 | `--tie-embeddings` | | |
| `d6_layernorm` | 关 | `--norm-type layer` | | |
| `d6_shve` | 关 | `--shared-value-embeds` | | |
| **优化器** ||||
| `d6_adv` | 关 | `--muon-advanced` | | |

**Δ < 0 表示比基线好**（bpb 越低越好）。

### 仓库现状

| | |
|---|---|
| `d6_base` | ✅ **已跑**（2026-10，`val_bpb = 1.0615`，Muon simple + trick 全关）|
| 其余 13 组 | ⬜ **未跑** |

⚠ **表里的每一格「未跑」都是本教程欠你的。**
各章的结论目前都是**幅度分析或外推**，不是端到端实测 ——
比如第 26 章说「谨慎 WD 只占梯度项 3%」、第 30 章说
「momentum warmup 的硬编码 400 占 full 档 19% 训练」，
都还没有对应的消融行。

### 三条纪律

1. **必须用 `ablation` 档**（`smoke` 档 896 步的 bpb 噪声 ±0.02 大于效应本身）。
2. **一次只改一个变量。**
3. **`|Δ| < 0.02` 时重复跑一次**，确认不是初始化随机性。

### 可测开关清单

```
残差流 5 trick  --no-resid-lambdas --no-x0-lambdas --no-value-embeds
                 --no-smear --no-backout              （卷 3）
架构            --no-rope(09) --no-qk-norm(12) --no-softcap(15)
                 --tie-embeddings(15) --norm-type layer(08)
                 --window-pattern(20) --shared-value-embeds(18)
优化器          --muon-advanced(24-26) --muon-weight-decay(26)
```

⚠ **卷 5-8 是只读章，没有消融行** —— 按设计如此。
它们的主题（续训、KV cache、终止控制、指标口径）不是「开关关掉看差多少」
能回答的，而是「读代码时哪里容易看错」。

---

## 六个入口脚本

```bash
bash script/train_base.sh smoke   # 预训练（自动串好 下载数据→训tokenizer→训练）
bash script/eval_base.sh  smoke   # val bpb + 多选题 + 采样
bash script/train_sft.sh  smoke   # 对话微调
bash script/eval_sft.sh   smoke   # 多选准确率 + GSM8K pass@1 + 多轮展示
bash script/chat.sh       smoke   # 交互式聊天
```

档位换成 `debug`（秒级，适合单步调试）或 `ablation`（d6，约 44 分钟，消融专用）。
想训最终的 d24 模型用 `full`（约 41 小时，只训一次）。

---

## 卡住了怎么办

1. **跑测试**：`uv run pytest tests/ -q`。212 个章节判据（另加 10 个判据自检）覆盖了形状、因果性、
   RoPE、KV cache、优化器等最容易出错的地方，挂了就说明你的改动引入了 bug。
2. **看注释**：本项目的代码注释密度很高，尤其是「为什么这么写」的部分，
   基本每个非显然的决定都有解释。
3. **看排错手册**：[第 37 章](37-全流程与排错.md) 的「排错清单」按**症状**组织
   —— NaN / CUDA driver error / bpb 不动 / 两个消融数字一样 / 参数分组 assert 失败 /
   续训后 loss 变大 / 中文乱码，每条都给排查顺序。

> ⚠ **「跑 pytest 全绿」不等于「文档里的数字对」。**
> 精确核对在 `uv run python scratch/audit_docs.py` —— 它用**精确相等**
> 比对章节判据数，新增/删除测试必须同步 4 处（见 [第 37 章](37-全流程与排错.md)）。
