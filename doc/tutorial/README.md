# 从零手抄一个 GPT

> **目标**：不是训出一个好模型，而是让你**亲手把 GPT 的每个组件写一遍**，
> 并且用数字看到每个设计到底值多少钱。

本教程假设你从空目录开始，跟着敲。每一章的代码块都可以直接复制，
敲进 `src/` 下对应的文件。

---

## 两条推进路线

| 路线 | 做法 | 适合 |
|---|---|---|
| **A. 跟敲（推荐）** | 每章的「手抄代码」块自己敲进 `src/`，不看现成代码 | 想真正理解，预算 15-25 小时 |
| **B. 通读** | 直接读 `src/` 里的成品代码，本教程当注释读 | 想快速建立全貌，预算 3-4 小时 |

如果你已经跑通 `bash script/train_base.sh smoke`，说明路线 B 的代码已经在你机器上了，
可以直接从「动手验证」小节开始。

---

## 环境自检

```bash
cd ~/Dev/Gpt_Base
uv sync                      # 建 .venv，torch 走 cu132
uv run pytest tests/ -q      # 18 个测试全绿说明环境没问题
bash script/train_base.sh smoke
```

**你的硬件够用吗？** 这个项目是按 16 GB 单卡（RTX 4060 Ti）设计的。
三档规模：

| 档位 | 模型 | 步数 | 耗时 | 峰值显存 |
|---|---|---|---|---|
| `debug` | d2, 32 维, 1 头 | 20 | **秒级** | < 100 MB |
| `smoke` | d4, 128 维, 4 头 | 896 | **2-3 分钟** | 650 MB |
| `full` | d6, 384 维, 6 头 | 3096 | **20-25 分钟** | ~3 GB |

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
| [11](11-为什么用SDPA和FlashAttention.md) | SDPA 与 FlashAttention | 为什么快（IO bound 视角） |
| [12](12-QKNorm与GQA.md) | QK Norm 与 GQA | 稳定训练 + 省 KV cache |
| [13](13-MLP激活函数.md) | MLP 与激活函数 | ReLU 平方 vs GELU |
| [14](14-残差流与Pre-LN.md) | 残差流与 Pre-LN | 梯度为什么要走旁路 |
| [15](15-权重绑定与logit-softcap.md) | 权重绑定与 logit softcap | 省参数 vs 防溢出 |

### 卷 3 · nanochat 的架构 trick
| 章 | 标题 | 核心问题 |
|---|---|---|
| [16](16-meta-device三步建模型.md) | **meta device 三步建模型** | 一个真实的性能陷阱 |
| [17](17-逐层标量resid与x0.md) | resid_lambdas 与 x0_lambdas | 逐层缩放残差流 |
| [18](18-Value-Embeddings与门控.md) | Value Embeddings 与门控 | ResFormer 做了什么 |
| [19](19-Smear与Backout.md) | Smear 与 Backout | 两个「便宜的小把戏」 |
| [20](20-滑动窗口注意力.md) | 滑动窗口注意力 | 原理，以及一个性能悬崖 |

### 卷 4 · 优化器（最硬核的一卷）
| 章 | 标题 | 核心问题 |
|---|---|---|
| [21](21-从SGD到AdamW.md) | 从 SGD 到 AdamW | 每个超参在干什么 |
| [22](22-为什么矩阵参数适合Muon.md) | 为什么矩阵参数适合 Muon | SVD 视角 |
| [23](23-手写Newton-Schulz正交化.md) | **手写 Newton-Schulz 正交化** | 20 行代码，5 步变成正交 |
| [24](24-Polar-Express.md) | Polar Express | 更好的每步系数 |
| [25](25-三个进阶修正.md) | MuonEq / Muon+ / NorMuon | 三个修正各修什么 |
| [26](26-谨慎权重衰减.md) | 谨慎权重衰减 | 只在「往 0 拉」时衰减 |
| [27](27-混合优化器与参数分组.md) | 混合优化器与参数分组 | 谁该用 Muon，谁该用 AdamW |
| [28](28-compile融合与0D-tensor技巧.md) | torch.compile 融合与 0-D tensor 技巧 | 怎么让编译不被超参变化打断 |

### 卷 5-8 · 训练循环、分布式、对齐、收尾
*（第二批交付：代码已在 `src/` 中，教程随第二批补上）*

卷 5 训练循环 · 卷 6 分布式（只读不跑）· 卷 7 推理与 SFT · 卷 8 全流程与排错

---

## 每章的固定结构

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

**★ 消融实验是本教程的核心。** 每章的「延伸」告诉你 nanochat 怎么做的，
「消融实验」让你亲手验证它到底值多少。全部做完，你会得到一张表：
每个组件对 val bpb 的贡献。

---

## 五个入口脚本

```bash
bash script/train_base.sh smoke   # 预训练（自动串好 下载数据→训tokenizer→训练）
bash script/eval_base.sh  smoke   # val bpb + 多选题 + 采样
bash script/train_sft.sh  smoke   # 对话微调
bash script/eval_sft.sh   smoke   # 多选准确率 + GSM8K pass@1 + 多轮展示
bash script/chat.sh       smoke   # 交互式聊天
```

档位换成 `debug`（秒级，适合单步调试）或 `full`（20-25 分钟）。

---

## 卡住了怎么办

1. **跑测试**：`uv run pytest tests/ -q`。18 个测试覆盖了形状、因果性、
   RoPE、KV cache、优化器等最容易出错的地方，挂了就说明你的改动引入了 bug。
2. **看注释**：本项目的代码注释密度很高，尤其是「为什么这么写」的部分，
   基本每个非显然的决定都有解释。
3. **看排错手册**：[第 43 章](43-全流程与排错.md)（第二批交付）。
