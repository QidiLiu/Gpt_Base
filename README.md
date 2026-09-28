# Gpt_Base

> 从零手抄一个 GPT —— **nanochat 的架构与训练机制** + **nanoGPT 的可读性**

一个用于「通过实践理解 GPT 各个组件原理」的教学型项目。

- 架构对齐 [nanochat](https://github.com/karpathy/nanochat)，可读性对齐 [nanoGPT](https://github.com/karpathy/nanoGPT)
- **`src/` 里是需要你亲手实现的骨架**（每个函数体是 `raise NotImplementedError` + 实现提示）
- 教程在 [`doc/tutorial/`](doc/tutorial/README.md)，代码块折叠，先自己想再看
- 每个设计都配**消融实验**：开关关掉，用数字看到真实贡献
- 章节完成判据就是对应的 pytest，`bash script/progress.sh` 逐章追踪

## 开始

```bash
uv sync
bash script/progress.sh          # 看你现在该做哪一章
```

## 学习路线

| 卷 | 内容 | 手抄 / 只读 |
|---|---|---|
| 0 | 环境与全景图 | 只读 |
| 1 | 数据：分词、对话模板、best-fit 打包、bpb | **手抄** |
| 2 | 模型：RMSNorm、RoPE、注意力、MLP、Pre-LN | **手抄** |
| 3 | nanochat 的架构 trick：meta device / resid_lambdas / value_embeds / smear / 滑窗 | **手抄** |
| 4 | 优化器：AdamW → Muon → Newton-Schulz → Polar Express → 方差缩减 | **手抄** |
| 5-8 | 训练循环、分布式、SFT、推理、收尾 | 只读 |

手抄总量约 2100 行 / 8-11 小时。

## 五个入口脚本

手敲完卷 1-4 之前它们跑不通（依赖你还没实现的部分）。
之后：

```bash
bash script/train_base.sh smoke   # 预训练（自动串好 下载数据→训tokenizer→训练）
bash script/eval_base.sh  smoke   # val bpb + 多选题 + 采样
bash script/train_sft.sh  smoke   # 对话微调
bash script/eval_sft.sh   smoke   # 多选准确率 + GSM8K pass@1 + 多轮展示
bash script/chat.sh       smoke   # 交互式聊天
```

档位：`debug`（秒级）/ `smoke`（2-3 分钟）/ `full`（20-25 分钟）。

## 项目结构

```
src/
├── common/       公共设施：设备/精度/路径/网络镜像 + 单一旋钮 config   → 卷0
├── data/         数据管线：tokenizer / dataset / dataloader / tasks    → 卷1  ✍
├── model/        模型：layers（组件） + gpt（组装与 trick）             → 卷2-3 ✍
├── optim/        优化器：orthogonalize + muon（MuonAdamW）            → 卷4  ✍
├── training/     训练与评测入口：train_base / train_sft / eval_* / chat → 卷5
├── inference/    推理引擎：KV cache + 工具调用状态机                    → 卷7
└── evaluation/   评测：bpb / 多选 loglikelihood / pass@k               → 卷7
script/           6 个入口脚本（train_base / eval_base / train_sft / eval_sft / chat / progress）
doc/tutorial/     教程（7 章已写：卷0 准备 + 卷1 数据；卷 2-8 待补）
scratch/          11 个实验脚本
tests/            51 个测试，每章的完成判据（另加 1 个判据自检文件）
```

## 看答案

```bash
git show solution:src/model/layers.py    # 只看一个文件
git checkout solution                      # 切到完整实现
git checkout main                          # 切回你的版本
git diff --stat                            # 你写了多少
```

## 致谢

- [nanoGPT](https://github.com/karpathy/nanoGPT) — 证明了 300 行能训 GPT-2
- [nanochat](https://github.com/karpathy/nanochat) — 极简全栈 LLM 训练框架，本项目的架构蓝本
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — Muon 与架构 trick 的来源
