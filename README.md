# Gpt_Base

> 从零手抄一个 GPT —— **nanochat 的架构与训练机制** + **nanoGPT 的可读性**

一个用于「通过实践理解 GPT 各个组件原理」的教学型项目。

- 代码按功能分包，每个包对应教程的一卷
- 教程全部**内嵌可手抄代码**，读者从零敲进 `src/`
- 每个设计都配**消融实验**：开关关掉，用数字看到真实贡献
- 架构对齐 [nanochat](https://github.com/karpathy/nanochat)，可读性对齐 [nanoGPT](https://github.com/karpathy/nanoGPT)

## 快速开始

```bash
# 1. 装依赖（uv 自动建 .venv，torch 走 cu132）
uv sync

# 2. 跑最小全流程（约 2-3 分钟，含数据下载与 tokenizer 训练）
bash script/train_base.sh smoke

# 3. 评测
bash script/eval_base.sh smoke

# 4. 对话
bash script/chat.sh smoke
```

`MODE` 可选 `debug`（单步调试用）/ `smoke`（2-3 分钟）/ `full`（20-25 分钟）。

## 学习路线

教程在 [`doc/tutorial/`](doc/tutorial/README.md)，从卷 0 环境与全景图开始。

## 项目结构

```
src/
├── common/       公共设施：设备/dtype/路径/网络镜像 + 单一旋钮 config
├── data/         数据管线：tokenizer / dataset / dataloader        → 卷1
├── model/        模型：layers（组件） + gpt（组装与 trick）          → 卷2-3
├── optim/        优化器：Muon（simple / advanced）                  → 卷4
├── training/     训练与评测入口：train_base / train_sft / eval_* / chat
├── inference/    推理引擎：KV cache + prefill
└── evaluation/   评测：bpb / CORE / pass@k
```

## 致谢

- [nanoGPT](https://github.com/karpathy/nanoGPT) — 证明了 300 行能训 GPT-2
- [nanochat](https://github.com/karpathy/nanochat) — 极简全栈 LLM 训练框架，本项目的架构蓝本
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — Muon 与架构 trick 的来源
