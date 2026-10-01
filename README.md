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

## 六个入口脚本

手敲完卷 1-4 之前它们跑不通（依赖你还没实现的部分）。
之后：

```bash
bash script/train_base.sh smoke   # 预训练（自动串好 下载数据→训tokenizer→训练）
bash script/eval_base.sh  smoke   # val bpb + 多选题 + 采样
bash script/train_sft.sh  smoke   # 对话微调
bash script/eval_sft.sh   smoke   # 多选准确率 + GSM8K pass@1 + 多轮展示
bash script/chat.sh       smoke   # 交互式聊天
```

### 四档规模

| 档位 | 模型 | 步数 | 耗时（RTX 4060 Ti） | 用途 |
|---|---|---|---|---|
| `debug` | d2, 32 维, 1 头 | 20 | **秒级** | 单步调试 |
| `smoke` | d4, 128 维, 4 头 | 896 | **2-3 分钟** | 跑通全流程，每章验证用 |
| `ablation` | d6, 384 维, 6 头 | 3096 | **约 44 分钟** | **消融实验专用**（基线保持中性） |
| `full` | d24, 768 维, 12 头 | 2088 | **约 41 小时** | 12.8 GB 峰值，消融后的最佳组合 |

`ablation` 和 `full` 回答的是**不同问题**，所以超参数取法正好相反：

- **`ablation`（d6）**回答「某个 trick 值多少钱」。基线必须保持中性 ——
  Muon `simple` + 5 个 trick 全关。一旦基线本身开了 trick，测出来的是
  「A 相对 A+B」，单项贡献就被稀释了。
- **`full`（d24）**回答「最终模型能有多好」。所以它开满 nanochat 的生产
  配置：5 个残差流 trick 全开 + Muon `advanced`，只训一次，不做对照。

> **`device_batch_size` 的实测阶梯**（trick 全开 + Muon advanced，T=1024，
> 连跑 3-4 个完整 step 的稳态值）：
>
> | dbs | accum | 峰值显存 | tok/s | MFU | 总耗时 | 稳定性 |
> |---|---|---|---|---|---|---|
> | 2 | 512 | 4.83 GiB | 13,220 | 21.1% | 46.2 h | ✅ |
> | 4 | 256 | 8.85 GiB (55%) | 14,854 | 23.7% | 40.9 h | ✅ |
> | **8（默认）** | 128 | **12.77 GiB (80%)** | 14,765 | 23.6% | **41.2 h** | ✅ |
> | 12 | 85 | 14.68 GiB | 13,898 | — | — | ❌ 第 2 步 CUDA driver error |
> | 16 | — | 18.77 GiB | 780 | — | — | ❌ 超物理显存 |
>
> 三个反直觉的结论：
>
> 1. **加大 batch 对吞吐毫无收益**：dbs 4 → 8 是 14,854 → 14,765（略降）。
>    GPU 在 dbs=4 时已经吃饱，MFU 稳定在 23.7% 是这张卡的真实上限。
>    所以 4 和 8 的取舍是**余量**，不是速度。
> 2. **dbs=12 会崩，而且不是 OOM**。单步测量它占 92%（14.68 GiB）看着
>    完全可用，但真实训练循环第 2 步就抛
>    `CUDA driver error: device not ready`。独立进程重试 3 次都复现。
> 3. **显存必须多步测**：优化器状态、梯度累积的中间张量、cudnn
>    workspace 都在第 2 步才到峰值。单步测量会严重低估，
>    而且会把 `opt.step()` 的耗时摊到单步上，造出「dbs=8 快 1.44×」
>    这种假象（真实的时间分布是 fwd 33% / bwd 67% / 优化器 0.6%）。
>
> 默认取 8：grad_accum 只有 4 的一半（128 vs 256），CPU 侧开销摊得更薄，
> 且 3.2 GiB 余量实测够 SFT / eval_sft 用。要更宽裕就 `--device-batch-size 4`。

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
scratch/          12 个实验脚本
tests/            191 个测试，每章的完成判据（另加 10 个判据自检，共 201）
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
