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
  配置：4 个残差流 trick + Muon `advanced`（`resid_lambdas` 已实测有害并去掉），
  只训一次，不做对照。

> **`device_batch_size` 的实测阶梯**（trick 全开 5 个 + Muon advanced，T=1024，
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
> ⚠ **上表的测量条件没有记录在仓库里（代码 commit、GPU 时钟、
> 是否有其他进程占用），所以无法复现或对账。**
>
> 2026-10 重测了两轮（`--num-iterations 13`，取 step 11-13 稳态）：
>
> | 轮次 | `tok/s` | 峰值显存 | MFU | 全程 |
> |---|---|---|---|---|
> | 上表（旧记录） | 14,765 | 12.77 GiB | 23.6% | 41.2 h |
> | V=8192（**修 bug 前**，缓存被污染） | 17,816 | 9.54 GiB | 27.68% | 34.1 h |
> | **V=16384（修 bug 后）** | **16,604** | **11.20 GiB** | **26.56%** | **36.6 h** |
>
> ⚠ **第二轮那行最容易误导** —— 它测的不是「`full` 档」，
> 而是一个从未被设计过的配置（d24 的模型配 V=8192）。
> 第三轮与上表只差约 12%（三项一致 → 同一个全局性原因，
> 最可能是 commit `b01200b`，但未实测）。
>
> **三个数并列而不是删掉旧的** —— 没有记录测量条件的数字无法对账，
> 挑一个看起来更权威的删掉另一个才是错的。
> 详见 [第 29 章](doc/tutorial/29-训练循环.md#实测full-档稳态基线)。
>
> 默认取 8：grad_accum 只有 4 的一半（128 vs 256），CPU 侧开销摊得更薄，
> 且 3.2 GiB 余量实测够 SFT / eval_sft 用。要更宽裕就 `--device-batch-size 4`。

---

## ✅ 已修：预设的 `V=16384` 曾经没有真正生效

**2026-10 实测发现，同日修复。** 修复前 `ablation` / `full` 实际跑 **V=8192**。

```
script/_common.sh:      ablation|full → VOCAB=16384      ← 本意
script/train_base.sh:   python -m training.train_tokenizer --vocab-size 16384
src/training/train_tokenizer.py:
    if os.path.exists(ckpt):  log0("tokenizer 已存在，跳过训练")   ← 只判存在，不判大小
```

因果链：某次 `smoke` 运行在 `~/.cache/gpt_base/tokenizer/` 建了一个 8192 的
tokenizer → 后来 `ablation`/`full` 传 `--vocab-size 16384` 时
**文件已存在，被静默复用** → `train_base` 读 `get_vocab_size()` = 8192。

| 量 | 预设（= 文档写的） | 修复前实际跑的 | **修复后** |
|---|---|---|---|
| `vocab_size` | 16384 | **8192** | 16384 |
| `ln(V)` 随机基线 | 9.7041 | **9.0109** | 9.7041 |
| `lm_head`（d24） | 12,582,912 | **6,291,456** | 12,582,912 |
| 12 张 ve 表（d24） | 150,994,944 | **75,497,472** | 150,994,944 |
| `scaling_params`（d24） | ~~195,035,136~~ | 176,160,768 | **182,452,224** |

⚠ 最后一行顺带修掉一个**独立的文档错误**：
`195,035,136` 多算了一个 `wte`，而 `scaling_params()` 明确不含它
（它是查找表，不是矩阵乘）。真实值 **182,452,224**。
（巧合：`195,035,136` 也正好是 V=32768 时的值。）

**修法三处**：

1. `train_tokenizer` 复用缓存前核对词表大小，不同则重建
   （读不出来也重建）。**大小一致时仍然复用** —— 不然每次跑入口脚本
   都要重训几分钟。
2. `report_compression(BPETokenizer.load(tokdir))` 补上 `tokdir` ——
   原来不传会回落到 `get_tokenizer_dir()`，报告的是「别处那个」tokenizer。
3. `train_base.main` 加 `vocab_mismatch_messages()` 告警作为最后一道防线
   （直接调 `train_base` 会绕过 `train_tokenizer`）。

**6 条新判据，全过故障注入**（`tests/test_presets.py` 15 → 21）。
其中 `test_token_bytes_cache_is_regenerated_together_with_tokenizer`
守的是第二阶后果：只换 `tokenizer.pkl` 而留着旧 `token_bytes.pt`
会让 **bpb 静默算错**。

**现在跑 `bash script/train_base.sh ablation` 会自动把缓存重建为 16384。**

根因分析、故障注入记录、以及受影响的章节清单：
[第 37 章](doc/tutorial/37-全流程与排错.md) 与
[第 29 章](doc/tutorial/29-训练循环.md)。

## 📊 消融实测结果（2026-10，全部 14 组 + 4 次基线复跑）

`ablation` 档 d6，每组 **39 分钟**，3,096 步 / 2.03 亿 token / V=16384。

### ⚠ 先说噪声底 —— 它比原先假设的小 46 倍

纪律第 3 条原本写着「|Δ| < 0.02 时重复跑」。**那个 0.02 从来没被测过，
是项目最早那个 commit 里凭经验写下的。** 实测（每组 39 分钟）：

| 跑法 | val_bpb | 说明 |
|---|---|---|
| `d6_base` | 1.0614995 | 形状派生种子 |
| `base_r1` | 1.0616731 | 同配置复跑 |
| `base_r2` | 1.0614084 | 同配置复跑 |
| `base_s7` | 1.0619637 | `--seed 7`，**换初始化** |

```
同种子三次的样本标准差 σ      = 0.000134
换初始化造成的偏移           = 0.000437  （3.2σ）
Δ 的合成噪声 σ_Δ = √2·σ     = 0.000190
=> 显著性阈值取 0.001 ≈ 5.3σ（比实测最大偏移高 2.3 倍）
```

⚠ 顺便发现**纪律第 3 条原本根本无法执行**：`GPT.init_weights` 里有一行
`torch.manual_seed(n_layer*1000 + n_embd)` **按模型形状重新播种**，
把 `compute_init` 的种子覆盖掉了；而 dataloader 按顺序读 row group
（数据集已预打乱），数据顺序也是确定的。
**唯一的随机来源就是那一行** —— 已加 `--seed` 修好（默认 None = 形状派生，
历史结果逐位不变）。

### 消融表（基线 = 上表三次的均值 1.0615）

| 实验 | val_bpb | Δ | σ 倍数 | 判定 |
|---|---|---|---|---|
| `d6_norope`（`--no-rope`）| 1.1145 | **+0.0530** | 279 | 显著有害 |
| `d6_tie`（`--tie-embeddings`）| 1.0882 | **+0.0267** | 141 | 显著有害 |
| `d6_resid`（`--use-resid-lambdas`）| 1.0676 | **+0.0061** | 32 | 显著有害 |
| `d6_noqk`（`--no-qk-norm`）| 1.0665 | **+0.0050** | 26 | 显著有害 |
| `d6_nosoftcap`（`--no-softcap`）| 1.0643 | **+0.0028** | 15 | 显著有害 |
| `d6_layernorm`（`--norm-type layer`）| 1.0629 | **+0.0013** | 7 | 弱但真实 |
| `d6_shve`（`--shared-value-embeds`）| 1.0616 | +0.0001 | **0.5** | ⚠ **真的测不出** |
| `d6_adv`（`--muon-advanced`）| 1.0600 | **−0.0016** | 8 | 弱但真实的改善 |
| `d6_backout`（`--use-backout`）| 1.0591 | **−0.0024** | 13 | 真实改善 |
| `d6_x0`（`--use-x0-lambdas`）| 1.0588 | **−0.0027** | 14 | 真实改善 |
| `d6_smear`（`--use-smear`）| 1.0566 | **−0.0050** | 26 | 真实改善 |
| `d6_ve`（`--use-value-embeds`）| 1.0389 | **−0.0226** | 119 | 强改善 |
| `d6_alltricks`（`--all-tricks`）| 1.0321 | **−0.0294** | 155 | 强改善 |

### ★ 13 项里只有 1 项真的测不出

按旧的 0.02 阈值，**11 项会被误判成「在噪声带内 / 测不出差异」**。
实测阈值下：

1. **只有 `shared-value-embeds`（+0.0001, 0.5σ）真的没有可测影响。**
   顺带说明它省下约 40% 参数却**不损失质量** —— 这正是第 18 章
   「我倾向共享，但没有实测」那句话现在可以删掉了。
2. **所有「在噪声带内」的 trick 其实都是真实效应**：
   smear −0.0050（26σ）、x0 −0.0027（14σ）、backout −0.0024（13σ）。
   它们只是**小**，不是**不存在**。
3. **`resid_lambdas` 确实有害**（+0.0061，32σ）—— **已据此把它从
   `full` 档默认里去掉**。这是本项目唯一一处因实测而改动 `full` 档默认配置的地方。
4. **RMSNorm 比 LayerNorm 好 0.0013**（7σ）—— 真实但很小。
   RMSNorm 的主要价值仍然是**零参数**和**少一次减均值**，不是质量。
5. **Muon advanced 比 simple 好 0.0016**（8σ）—— 真实但很小。
   五个改进合计换来的就是这个量级。第 26 章说「谨慎 WD 只占梯度项 3%」
   与此一致。

### 一个仍然是硬结论的对比

**RoPE 的贡献比 5 个残差流 trick 加起来还多**：去掉它损失 0.0530，
而 5 个 trick 全开只赚回 0.0294。

⚠ **都是 d6 档 / 2 亿 token 的结论，不能外推到 `full` 档 d24。**
第 22 章说过「矩阵越大 Muon 优势越大」，而 Polar Express 修的正是 0 点斜率
—— 效果很可能随规模变化。**在 `full` 档上分辨 0.001 量级的差异，
需要多 seed 重复，那是 36 小时 × N，本项目做不到。**

完整分析见 [教程 README 的消融总表](doc/tutorial/README.md#-消融总表)。

## ✅ 已解的三个悬置决策（2026-10）

| 决策 | 结论 | 是否改变已有训练结果 |
|---|---|---|
| Muon advanced 的 5 个子开关无 CLI 入口 | **补上** `--no-polar-express` / `--no-muon-eq` / `--no-frobenius-snap` / `--no-nor-muon` / `--no-cautious-wd` | ❌ 纯增益 |
| `use_muon_plus` 不是论文的 | **拆成两个**：`use_frobenius_snap`（旧行为，默认开）+ `use_muon_plus`（论文版，默认关）| ❌ 默认逐位不变（实测 max diff = 0.0）|
| 形状补偿 `max(1,m/n)^0.5` ≠ 论文 `√(m/n)` | **只记录不改** | — （改了要重扫 lr）|

**第一个是纯缺口**：`MuonConfig` 的子开关全默认 True，而
`flavor="advanced"` 是 `full` 档默认 —— 也就是**默认路径上有 5 个
既不可见也不可调的旋钮**。现在每一项都能单独消融。

**第二个是一个名字盖着两种差 6 个数量级的行为**：

```
仅正交化（基线）               行方差 5.972e-04
Frobenius snap（旧 use_muon_plus）  行方差 5.663e-04   <- 只改善 5.2%
论文 Muon+（--use-muon-plus）      行方差 1.797e-15   <- 改善 3.3e11 倍
```

改名后默认行为**逐位不变**（拿旧代码对照验证），
所以历史结果仍可复现。论文版默认**关闭** —— 打开它需要重扫 lr，
那是一次研究，不该混在清理里做。

**第三个需要重扫 lr**（宽矩阵 lr 会差 2 倍），同样留给独立实验。

## 📊 本仓库的真实测量结果（2026-10）

**`ablation` 档 d6 完整训过一遍** —— 3,096 步 / 2.03 亿 token / 39m32s
（RTX 4060 Ti 16GB）：

| 量 | 实测 |
|---|---|
| **`val_bpb`** | **1.0615**（训练循环）/ 1.0754（`eval_base` 全验证集）|
| 随机字节基线（256 符号） | 8.0000 |
| 均匀 over 词表基线 `log2(16384)/4.4492` | 3.1466 |
| train loss | 9.7044 → 3.3667 |
| 峰值显存 | 5,692 MiB |
| MFU | 13.74% |
| MMLU / ARC-Easy / ARC-C | 27.0% / 24.0% / 22.7%（**都≈随机**）|
| `scaling_params` | 16,908,288 |

**三点值得注意**：

1. **多选题≈随机不是 bug。** 基座模型没做过 SFT，不知道「要回答一个字母」，
   而评分比的是「选项文本的平均 logprob」。23M 参数 / 2 亿 token
   没有选择题推理能力是预期的。
2. **`ablation` 档 MFU 只有 13.74%**（`full` 档 26.56%）—— d6 的 GEMM 太小，
   喂不饱这张卡。**所以 `ablation` 只适合做消融，性能测试要用 `full`。**
3. **结束时 `val_bpb` 仍在下降**（最后 96 步 −0.0015）—— 3,096 步没跑够。

样本输出是连贯英文但有明显重复退化（小基座模型的典型表现）：

```
'The capital of France is'
    -> ' the largest city in the world. It is the largest city in the world. It is ...'
```

### SFT 也跑过了（`ablation` 档，869 步 / **1m30s**）

| 指标 | 结果 |
|---|---|
| GSM8K pass@1 | **3.8%**（n=40 × 4 采样）|
| MMLU / ARC-E / ARC-C（**生成式**评分）| 23.7% / 24.7% / 29.3% |

**学到 vs 没学到：**

| 学到 | 没学到 |
|---|---|
| 对话模板与 turn 边界（四轮都接上了）| 内容质量（退化重复）|
| **工具调用协议**（`<|python_start|>12+7<|python_end|><|output_start|>19<|output_end|>` —— 结构全对）| 算术（12+7=19）|
| 「什么时候该调工具」| 多轮上下文一致性（「And of Japan?」没接住）|
| | 任何选择题推理 |

⚠ **多选题那两列不能和基座比** —— 基座用 loglikelihood 评分、
SFT 用生成式评分，**ARC-C 那个 +6.6 主要是评分方式变的**。

**`full` 档 d24 仍是外推**（`runs/` 不入版本库，仓库里没有它的存档）。

数字来源与口径：[第 06 章](doc/tutorial/06-bits-per-byte.md)，
复现步骤与排错：[第 37 章](doc/tutorial/37-全流程与排错.md)。

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
doc/tutorial/     教程 37 章（卷0 准备 + 卷1-4 手抄 + 卷5-8 只读）
scratch/          13 个实验脚本（含 measure_noise_floor.py：消融显著性可复算）
tests/            224 个测试，每章的完成判据（另加 15 个判据自检，共 236）
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
