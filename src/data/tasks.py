"""
任务数据集：SFT 混合 与 评测题库。

依赖 HuggingFace datasets（拉取 parquet、切片、打乱都靠它），
网络受限环境下用 HF_ENDPOINT=https://hf-mirror.com 指向镜像。

对应教程卷7（评测）与卷7（SFT 数据混合）。
"""

import re
import random
from dataclasses import dataclass, field

import numpy as np

from common import log0


# ===========================================================================
# 数据集包装
# ===========================================================================
class HubDataset:
    """
    对 datasets.Dataset 的极简替代，只提供我们用到的三个能力：
        len(ds) / ds[i] / ds.shuffle(seed)

    为什么要自己写？
      完整版 datasets 有 30+ 个传递依赖，nanochat 把它整个删了
      （见其 commit「delete datasets dependency bye」）。
      我们只需要这三个方法，几十行就够，还能顺便看清「打乱」到底怎么实现的。
    """

    def __init__(self, table, permutation=None):
        self.table = table
        self.permutation = permutation

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, i):
        j = i if self.permutation is None else int(self.permutation[i])
        return {c: self.table[c][j].as_py() for c in self.table.column_names}

    def shuffle(self, seed: int):
        return HubDataset(self.table, np.random.default_rng(seed).permutation(len(self)))


def load_hub_dataset(repo_id, subset="default", split="train",
                     max_shards: int = None) -> HubDataset:
    """
    极简 load_dataset：调 Hub 的 parquet 导出 API 列出分片 -> 下载 -> 读。
    返回的表全部 concat 到内存（我们的任务数据都很小：
    MMLU auxiliary_train 约 100K 行，GSM8K 约 7K 行）。

    max_shards：只取前 N 个分片。
      SmolTalk 有 9 片共约 2 GB，全下要 4 分钟。
      SFT 只需要其中一小部分数据，smoke 档取 1 片（224 MB）就够了。

    ── 国内网络要绕两个坑 ──────────────────────────────────────
    1) 必须显式带 User-Agent，否则镜像返回 403
       （Python 内置 urllib 的默认 UA 会被镜像拒掉）
    2) 镜像返回的分片 URL 指向 huggingface.co，而那个域名本机不可达，
       必须重写到镜像端点
    """
    import json
    import os
    import requests
    import pyarrow as pa
    import pyarrow.parquet as pq
    from common import HF_ENDPOINT, get_task_dir, download_file

    slug = repo_id.replace("/", "--")
    d = os.path.join(get_task_dir(), slug, subset, split)
    manifest = os.path.join(d, "manifest.json")
    # manifest 最后写，存在即代表下载完成
    if not os.path.exists(manifest):
        os.makedirs(d, exist_ok=True)
        log0(f"下载 {repo_id} [{subset}/{split}] ...")
        url = f"{HF_ENDPOINT}/api/datasets/{repo_id}/parquet/{subset}/{split}"
        resp = requests.get(url, timeout=60,
                            headers={"User-Agent": "python-requests/gpt-base"})
        resp.raise_for_status()
        urls = resp.json()
        names = []
        for i, u in enumerate(urls):
            if max_shards is not None and i >= max_shards:
                break
            # 镜像返回的分片 URL 指向 huggingface.co，而那个域名本机不可达。
            # 必须重写到镜像端点，否则下载会卡死。
            u = u.replace("https://huggingface.co", HF_ENDPOINT)
            name = f"{i:05d}.parquet"
            download_file(u, os.path.join(d, name), desc=f"{repo_id}/{name}")
            names.append(name)
        with open(manifest, "w") as f:
            json.dump(names, f)
    with open(manifest) as f:
        names = json.load(f)
    if max_shards is not None:
        names = names[:max_shards]
    return HubDataset(pa.concat_tables([pq.read_table(os.path.join(d, n)) for n in names]))


# ===========================================================================
# Task 基类
# ===========================================================================
@dataclass
class Task:
    """
    一个任务 = 一组对话 + 各自的评测方式。

    start/stop/step 提供「逻辑切片」：不复制数据，只在取元素时算物理下标。
    好处是切 100 个切片也只花几 KB 内存。
    """
    start: int = 0
    stop: int | None = None
    step: int = 1

    def __post_init__(self):
        assert self.start >= 0
        assert self.stop is None or self.stop >= self.start
        assert self.step >= 1

    @property
    def eval_type(self) -> str:
        """'categorical'（选一个）| 'generative'（生成文本）"""
        raise NotImplementedError

    def num_examples(self) -> int:
        raise NotImplementedError

    def get_example(self, i: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        stop = self.num_examples() if self.stop is None else self.stop
        return (stop - self.start + self.step - 1) // self.step

    def __getitem__(self, i: int) -> dict:
        assert isinstance(i, int), f"索引必须是整数，得到 {type(i)}"
        if i < 0 or i >= len(self):
            raise IndexError(f"索引 {i} 越界（任务长度 {len(self)}）")
        return self.get_example(self.start + i * self.step)

    def evaluate(self, conversation, completion) -> float:
        raise NotImplementedError

    def reward(self, conversation, completion) -> float:
        """RL 用的奖励。默认直接复用 evaluate。"""
        return float(self.evaluate(conversation, completion))


def render_mc(question: str, letters, choices) -> str:
    """
    多选题的统一渲染格式。

    两个容易被忽略但很关键的细节（nanochat 注释里专门强调过）：

    1) **字母放在选项后面**（`- Paris=F` 而不是 `- F: Paris`）
       大模型不在乎，但**小模型明显更擅长「读到最后找字母」**。

    2) **等号和字母之间不能有空格**（`=F` 而不是 `= F`）
       因为 tokenizer 会把 " F" 和 "F" 切成不同的 token。
       assistant 回答时只输出 "F"（不带空格），
       所以 prompt 里也必须是同一个 token "F"，否则模型学到的映射对不上。
       这就是「训练目标和推理格式必须 token 级一致」的具体体现。
    """
    q = f"Multiple Choice question: {question}\n"
    q += "".join(f"- {c}={l}\n" for l, c in zip(letters, choices))
    q += "\nRespond only with the letter of the correct answer."
    return q


# ===========================================================================
# MMLU —— 教多选格式
# ===========================================================================
class MMLU(Task):
    """
    MMLU（多项选择，学科知识）。https://huggingface.co/datasets/cais/mmlu

    在 SFT 里的作用不是「灌输知识」，而是**教会模型多选题的输出格式**：
    看到 "Respond only with the letter" 就直接吐一个字母，不废话。
    """
    LETTERS = "ABCD"

    def __init__(self, subset="all", split="auxiliary_train", shuffle_seed=42, **kw):
        self.subset, self.split, self.shuffle_seed = subset, split, shuffle_seed
        self._ds = None
        super().__init__(**kw)

    @property
    def _data(self):
        if self._ds is None:
            ds = load_hub_dataset("cais/mmlu", self.subset, self.split)
            self._ds = ds.shuffle(self.shuffle_seed) if self.shuffle_seed is not None else ds
        return self._ds

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self._data)

    def get_example(self, i):
        row = self._data[i]
        q = render_mc(row["question"], self.LETTERS, row["choices"])
        return {"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": self.LETTERS[row["answer"]]},
        ], "gold": row["answer"], "choices": row["choices"],
            "question": row["question"]}

    def evaluate(self, conversation, completion) -> int:
        pred = completion.strip()[:1].upper()
        return int(pred == self.LETTERS[conversation["gold"]])


# ===========================================================================
# ARC —— 难度更高的科学多选
# ===========================================================================
class ARC(Task):
    LETTERS = "ABCD"

    def __init__(self, subset="ARC-Easy", split="train", **kw):
        assert subset in ("ARC-Easy", "ARC-Challenge")
        self.subset, self.split = subset, split
        self._ds = load_hub_dataset("allenai/ai2_arc", subset, split).shuffle(42)
        super().__init__(**kw)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self._ds)

    def get_example(self, i):
        row = self._data_row(i)
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        gold = labels.index(row["answerKey"]) if row["answerKey"] in labels else 0
        q = render_mc(row["question"], self.LETTERS, texts)
        return {"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": self.LETTERS[gold]},
        ], "gold": gold, "choices": texts, "question": row["question"]}

    def _data_row(self, i):
        return self._ds[i]

    def evaluate(self, conversation, completion) -> int:
        return int(completion.strip()[:1].upper() == self.LETTERS[conversation["gold"]])


# ===========================================================================
# GSM8K —— 教数学 + 教工具调用
# ===========================================================================
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_gsm_answer(text: str):
    m = GSM_RE.search(text or "")
    if not m:
        return None
    return m.group(1).strip().replace(",", "")


class GSM8K(Task):
    """
    GSM8K（小学数学应用题）。https://huggingface.co/datasets/openai/gsm8k

    这个任务的价值在 SFT 里是双重的：

    1) **教数学推理**
    2) **教工具调用** —— GSM8K 的标准答案里把中间计算写在 `<<12/60=0.2>>` 里。
       我们把它解析成结构化的「python 调用 + python 输出」消息片段，
       相当于用真实数据教模型「什么时候该调计算器、调用长什么样」。
       这是「让小模型会用工具」最省力的数据来源。

    在 RL 里它是奖励信号：答对 1 分，答错 0 分。
    """
    def __init__(self, subset="main", split="train", **kw):
        assert subset in ("main", "socratic")
        self.subset, self.split = subset, split
        self._ds = load_hub_dataset("openai/gsm8k", subset, split).shuffle(42)
        super().__init__(**kw)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self._ds)

    def get_example(self, i):
        row = self._ds[i]
        question, answer = row["question"], row["answer"]
        # 把 <<12/60=0.2>> 拆成 [文本, python调用, python输出, 文本, ...]
        parts, text = [], ""
        for piece in re.split(r"(<<[^>]+>>)", answer):
            if piece.startswith("<<") and piece.endswith(">>"):
                inner = piece[2:-2]
                expr, _, result = inner.rpartition("=")
                if not expr:            # 没有等号就整体当表达式
                    expr, result = inner, ""
                if text:
                    parts.append({"type": "text", "text": text}); text = ""
                parts.append({"type": "python", "text": expr})
                parts.append({"type": "python_output", "text": result})
            else:
                text += piece
        if text:
            parts.append({"type": "text", "text": text})
        return {"messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": parts},
        ], "gold_answer": extract_gsm_answer(answer)}

    def evaluate(self, conversation, completion) -> int:
        return int(extract_gsm_answer(completion) == conversation["gold_answer"])

    def reward(self, conversation, completion) -> float:
        return float(self.evaluate(conversation, completion))


# ===========================================================================
# SmolTalk —— 教日常对话
# ===========================================================================
class SmolTalk(Task):
    """
    SmolTalk（HuggingFaceTB 的高质量多轮对话集），SFT 的主体。

    它提供 MMLU/GSM8K 都没有的东西：**多轮对话的节奏感**
    （用户追问、模型简洁作答、话题切换）。
    混合比例上它占大头，MMLU/GSM8K 只是「格式教学」的小辅料。
    """
    def __init__(self, subset="all", split="train", max_shards=None, **kw):
        self._ds = load_hub_dataset("HuggingFaceTB/smoltalk", subset, split,
                                    max_shards=max_shards).shuffle(42)
        super().__init__(**kw)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self._ds)

    def get_example(self, i):
        row = self._ds[i]
        msgs = []
        for m in row["messages"]:
            role = m["role"]
            content = m["content"]
            # SmolTalk 里 user 也可能是 system 角色，统一映射
            role = "user" if role in ("user", "system") else "assistant"
            if msgs and msgs[-1]["role"] == role:
                # 连续同角色会破坏 user/assistant 交替的约定，合并
                msgs[-1]["content"] += "\n\n" + content
            else:
                msgs.append({"role": role, "content": content})
        if msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()   # SFT 需要以 user 结尾，assistant 补全
        if len(msgs) < 2:
            msgs = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
        return {"messages": msgs}


# ===========================================================================
# 混合
# ===========================================================================
class TaskMixture(Task):
    """
    把多个任务混在一起，**重复传入同一个任务就是上采样**。

    「上采样为什么要靠重复传入」是个很反直觉的设计，但它极其直观：
        TaskMixture([SmolTalk(), MMLU(), MMLU(), MMLU(), GSM8K()])
                        1 份      3 份         4 份
    比例直接由列表里的重复次数决定，不需要引入任何权重参数。

    做法：先算出全部 (任务下标, 任务内下标) 对，用固定种子（42）打散。
    这样不管每个任务多大，混合后的顺序都是确定的、可复现的。
    """
    def __init__(self, tasks: list[Task], **kw):
        super().__init__(**kw)
        self.tasks = tasks
        self.index_map = [(ti, j) for ti, t in enumerate(tasks) for j in range(len(t))]
        random.Random(42).shuffle(self.index_map)

    def num_examples(self):
        return len(self.index_map)

    def get_example(self, i):
        ti, j = self.index_map[i]
        return self.tasks[ti][j]


def default_sft_mixture(mmlu_epochs=3, gsm8k_epochs=4, smoltalk_shards=None):
    """
    SFT 的默认混合（沿用 nanochat 的配比）：

        SmolTalk  1 份    主体：日常多轮对话
        MMLU      3 份    教「输出一个字母」的多选格式（每份约 100K 行）
        GSM8K     4 份    教数学 + 工具调用（每份约 7.5K 行）

    注意 MMLU 的绝对量远大于 GSM8K（300K vs 30K），
    所以「教格式」的信号远强于「教数学」。这是有意的：
    格式比知识更容易学，也更该优先保证。

    smoltalk_shards：SmolTalk 有 9 片共约 2 GB。smoke 档传 1 就够
    （约 40K 条对话，已经远超本项目单卡能消化的量）。
    """
    return TaskMixture([
        SmolTalk(max_shards=smoltalk_shards),
        *([MMLU("all", "auxiliary_train")] * mmlu_epochs),
        *([GSM8K("main", "train")] * gsm8k_epochs),
    ])


def default_sft_validation() -> TaskMixture:
    """
    验证集要**按训练的比例**截断，否则分布不一致，指标会失真。

    训练里 MMLU:GSM8K 的行数比约 10:1，所以验证集也取 MMLU 5200 / GSM8K 420。
    """
    return TaskMixture([
        MMLU("all", "test", stop=5200),
        GSM8K("main", "test", stop=420),
    ])
