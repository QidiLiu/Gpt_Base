"""
卷 0（配置与单一旋钮）的测试。

  uv run pytest tests/test_presets.py -v
"""

import pytest

from common.config import (
    make_run_config, resolve_scaling, estimate_flops_per_token,
    build_model_config, scaling_params, ModelConfig,
    default_tag, default_depth, PRESETS,
)

# 四档的期望词表大小。debug/smoke 用小词表跑得快，ablation/full 要 16384。
ALL_MODES = [("debug", 8192), ("smoke", 8192),
             ("ablation", 16384), ("full", 16384)]


def test_presets_are_self_consistent():
    """
    四档预设必须自洽：梯度累积步数 >= 1、步数 >= 1。

    曾经踩过：total_batch_size=65536，device_batch_size=24，
    sequence_len=1024 -> 24*1024 = 24576 整除不了 65536。
    resolve_scaling 里的 assert 当场抓住了它。

    ★ 这个循环必须覆盖**全部** PRESETS 键，而不是手抄一份名单。
      真实事故：档位从 3 个增加到 4 个（full 改名 ablation + 新增 d24 的 full），
      这里的手抄名单漏掉了新档位，于是新 full 的整除性从未被验证过。
      手抄的名单和被测对象之间没有任何强制关系 —— 直接遍历 PRESETS。
    """
    assert set(m for m, _ in ALL_MODES) == set(PRESETS), (
        f"ALL_MODES 与 PRESETS 不一致："
        f"PRESETS={sorted(PRESETS)}，ALL_MODES={sorted(m for m, _ in ALL_MODES)}"
    )
    for mode, V in ALL_MODES:
        cfg = make_run_config(mode, vocab_size=V)
        sc = resolve_scaling(cfg, log=lambda *a: None)
        assert sc["grad_accum_steps"] >= 1, f"{mode} 档 grad_accum < 1"
        assert sc["num_iterations"] >= 1, f"{mode} 档步数为 0"
        assert sc["total_batch_size"] > 0
        assert cfg.train.device_batch_size > 0, f"{mode} 档 device_batch_size 必须为正"


def test_preset_tag_matches_depth():
    """
    tag 必须是 `d{depth}` —— tag 就是存档目录名，depth 是唯一旋钮。

    真实事故（两条，都真实发生过）：

    1) `tag="d6"` 配 `depth=24` —— 权重和深度对不上。auto-resume 会在
       `runs/base_checkpoints/d6/` 里翻出一个 6 层的存档，`load_state_dict`
       静默地只加载能匹配的那部分，剩下的保持 meta device 的垃圾内存。
       症状是 loss 突然变成 NaN，没有任何形状不匹配的报错。

    2) `tag` 绑**档位**而不绑深度 —— `--depth 8` 会把 8 层权重写进
       `d4/` 目录，于是同一目录里混着两种深度的存档。

    所以这两者必须由一个不变量钉住：tag 就是 `d{depth}`。
    """
    for mode in PRESETS:
        assert default_tag(mode) == f"d{default_depth(mode)}", (
            f"{mode} 档的 tag={default_tag(mode)!r} 与 "
            f"depth={default_depth(mode)} 不一致。tag 是存档目录名，"
            f"必须是 'd{{depth}}' 形式，否则权重和深度会对不上。"
        )


def test_full_preset_carries_the_best_known_combination():
    """
    full 档 = 消融后的最佳超参数组合，preset 必须真的把它们带进去。

    这条测试守的是「preset 里多写的键会不会被静默忽略」这个坑。
    make_run_config 之前只把 `**overrides` 分发到 ModelConfig/TrainConfig，
    preset 自己多写的键（比如 muon_flavor=、use_smear=）**完全不生效**，
    而且不报错。现在两条路径合并了，但这个失败模式太隐蔽，必须钉住。
    """
    cfg = make_run_config("full", vocab_size=16384)
    m, mu = cfg.model, cfg.optim.muon
    for name in ("use_resid_lambdas", "use_x0_lambdas", "use_value_embeds",
                 "use_smear", "use_backout"):
        assert getattr(m, name) is True, (
            f"full 档应当开启 {name}（消融结论），实际是 {getattr(m, name)}。"
            f"如果这个断言挂了，先查 make_run_config 的分发循环是不是"
            f"又把 preset 的键漏掉了。"
        )
    assert mu.flavor == "advanced", (
        f"full 档应当用 Muon advanced，实际是 {mu.flavor!r}。"
    )
    # 形状必须是对齐后的 d24：aspect_ratio 必须从 ablation 档的 64 改成 32。
    # 沿用 64 会得到 1536 维 / 705M 参数，本机 16GB 放不下。
    assert cfg.aspect_ratio == 32, f"full 档 aspect_ratio 应为 32，得到 {cfg.aspect_ratio}"
    assert m.n_layer == 24 and m.n_embd == 768 and m.n_head == 12, (
        f"full 档形状应为 d24/768维/12头，得到 d{m.n_layer}/{m.n_embd}维/{m.n_head}头"
    )


def test_common_sh_matches_presets():
    """
    script/_common.sh 里的 TAG / SHARDS / VOCAB 必须与 config.py 的 PRESETS 一致。

    为什么需要这个测试：bash 读不到 Python 的 PRESETS，所以两份映射必然
    是同一个事实的两份**副本**。副本会漂移 —— 而且漂移了不报错，只表现为
    「模型存到了意外���目录」或「tokenizer 词表对不上」这种下游怪异现象。

    这是本项目一贯的做法：可证伪的声明就用可证伪的方式守住
    （见 tests/test_judging_soundness.py）。
    """
    import pathlib
    import re

    script = (pathlib.Path(__file__).resolve().parent.parent
              / "script" / "_common.sh").read_text(encoding="utf-8")

    def _case_field(field: str) -> dict:
        """解析 `debug) TAG="d2";  SHARDS=1 ;;` 形式的 case 分支。"""
        out = {}
        for m in re.finditer(r"^\s*(\w+)\)(.*?);;\s*$", script, re.M):
            mode, body = m.group(1), m.group(2)
            fm = re.search(rf'{field}="?([^";]+)"?', body)
            if fm:
                out[mode] = fm.group(1).strip()
        return out

    tags = _case_field("TAG")
    shards = _case_field("SHARDS")
    for mode in PRESETS:
        assert mode in tags, f"_common.sh 的 TAG 分支里没有 {mode} 档"
        assert tags[mode] == default_tag(mode), (
            f"{mode} 档：_common.sh 说 TAG={tags[mode]}，"
            f"config.py 的 PRESETS 说 {default_tag(mode)}"
        )
        assert mode in shards, f"_common.sh 的 SHARDS 分支里没有 {mode} 档"
        assert int(shards[mode]) == PRESETS[mode]["data_shards"], (
            f"{mode} 档：_common.sh 说 SHARDS={shards[mode]}，"
            f"config.py 说 data_shards={PRESETS[mode]['data_shards']}"
        )

    # 词表：debug/smoke 用 8192，ablation/full 用 16384
    vocab_small = set(re.findall(r"^\s*(\w+)\|(\w+)\)\s+VOCAB=8192", script, re.M)[0]) \
        if re.search(r"^\s*(\w+)\|(\w+)\)\s+VOCAB=8192", script, re.M) else set()
    vocab_big = set(re.findall(r"^\s*(\w+)\|(\w+)\)\s+VOCAB=16384", script, re.M)[0]) \
        if re.search(r"^\s*(\w+)\|(\w+)\)\s+VOCAB=16384", script, re.M) else set()
    for mode, V in ALL_MODES:
        want = vocab_small if V == 8192 else vocab_big
        assert mode in want, f"_common.sh 的 VOCAB 分支没覆盖 {mode} 档"


def test_depth_drives_width_linearly():
    """
    单一旋钮的核心保证：n_embd 与 depth 成正比（aspect_ratio 固定）。

    这是 scaling law 能跨 depth 迁移的前提 ——
    只有形状一致，推导出的 batch size / LR / WD 才能通用。
    """
    for depth in (4, 6, 8, 12):
        for aspect in (32, 64):
            cfg = build_model_config(depth, aspect, 64, 1024, 16384)
            base = depth * aspect
            n_embd = ((base + 63) // 64) * 64        # 向上取整到 head_dim
            assert cfg.n_embd == n_embd
            assert cfg.n_embd % cfg.head_dim == 0
            assert cfg.n_head == cfg.n_embd // cfg.head_dim
            assert cfg.n_embd >= base, "n_embd 应该 >= depth*aspect"


def test_scaling_params_excludes_embeddings():
    """
    scaling 用的参数量口径：transformer 矩阵 + lm_head，**不含 wte 嵌入表**。

    为什么？查表没有矩阵乘的 FLOPs。
    （nanochat 的实验结论见其 dev/LOG.md 2026-01-27：
      这个组合能得到最干净的 scaling law。）
    """
    cfg = build_model_config(6, 64, 64, 1024, 16384)
    P = scaling_params(cfg)
    # lm_head 恰好是 n_embd * vocab_size（不含 wte 的那一份）
    assert P >= cfg.n_embd * cfg.vocab_size
    # 加一层矩阵约 6*4*d^2
    per_layer = 6 * 4 * cfg.n_embd ** 2
    assert P < 6 * per_layer + cfg.n_embd * cfg.vocab_size + 1


def test_flops_estimate_is_positive_and_monotonic():
    """更深/更宽的模型，FLOPs/token 必须更大。"""
    small = build_model_config(4, 32, 32, 512, 8192)
    large = build_model_config(8, 64, 64, 1024, 16384)
    assert estimate_flops_per_token(large) > estimate_flops_per_token(small)


def test_window_sizes_respect_pattern():
    """
    window_pattern 的展开规则：
      · 只允许 S 和 L
      · 模式按层平铺（"SSL" -> S,S,L,S,S,L,...）
      · 最后一层强制 L（最靠近输出的层需要看全局）
    """
    cfg = ModelConfig(n_layer=6, sequence_len=2048,
                      window_pattern="SSL")
    sizes = cfg.window_sizes()
    assert len(sizes) == 6
    assert sizes[-1][0] == 2048, "最后一层必须全上下文"
    for i in (0, 1, 3, 4):
        assert sizes[i][0] < 2048, f"第 {i} 层应该是滑窗"
    for i in (2, 5):
        assert sizes[i][0] == 2048, f"第 {i} 层应该是全上下文"
    # 非法字符必须报错
    #
    # ⚠ 这里必须用 pytest.raises，不能手写 try/except + raise AssertionError。
    #   被测代码抛的正是 AssertionError，手抛的那个会被同一个 except 吞掉：
    #       try:
    #           ModelConfig(...).window_sizes()
    #           raise AssertionError("非法 ...")   # 永远到不了 except 之外
    #       except AssertionError as e:
    #           if "非法" not in str(e): raise      # 消息里有"非法" -> 永不重抛
    #   也就是说「校验被删掉」时这个测试照样绿。实测：把 config.py 的 assert
    #   换成静默强制转换，本测试 6 passed。
    with pytest.raises(AssertionError, match="非法"):
        ModelConfig(window_pattern="XY").window_sizes()


def test_unknown_override_raises():
    """拼错配置项名要立刻报错，不能静默忽略。"""
    # 同上：用 pytest.raises，让「没报错」直接变成失败。
    with pytest.raises(ValueError, match="未知的配置项"):
        make_run_config("debug", vocab_size=64, lrr=0.1)


def test_preset_keys_actually_reach_the_config():
    """
    preset 里写的键必须真的生效 —— 这条守的是「静默忽略」这个坑。

    真实事故：make_run_config 原来只把 `**overrides`（调用方传的）分发到
    ModelConfig/TrainConfig，**preset 自己多写的键完全不生效**。
    于是往 PRESETS 里加一行 `use_smear=True` 会毫无反应，而且不报错。
    当时之所以没被发现，是因为那时的 preset 键恰好都显式传给了构造函数。

    现在两条路径合并成 {**preset, **overrides}，这个测试守住它。
    """
    cfg = make_run_config("debug", vocab_size=8192,
                          muon_flavor="advanced", use_smear=True,
                          adamw_embedding_lr=0.7)
    assert cfg.optim.muon.flavor == "advanced"
    assert cfg.model.use_smear is True
    assert cfg.optim.adamw.embedding_lr == 0.7


def test_unknown_optim_prefix_raises():
    """optim 的前缀分发也不能吞掉拼错的键（muon_/adamw_ 后面必须是真的字段）。"""
    with pytest.raises(ValueError, match="未知的配置项"):
        make_run_config("debug", vocab_size=8192, muon_flavour="advanced")
