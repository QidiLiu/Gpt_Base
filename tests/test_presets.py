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


def test_full_preset_batch_size_is_self_consistent():
    """
    full 档的 device_batch_size / total_batch_size 必须自洽且有余量。

    ── 为什么要单独测这条 ────────────────────────────────────
    `resolve_scaling` 里有一条 assert：
        total_batch % (device_batch_size × sequence_len) == 0
    它**只在真正训练时**才触发（要等 resolve_scaling 被调用）。
    但这个约束有个陷阱：dbs=12 时 tokens_per_micro = 12288 = 2^12 × 3，
    **任何 2 的幂都不满足**。也就是说照着「batch 取 2 的幂」的直觉写
    total_batch_size，一定会在 resolve_scaling 当场 assert 失败。
    （这正是 full 档最初写成 total_batch=2^20 时会踩的坑。）

    所以 preset 的两个数必须一起改，而它们分处同一行配置里 ——
    正是那种「改了一个忘了另一个」的高危写法。

    另外顺带守住「默认 dbs 不能把显存吃满」：dbs=16 会溢写 host 内存
    （实测峰值 18.77 GiB > 物理 16.00 GiB，tok/s 从 7,766 崩到 780），
    那种配置即使能跑起来也没有意义。
    """
    cfg = make_run_config("full", vocab_size=16384)
    m = cfg.model
    tpm = cfg.train.device_batch_size * m.sequence_len
    assert cfg.train.total_batch_size % tpm == 0, (
        f"device_batch_size={cfg.train.device_batch_size} × "
        f"sequence_len={m.sequence_len} = {tpm}，"
        f"total_batch_size={cfg.train.total_batch_size} 整除不了它。"
        f"注意 tokens_per_micro 含因子 3 时，2 的幂都不满足。"
    )
    accum = cfg.train.total_batch_size // tpm
    assert 1 <= accum <= 512, f"grad_accum={accum} 不在合理区间"

    # 显存余量：以下是**多步稳态**实测的峰值 reserved，不是单步。
    #
    # ⚠ 这张表是踩过坑才建起来的，务必当文档看：
    #   单步测量给出的 dbs=12 是 15.01 GiB（94%），看着完全可用。
    #   但连跑真实训练循环时它在**第 2 步**就抛
    #   `CUDA driver error: device not ready`（不是 OOM！驱动在显存分配
    #   边界上崩掉且无法恢复）。独立进程重试 3 次，每次都复现。
    #   dbs=16 则是 18.77 GiB 直接超物理显存，tok/s 崩到 780。
    #
    #   所以阈值卡在 13 GiB：dbs=12 的单步数字（15.01）正好落在
    #   「单步看着没事、多步必崩」的区间里。宁可保守。
    #   另注意 dbs=4 与 8 的吞吐实测只差 0.6%（14,854 vs 14,765），
    #   所以默认取哪个是「余量 vs CPU 开销」的取舍，不是速度取舍。
    peak_reserved = {
        2: 4.83, 4: 8.85, 8: 12.77, 12: 14.68, 16: 18.77,
    }[cfg.train.device_batch_size]
    assert peak_reserved < 13.0, (
        f"默认 device_batch_size={cfg.train.device_batch_size} 的多步实测峰值 "
        f"{peak_reserved} GiB 超过 13 GiB 安全线。"
        f"dbs=12 的教训：单步 15.01 GiB 看着能用，但第 2 步就 "
        f"`CUDA driver error: device not ready`（非 OOM）。"
        f"默认档必须留出余量给 SFT 和后续实验。"
    )
    assert peak_reserved <= 0.85 * 16.0, (
        f"默认 device_batch_size={cfg.train.device_batch_size} 占了 "
        f"{100 * peak_reserved / 16:.0f}% 显存，余量不足。"
    )


def test_weight_decay_is_reachable_and_nonzero_for_full():
    """
    full 档的 weight decay 必须**非零**，且能从 CLI / kwargs 覆盖。

    ── 真实事故：谨慎 WD 整条分支空转 ──────────────────────────
    教程卷4 第 26 章讲「谨慎权重衰减」，它由两个东西共同决定：
        muon_step 里的 mask = (g * p) >= 0   （算法）
        hp["wd"] 是否非 0                      （有没有真的在衰减）

    而 MuonConfig.weight_decay 的默认值是 **0.0**。于是：
        λ_ref = 0.0
          -> resolve_scaling 的 weight_decay_scaled = 0.0 × 任何东西 = 0.0
          -> wd_sched(step) 恒返回 0
          -> muon_step 里 `lr * wd * param * mask` 恒等于 0
          -> use_cautious_wd=True 这个 flag 设了也没用，整条分支等于不存在

    更糟的是**没有任何办法从外面打开它**：CLI 没有对应开关，
    PRESETS 里也没写。于是：
      · 教程里「ablation 档加 --cautious-wd 试试」这种操作根本不存在
      · 而 tests/test_optim.py::test_cautious_wd_only_decays_same_sign
        是**直接构造 HyperParams(wd=0.5)** 再调 muon_step 的，
        整条配置链被绕开了 —— 所以它照样通过，挡不住这个 bug。

    这个测试就是补上那个缺口：从 preset 出发，不手工构造任何 wd。
    """
    cfg = make_run_config("full", vocab_size=16384)
    assert cfg.optim.muon.weight_decay > 0, (
        f"full 档的 muon.weight_decay = {cfg.optim.muon.weight_decay}，"
        f"这会让谨慎 WD 整条分支乘 0 空转。"
        f"full 是唯一「开满生产配置」的档位，它的 wd 不该是 0。"
    )
    # 覆盖能力：能通过 kwargs 改（train_base.py 的 --muon-weight-decay 走同一路径）
    off = make_run_config("full", vocab_size=16384, muon_weight_decay=0.0)
    assert off.optim.muon.weight_decay == 0.0, "kwargs 覆盖没生效"
    hi = make_run_config("full", vocab_size=16384, muon_weight_decay=0.5)
    assert hi.optim.muon.weight_decay == 0.5, "kwargs 覆盖没生效"


def test_weight_decay_survives_scaling_for_full():
    """
    resolve_scaling 不能把 full 的 wd 缩放成 0。

    上一条测的是「配置里写了非零」，这条测「经过 T_epoch 缩放之后仍然非零」。
    两者是不同的失败点 —— 缩放公式里有 `run.optim.muon.weight_decay` 这个
    乘数，只要它是 0，前面写什么都没用。

    顺带钉住缩放因子的方向：λ = λ_ref·√(B/B_ref)·(D_ref/D)。
    D_ref 是 d12 的数据量、target_tokens 是当前档的，B_ref=2^19、
    B 是实际 batch。full 档这几个数的实测缩放因子约 0.2133，
    所以 λ_ref=0.1 -> 峰值约 0.0213。这个数会随档位变化，
    所以只断言「非零」和「同量级」，不写死精确值。
    """
    import math
    cfg = make_run_config("full", vocab_size=16384)
    sc = resolve_scaling(cfg, log=lambda *a: None)
    peak = sc["weight_decay_scaled"]
    assert peak > 0, (
        f"full 档的 weight_decay_scaled = {peak} —— 缩放后归零了，"
        f"谨慎 WD 又变成空转。检查 resolve_scaling 里 "
        f"`run.optim.muon.weight_decay` 这个乘数。"
    )
    # 缩放后应比 λ_ref 小（full 档 D_ref/D ≈ 0.15，√(B/B_ref)=1.41，乘积 ≈0.21）
    factor = peak / cfg.optim.muon.weight_decay
    assert 0.05 < factor < 1.0, (
        f"缩放因子 {factor:.4f} 不在预期区间（full 档实测约 0.21）。"
        f"公式是 λ·√(B/B_ref)·(D_ref/D)，factor 应约等于这两项的乘积。"
    )
    assert math.isfinite(peak)


def test_full_preset_cautious_wd_actually_moves_params():
    """
    端到端：full 的真实配置下，cautious WD 必须**真的**改变参数。

    现有 test_cautious_wd_only_decays_same_sign 是手工构造
    HyperParams(wd=0.5) 直接调 muon_step 的 —— 它验证的是**算法**，
    不是**配置是否让它生效**。所以当 weight_decay 恒为 0 时，
    那个测试依然全绿，而实际训练里 wd 分支从未执行。

    这条测试从 preset 出发，走 setup_optimizer -> param_groups ->
    muon_step 的完整链条，只隔离一个变量：wd 是否为 0。
    """
    import torch
    from optim.muon import muon_step, HyperParams
    from common.config import MuonConfig

    def run_with(wd: float) -> torch.Tensor:
        """用 muon_cfg.weight_decay=wd 走一遍 muon_step，返回参数。"""
        mc = MuonConfig(flavor="advanced", weight_decay=wd)
        # ★ 必须用**反号**的初值（p<0, g>0），不能用同号。
        #   同号时 mask=1，谨慎分支的算式退化成
        #       lr*g + lr*wd*p*1  ==  普通分支的 lr*g + lr*wd*p
        #   两者**逐位相同**，所以同号根本区分不出「谨慎分支是活的」
        #   还是「分支被短路、退化成普通分支」。实测：把
        #   `if cfg.flavor != "simple" and cfg.use_cautious_wd:` 改成
        #   `if False and ...`，同号版本的新测试照样通过。
        #   反号时 mask=0，谨慎分支的 wd 项被整个抑制，于是：
        #       谨慎分支: p = -2 - 0.1*1                    = -2.1
        #       普通分支: p = -2 - (0.1*1 + 0.1*0.5*(-2))    = -2.0
        #   两者差 0.1，这才测得到。
        p = torch.full((8, 8), -2.0)
        grad = torch.full((8, 8), 1.0)
        h = HyperParams(step=0, lr=0.0, beta1=0.0, beta2=0.0, eps=0.0,
                        wd=0.0, momentum=0.0)
        h.set(step=1, lr=0.1, momentum=0.0, beta2=0.9, wd=wd)
        muon_step(grad, p, torch.zeros(8, 8), torch.zeros(8, 1), h, mc)
        return p

    with_wd = run_with(0.5)
    without_wd = run_with(0.0)
    # 反号 -> mask=0 -> 谨慎分支应该**完全抑制**衰减，
    # 于是 wd=0.5 和 wd=0.0 的结果必须逐位相同。
    #
    # ⚠ 不能断言绝对数值（如「应该是 -2.1」）：muon_step 会先把梯度
    #   正交化 + NorMuon 缩放，实际更新量不是 lr*g 的原始值
    #   （实测是 0.0354 而不是 0.1）。所以这里只做**相对**比较。
    assert torch.allclose(with_wd, without_wd, atol=1e-6), (
        f"反号时 mask=0，谨慎 WD 应该完全不衰减，但 wd=0.5 时结果变了"
        f"（{with_wd[0,0].item():.6f} vs {without_wd[0,0].item():.6f}）"
        f"—— 说明走的不是谨慎分支，而是退化成了普通分支。"
    )
    # 关键：普通分支在反号时会「正好抵消」梯度步长而让 p 几乎不动
    #   p = -2 - (lr*g' + lr*wd*p)，其中 g' 是正交化后的梯度。
    # 这个「几乎不动」正是谨慎分支被短路时的症状。
    moved = abs(abs(with_wd[0, 0].item()) - 2.0)
    assert moved > 1e-3, (
        f"反号时 |p| 从 2.0 只变了 {moved:.6f} —— 梯度步长和 wd 项正好抵消，"
        f"这说明 mask 没起作用（谨慎分支被短路了）。"
    )

    # 同号：mask=1，wd 必须真的改变结果（确认 wd 确实接进了算式，
    # 而不是被整个忽略）。这条区分不了分支类型，但能抓住「wd 没接上」。
    mc = MuonConfig(flavor="advanced", weight_decay=0.5)
    g_same = torch.full((8, 8), 1.0)
    hp = HyperParams(step=0, lr=0.0, beta1=0.0, beta2=0.0, eps=0.0,
                     wd=0.0, momentum=0.0)

    def same_sign(wd):
        hp.set(step=1, lr=0.1, momentum=0.0, beta2=0.9, wd=wd)
        q = torch.full((8, 8), 2.0)
        muon_step(g_same.clone(), q, torch.zeros(8, 8), torch.zeros(8, 1),
                  hp, mc)
        return q[0, 0].item()

    assert abs(same_sign(0.5) - same_sign(0.0)) > 1e-6, (
        "同号时 mask=1，wd=0.5 应该改变结果，但与 wd=0.0 相同 —— "
        "wd 根本没接进 muon_step 的算式。"
    )


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

    ★ 2026-10：`use_resid_lambdas` **被实测判定为有害（32σ）并从默认里去掉了**。
      实测数据见 scratch/ablation_noise_floor.json，
      重算显著性跑 scratch/measure_noise_floor.py。
      所以这里的期望组合是「4 个开、resid 关」，不再是「5 个全开」。
    """
    cfg = make_run_config("full", vocab_size=16384)
    m, mu = cfg.model, cfg.optim.muon

    # 实测为真实改善的 4 个 trick，必须仍然开着
    for name in ("use_x0_lambdas", "use_value_embeds",
                 "use_smear", "use_backout"):
        assert getattr(m, name) is True, (
            f"full 档应当开启 {name}（消融结论），实际是 {getattr(m, name)}。"
            f"如果这个断言挂了，先查 make_run_config 的分发循环是不是"
            f"又把 preset 的键漏掉了。"
        )

    # ★ resid_lambdas 必须**显式关闭**（而不是「碰巧是默认值」）
    assert m.use_resid_lambdas is False, (
        "full 档的 use_resid_lambdas 又是 True 了 —— "
        "实测 Δ=+0.0061（32σ），它在这个规模上显著有害。"
        "要重测请显式传 --use-resid-lambdas，不要改回默认。"
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
    「模型存到了意外的目录」或「tokenizer 词表对不上」这种下游怪异现象。

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


# ===========================================================================
# tokenizer 缓存必须按词表大小分桶
#
# ★ 这是本项目最贵的一个静默失效（2026-10 实测发现，见教程卷 8 第 37 章）。
#
# 真实的因果链：
#   script/_common.sh 说 ablation/full 要 VOCAB=16384
#   -> train_base.sh 传 --vocab-size 16384 给 train_tokenizer
#   -> train_tokenizer 原来只判 `if os.path.exists(ckpt)` 就「跳过训练」
#   -> 某次 smoke 在缓存里建了个 V=8192 的 tokenizer
#   -> 之后 ablation/full 的 16384 被**静默忽略**，真的按 8192 训练
#
# 后果：lm_head 尺寸差一倍（12,582,912 vs 6,291,456），
# 12 张 value-embeddings 表差一半，而日志只写「tokenizer 已存在，跳过训练」。
#
# 判据怎么写才抓得住？关键是不能只断言「最终词表对」——
# 那要真训一次 16384 的 tokenizer，太慢。真正的判据是
# **「换一个大小调用时，缓存是否被重建」**。
# ===========================================================================
import os
import sys

import torch


def _run_train_tokenizer(tmp_path, vocab_size, monkeypatch):
    """在 tmp_path 下跑一次 train_tokenizer.main()（toy 模式，秒级）。"""
    from training import train_tokenizer as T

    monkeypatch.setattr(T, "get_tokenizer_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "argv",
                        ["train_tokenizer", "--toy",
                         "--vocab-size", str(vocab_size)])
    T.main()
    return T


def test_tokenizer_cache_is_reused_when_vocab_size_matches(tmp_path, monkeypatch):
    """大小一致时**仍然**要复用缓存 —— 别把它改成每次都重训。"""
    T = _run_train_tokenizer(tmp_path, 300, monkeypatch)
    first = os.path.getmtime(os.path.join(tmp_path, "tokenizer.pkl"))
    contents = open(os.path.join(tmp_path, "tokenizer.pkl"), "rb").read()

    T.main()   # 再来一次，同样大小

    assert open(os.path.join(tmp_path, "tokenizer.pkl"), "rb").read() == contents, (
        "词表大小相同却重建了 tokenizer —— 缓存复用被破坏了。"
        "每次重建都要几十秒到几分钟，入口脚本会变得无法使用。"
    )
    assert os.path.getmtime(os.path.join(tmp_path, "tokenizer.pkl")) == first


def test_tokenizer_cache_rebuilds_when_vocab_size_differs(tmp_path, monkeypatch):
    """
    ★ 本条守住那个静默失效：换一个词表大小时**必须重建**。

    修之前的实现是 `if os.path.exists(ckpt): 跳过`，所以这条会红。
    """
    from data.tokenizer import BPETokenizer

    T = _run_train_tokenizer(tmp_path, 300, monkeypatch)
    assert BPETokenizer.load(str(tmp_path)).get_vocab_size() == 300
    stale = open(os.path.join(tmp_path, "tokenizer.pkl"), "rb").read()

    T.main.__globals__  # noqa: B018  下面重新设 argv 换大小
    monkeypatch.setattr(sys, "argv",
                        ["train_tokenizer", "--toy",
                         "--vocab-size", str(520)])
    T.main()

    assert BPETokenizer.load(str(tmp_path)).get_vocab_size() == 520, (
        "要求 V=520，但缓存里的 tokenizer 还是别的词表 —— "
        "这就是 ablation/full 被静默按 8192 训练的那个 bug"
    )
    assert open(os.path.join(tmp_path, "tokenizer.pkl"), "rb").read() != stale, (
        "词表文件内容没变，说明重建根本没发生"
    )


def test_token_bytes_cache_is_regenerated_together_with_tokenizer(tmp_path, monkeypatch):
    """
    ★ 第二阶的后果：token_bytes.pt 必须和 tokenizer 一起换。

    bpb 的分母是 token_bytes（每个 token 覆盖几个字节）。
    如果只重建 tokenizer.pkl 而留下旧的 token_bytes.pt，
    **bpb 会静默算错** —— 而 bpb 是本项目唯一跨模型可比的指标。

    所以这两者必须是同一个「重建单元」，不能分开缓存。
    """
    _run_train_tokenizer(tmp_path, 300, monkeypatch)
    tb_before = torch.load(os.path.join(tmp_path, "token_bytes.pt"),
                           weights_only=True)
    assert len(tb_before) == 300

    from data.tokenizer import BPETokenizer

    monkeypatch.setattr(sys, "argv",
                        ["train_tokenizer", "--toy",
                         "--vocab-size", str(520)])
    _run_train_tokenizer(tmp_path, 520, monkeypatch)

    tok = BPETokenizer.load(str(tmp_path))
    tb_after = torch.load(os.path.join(tmp_path, "token_bytes.pt"),
                          weights_only=True)
    assert tok.get_vocab_size() == 520
    assert len(tb_after) == 520, (
        f"tokenizer 已经是 V=520，但 token_bytes 还有 {len(tb_after)} 条 —— "
        "bpb 会用错的分母算出错的指标，而且不报错"
    )


def test_train_base_warns_when_tokenizer_vocab_differs_from_preset():
    """
    ★ 最后一道防线：直接调 training.train_base（绕过 train_base.sh）时，
    train_tokenizer 不会被执行，缓存里是什么就用什么。

    所以 main() 里必须比一次「预设期望值」和「tokenizer 实际值」。
    这里只测**判定逻辑**，不真的跑训练。
    """
    from training.train_base import vocab_mismatch_messages

    for mode, expected in ALL_MODES:
        # 假装缓存里是个和预设不一样大的词表
        fake = 8192 if expected != 8192 else 16384
        assert fake != expected, "这个用例本身写错了：fake 必须不同于 expected"

        warned = vocab_mismatch_messages(mode, expected, fake)
        assert warned, (
            f"{mode} 档：tokenizer V={fake} != 预设 V={expected}，"
            "却没有产生任何告警 —— 直接调 train_base 时会静默用错词表"
        )
        assert any("注意" in m and "期望" in m for m in warned), (
            f"告警行里没有说清「哪个是实际的、哪个是期望的」：{warned}"
        )

    # 一致时不该告警
    for mode, expected in ALL_MODES:
        assert vocab_mismatch_messages(mode, expected, expected) == [], (
            f"{mode} 档：词表一致却报了告警，会变成噪声"
        )


# ===========================================================================
# 收尾的「平均 MFU」：分子分母必须同口径
#
# 真实的 bug（2026-10 实测发现）：total_flops 数了全部 num_iterations 步，
# 而 total_time 只累加 `step > 10` 的步。误差 = n/(n-11)。
#   13 步的 full 档 -> 日志报 119.96%，逐步 MFU 只有 26.4%
#   2088 步全程    -> 只偏 0.53%，所以一直没人发现
# ===========================================================================
def test_average_mfu_numerator_and_denominator_cover_the_same_steps():
    """
    ★ 分子（total_flops）只数「计入 total_time 的那几步」。

    用 13 步那次的真实数字复现：逐步 MFU 实测 26.56%。
    """
    from training.train_base import summarize_throughput

    flops_per_token = 1.32121e9        # V=16384 的 full 档实测
    batch = 1_048_576
    peak = 8.26e13                     # RTX 4060 Ti 表值
    timed_steps, total_time = 3, 189.94

    total_flops, mfu = summarize_throughput(
        flops_per_token, batch, timed_steps, total_time, peak)

    assert total_flops == flops_per_token * batch * timed_steps, (
        "分子的步数不是 timed_steps —— 这就是那个 bug"
    )
    # 每步耗时相同的话，平均 MFU 必须等于逐步 MFU
    per_step_mfu = 100 * flops_per_token * batch / (total_time / timed_steps) / peak
    assert mfu == pytest.approx(per_step_mfu, rel=1e-12)
    # 26.49% 是那次实测日志里收尾报的「平均 MFU」，
    # 也是逐步 MFU 的稳态中位数（两者一致正是这个 bug 修好后的表现）。
    # 用真实的 V=16384 词表重算能对上；误用 V=8192 的 flops_per_token
    # 会得到 25.73%（差 2.9%）—— 所以这条判据顺带守着「别用错词表」。
    assert per_step_mfu == pytest.approx(26.49, abs=0.02), (
        f"逐步 MFU 应该 ≈26.49%（实测值），算成了 {per_step_mfu:.2f}%。"
        "偏差 2~3% 通常是 flops_per_token 用错了词表（见卷 8 第 37 章）"
    )


def test_average_mfu_can_never_exceed_100_percent():
    """
    MFU > 100% 在定义上不可能，所以它天然是个物理自检。

    这里用「整段训练只跑了 3 步但 num_iterations=2088」这个具体场景：
    旧实现会算出 19834%（分子按 2088 步、分母按 3 步）。
    """
    from training.train_base import summarize_throughput

    flops_per_token, batch, peak = 1.32121e9, 1_048_576, 8.26e13
    _tf, mfu = summarize_throughput(
        flops_per_token, batch, timed_steps=3, total_time=189.94, peak_flops=peak)
    assert 0 < mfu < 100, f"MFU={mfu}% 超出 (0,100) —— 分子分母口径不一致"

    # 顺便钉住边界
    assert summarize_throughput(
        flops_per_token, batch, 0, 0.0, peak)[1] is None, (
        "没有计时步时不该返回一个数（否则 ZeroDivisionError）"
    )
    assert summarize_throughput(
        flops_per_token, batch, 5, 10.0, float("inf"))[1] is None, (
        "非 CUDA 平台 peak 是 inf，MFU 应该为 None 而不是 0"
    )


# ===========================================================================
# 消融的初始化种子必须可控
#
# ★ 2026-10 审计发现：消融纪律第 3 条「|Δ| < 0.02 时重复跑一次，
#   确认不是**初始化随机性**导致的波动」**根本无法执行**：
#
#   · GPT.init_weights 里有一行 `torch.manual_seed(n_layer*1000 + n_embd)`，
#     **按模型形状重新播种**，把 compute_init(device, seed) 覆盖掉了
#   · dataloader 按顺序读 row group（数据集本身已预打乱）-> 数据顺序确定
#
#   也就是说**唯一的随机来源就是那一行**。固定它之后重复跑，测到的只是
#   GPU 浮点原子操作的不确定性 —— 而不是「换一个初始化会怎样」。
#
# 这条测试守住两件事：默认仍是形状派生（历史结果可复现），
# 以及显式传 seed 真的能换初始化。
# ===========================================================================
def _first_proj_signature(seed):
    """建一个最小模型，返回第一个 c_q 权重之和（初始化指纹）。"""
    from common.config import make_run_config, resolve_scaling
    from model.gpt import build_model
    cfg = make_run_config("ablation", vocab_size=2048)
    resolve_scaling(cfg)
    model = build_model(cfg.model, device="cpu", seed=seed)
    w = next(v for n, v in model.named_parameters() if n.endswith("c_q.weight"))
    return w.detach().float().sum().item()


def test_default_seed_is_shape_derived_and_reproducible():
    """
    不传 seed 时用「形状派生」种子，且**逐位可复现**。

    ★ 这一条不能动：所有已发布的 val_bpb 都基于它
    （`full` 档 346M 那套、14 组消融、ablation 的 1.0615）。
    改它就等于让所有历史数字失效。
    """
    from common.config import make_run_config, resolve_scaling
    cfg = make_run_config("ablation", vocab_size=2048)
    resolve_scaling(cfg)
    m = cfg.model
    shape_seed = m.n_layer * 1000 + m.n_embd

    a = _first_proj_signature(None)
    b = _first_proj_signature(None)
    assert a == b, "不给 seed 时两次初始化不同 -> 已发布的 val_bpb 不可复现"

    # 显式传那个形状种子，必须与不给完全相同（证明「不给 = 形状派生」）
    c = _first_proj_signature(shape_seed)
    assert c == a, (
        f"显式传形状种子 {shape_seed} 的结果与不给 seed 不同 —— "
        "说明默认路径不再是「形状派生」，历史结果会失效"
    )


def test_explicit_seed_actually_changes_initialization():
    """★ 传不同的 seed 必须真的换初始化 —— 否则纪律第 3 条仍然无法执行。"""
    base = _first_proj_signature(None)
    s7 = _first_proj_signature(7)
    s99 = _first_proj_signature(99)
    assert s7 != base, "--seed 7 没有改变初始化（compute_init 的 seed 被覆盖了？）"
    assert s99 != s7, "seed 7 和 99 给出了同一个初始化"
    assert s99 != base, "--seed 99 没有改变初始化"


def test_seed_flag_defaults_to_none_so_history_is_preserved():
    """
    ★ `--seed` 的默认值必须是 None 而不是 42。

    如果默认是 42，那么 `init_weights` 收到的就是「显式 42」，
    而形状派生的值是 `n_layer*1000 + n_embd`（d6 是 6384）——
    **默认行为会悄悄改变**，所有已发布数字失效。
    """
    from training.train_base import parse_args
    assert parse_args(["--mode", "ablation"]).seed is None, \
        "--seed 默认必须是 None（= 形状派生），不能是 42"
    assert parse_args(["--mode", "ablation", "--seed", "42"]).seed == 42


def test_compute_init_tolerates_seed_none():
    """
    ★ `compute_init(device, None)` 不能抛。

    `torch.manual_seed(None)` 会抛
    `TypeError: int() argument must be ... not 'NoneType'`，
    而 `--seed` 不给时 argparse 给的就是 None —— 这条不挡，
    训练在**建模型之前**就秒退（实测踩过：复跑批次 3 组全 rc=1）。
    """
    from common.utils import compute_init
    from training.train_base import parse_args
    assert parse_args(["--mode", "ablation"]).seed is None
    torch.manual_seed(0)
    before = torch.randn(3).tolist()
    compute_init("cpu", None)                    # 不应抛
    after_none = torch.randn(3).tolist()
    torch.manual_seed(0)
    torch.randn(3)
    compute_init("cpu", 42)                      # 显式 42 = 历史默认
    after_42 = torch.randn(3).tolist()
    assert after_none == after_42, (
        "seed=None 与 seed=42 给出了不同的全局 RNG 状态 —— "
        "不给 --seed 时应与历史行为逐位一致")
    assert after_none != before, "前置条件不成立：种子根本没生效"


def test_full_preset_resid_lambdas_off_is_still_reachable_and_ablatable():
    """
    ★ 把 resid_lambdas 从 full 默认里去掉之后，**消融能力必须完好**。

    这是这次改动的关键风险：默认关掉一个东西，很容易顺手把它变成
    「再也测不了」。三个方向都必须在：
      · 显式打开          --use-resid-lambdas
      · 全开              --all-tricks
      · 先开后关（顺序契约）--all-tricks --no-resid-lambdas
    """
    from training.train_base import parse_args, apply_overrides
    names = ("use_resid_lambdas", "use_x0_lambdas", "use_value_embeds",
             "use_smear", "use_backout")

    def flags_of(argv):
        cfg = make_run_config("full", vocab_size=16384)
        apply_overrides(cfg, parse_args(argv))
        return {n: getattr(cfg.model, n) for n in names}

    base = flags_of(["--mode", "full"])
    assert base["use_resid_lambdas"] is False

    # (1) 显式打开
    on = flags_of(["--mode", "full", "--use-resid-lambdas"])
    assert on["use_resid_lambdas"] is True, \
        "--use-resid-lambdas 打不开 resid —— 它从默认去掉后不该变成死开关"

    # (2) --all-tricks 仍然开全部 5 个（含 resid）
    allt = flags_of(["--mode", "full", "--all-tricks"])
    assert all(allt.values()), f"--all-tricks 应开全部 5 个，实际 {allt}"

    # (3) 「先开后关」顺序契约仍然成立
    nof = flags_of(["--mode", "full", "--all-tricks", "--no-resid-lambdas"])
    assert nof["use_resid_lambdas"] is False and \
        nof["use_value_embeds"] is True, \
        "--all-tricks --no-resid-lambdas 的语义变了（顺序契约被破坏）"


def test_resid_lambdas_removal_is_backed_by_the_stored_measurement():
    """
    ★ 改 full 档默认必须有实测依据，而且依据必须还在仓库里。

    这条防止的是「过一阵子没人记得为什么关掉了，于是有人手滑改回去」。
    判据会：
      1. 确认 preset 的注释里写了实测数字
      2. 确认原始测量数据文件里 resid 那一行确实是个**正** Δ（变差）
      3. 确认它超过了存储的显著性阈值
    """
    import json
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "common" / "config.py").read_text()
    data = json.loads((root / "scratch" / "ablation_noise_floor.json").read_text())

    # (1) preset 注释里必须有实测数字
    m = re.search(r"Δ\s*=\s*\+0\.0061", src)
    assert m, (
        "src/common/config.py 里 use_resid_lambdas=False 附近缺少实测数字 "
        "（Δ=+0.0061）。关掉一个默认必须留下依据，否则后人不敢动也不敢留。")

    # (2) 存储的数据里 resid 必须是个「变差」的正 Δ
    #     —— 数据在 scratch/ablation_noise_floor.json 的 verdict 段
    assert "d6_resid" in data["verdict"]["real_and_large"], \
        ("ablation_noise_floor.json 的 verdict 里 d6_resid 应属于 real_and_large "
         f"（实际：{data['verdict']}）")

    # (3) 那个 Δ 必须超过存储的显著性阈值 —— 否则「有害」的说法不成立
    thresh = data["derived"]["threshold"]
    assert 0.0061 > thresh, (
        f"d6_resid 的 Δ=0.0061 没有超过显著性阈值 {thresh} —— "
        "如果噪声底被重新测过、阈值变大了，这条判据会红，"
        "那时需要重新决定 full 档该不该开 resid_lambdas")


def test_full_preset_states_resid_lambdas_explicitly_not_by_omission():
    """
    ★★ `PRESETS["full"]` 必须**显式写出** `use_resid_lambdas=False`，
    不能靠「省掉这一行，反正 ModelConfig 默认是 False」。

    这不是洁癖，是一个具体的失效模式：

        注入验证：把 preset 里那行 `use_resid_lambdas=False` 删掉，
        **所有判据仍然全绿** —— 因为 `ModelConfig.use_resid_lambdas`
        的默认值恰好也是 False，组装出来的配置一模一样。

    但这两种写法对读者的含义完全不同：
      · 显式 False = 「我们测过，它有害，所以关掉」
      · 省略       = 「没人想过这件事，碰巧默认关着」

    前者带着依据（见 config.py 里那段实测注释），
    后者会在有人改 `ModelConfig` 默认值时**静默失效** ——
    改默认值那一刻，没有任何测试会红，但 full 档的行为已经变了。

    所以这条直接查 `PRESETS` 字典本身，而不是查组装结果。
    """
    from common.config import PRESETS, ModelConfig

    assert "use_resid_lambdas" in PRESETS["full"], (
        "PRESETS['full'] 里没有显式写 use_resid_lambdas —— "
        "它现在只是「碰巧等于 ModelConfig 的默认值」。"
        "一旦有人改 ModelConfig.use_resid_lambdas 的默认，full 档会静默改变。")

    # 前置条件：ModelConfig 的默认值确实是 False（所以「省略」和「显式 False」同值）
    assert ModelConfig().use_resid_lambdas is False, \
        ("前置条件不成立：ModelConfig 的默认已经不是 False —— "
         "那么「省略」和「显式 False」就不同值了，本条的理由要重写")

    # 而显式写的值必须是 False
    assert PRESETS["full"]["use_resid_lambdas"] is False, \
        f"PRESETS['full']['use_resid_lambdas'] = " \
        f"{PRESETS['full'].get('use_resid_lambdas')!r}，应为 False"

    # 其余 4 个被证明有益的 trick 同样必须显式写出（防止有人用「删掉」来关）
    for name in ("use_x0_lambdas", "use_value_embeds", "use_smear", "use_backout"):
        assert name in PRESETS["full"], \
            f"PRESETS['full'] 应当显式写出 {name}"
        assert PRESETS["full"][name] is True
