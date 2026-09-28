"""
卷 4（优化器）的测试。

这一卷的完成判据：全部通过。
  uv run pytest tests/test_optim.py -v
"""

import pytest
import torch

from optim.orthogonalize import (
    orthogonalize_simple, orthogonalize_advanced,
    orthogonality_error, condition_number, nor_muon_scale,
)
from optim.muon import HyperParams, adamw_step, setup_optimizer, muon_step
from common.config import MuonConfig, OptimConfig

# AdamW 组里允许出现的 2D 参数：嵌入类（分词嵌入 / 反嵌入 / value embedding）
EMBED_PREFIXES = ("lm_head.", "transformer.wte.", "value_embeds.")


# ===========================================================================
# ch21-22：AdamW 一步
# ===========================================================================
def _hp(**kw):
    """HyperParams.__init__ 只登记 key（值恒为 0），赋值必须走 .set()。"""
    h = HyperParams(**{k: 0.0 for k in kw})
    h.set(**kw)
    return h


def test_adamw_step_matches_reference():
    """
    回归测试：adamw_step 的结果必须与「教科书 AdamW」逐元素一致。

    参考实现（PyTorch 官方的等价写法）：
        p -= lr * wd * p                 # 解耦权重衰减
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g^2
        p -= lr * m / (sqrt(v) + eps)   # 注意 PyTorch 的 eps 在 sqrt 外面
    """
    torch.manual_seed(0)
    p0 = torch.randn(64)
    g = torch.randn(64)

    # 被测实现
    p, m, v = p0.clone(), torch.zeros(64), torch.zeros(64)
    for step in range(1, 11):
        hp = _hp(step=step, lr=0.01, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1)
        adamw_step(p, g, m, v, hp)

    # 独立参考：偏差校正必须用「当前步数」，不是固定的 10
    p_ref, m_ref, v_ref = p0.clone(), torch.zeros(64), torch.zeros(64)
    lr, b1, b2, eps, wd = 0.01, 0.9, 0.95, 1e-8, 0.1
    for step in range(1, 11):
        p_ref.mul_(1 - lr * wd)
        m_ref.lerp_(g, 1 - b1)
        v_ref.lerp_(g.square(), 1 - b2)
        denom = (v_ref / (1 - b2 ** step)).sqrt() + eps
        p_ref.add_(m_ref / denom, alpha=-(lr / (1 - b1 ** step)))
    assert torch.allclose(p, p_ref, atol=1e-6), \
        f"adamw_step 与参考不一致，最大误差 {(p - p_ref).abs().max().item()}"


def test_adamw_step_actually_moves_params():
    """冒烟测试：一步之后参数必须真的变了，且没有 NaN。

    ⚠ NaN 守卫不是多余的：曾经这个测试「通过」是因为参数全变成了 NaN，
    而 `NaN != 1.0` 恒为真。根因是 HyperParams 的值没填（构造函数只登记 key），
    导致偏差校正里 `1 - beta**0 == 0` 除零。任何数值型判据都要带这道守卫。
    """
    p = torch.ones(32)
    g = torch.full((32,), 0.1)
    hp = _hp(step=1, lr=0.1, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.0)
    adamw_step(p, g, torch.zeros(32), torch.zeros(32), hp)
    assert not torch.isnan(p).any(), (
        "adamw_step 产生了 NaN —— 多半是超参没填（HyperParams 要用 .set()），"
        "或偏差校正的分母为 0")
    assert (p != 1.0).any(), "参数没有更新"


def test_adamw_bias_correction_matters():
    """
    回归测试：偏差校正是必须的。
    没有它，第 1 步的有效步长会小到几乎为零（因为 v 还是 0）。
    """
    p = torch.ones(8)
    hp = _hp(step=1, lr=0.1, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.0)
    adamw_step(p, torch.full((8,), 1.0), torch.zeros(8), torch.zeros(8), hp)
    # 第 1 步的更新量应该接近 -lr（因为 1-b1^1 = 0.1，1-b2^1 = 0.05，
    # 归一化后 m/denom ≈ 1）
    assert p[0] < 0.95, f"第 1 步移动量应接近 lr=0.1，实际 {1 - p[0].item():.4f}"


# ===========================================================================
# ch23-25：正交化
# ===========================================================================
def test_orthogonalize_improves_conditioning():
    """回归测试：正交化之后条件数必须显著下降（趋近 1）。"""
    torch.manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(128, 4))
    V, _ = torch.linalg.qr(torch.randn(4, 128))
    G = U @ torch.diag(torch.tensor([50.0, 10.0, 2.0, 1.0])) @ V
    before = condition_number(G)
    after = condition_number(orthogonalize_simple(G))
    assert after < before / 5, f"条件数没有改善: {before:.1f} -> {after:.1f}"


def test_orthogonalize_preserves_singular_values_scale():
    """
    回归测试：半正交矩阵的 Frobenius 范数是 √min(m, n)。
    simple 版不完全收敛，所以应该在这个值附近（±20%）。
    """
    torch.manual_seed(0)
    G = torch.randn(64, 128)
    X = orthogonalize_simple(G, steps=20)
    expect = min(64, 128) ** 0.5
    assert abs(X.norm().item() - expect) / expect < 0.25, \
        f"Frobenius 范数应接近 √min(m,n)={expect:.2f}，实际 {X.norm().item():.2f}"


def test_polar_express_beats_newton_schulz():
    """
    回归测试（ch24）：Polar Express 的系数在正交化效果上优于
    经典 Newton-Schulz 固定系数。

    实测（随机矩阵）：正交误差 0.47（classic） vs 0.17（polar express）
    """
    torch.manual_seed(0)
    G = torch.randn(128, 384).float()
    classic = orthogonalize_advanced(G, use_polar_express=False, steps=5)
    polar = orthogonalize_advanced(G, use_polar_express=True, steps=5)
    e_classic = orthogonality_error(classic)
    e_polar = orthogonality_error(polar)
    assert e_polar < e_classic, (
        f"Polar Express 应该更好: classic={e_classic:.4f} polar={e_polar:.4f}")


def test_muon_plus_rescues_low_rank():
    """
    回归测试（ch25，本教程最重要的一条）：
    Newton-Schulz 推不动近零奇异值。

    低秩矩阵的谱里有大量 0，而 0 是 Newton-Schulz 迭代的**不动点**，
    所以无论迭代多少步都救不回来。Muon+ 的重归一化能把整体范数 snap 回去，
    从而把「推不动的方向」整体放大回来。

    实测（rank=8 的 128×192 矩阵，正交误差）：
        simple      3.34
        polar expr  3.73   <- 单独用也救不回来
        + MuonEq    3.68
        完整        0.97   <- 只有 Muon+ 有效
    """
    torch.manual_seed(0)
    G = torch.randn(64, 8) @ torch.randn(8, 192)     # rank = 8
    simple = orthogonalize_advanced(G, use_muon_eq=False, use_muon_plus=False)
    full = orthogonalize_advanced(G, use_muon_eq=False, use_muon_plus=True)
    e_simple, e_full = orthogonality_error(simple), orthogonality_error(full)
    assert e_full < e_simple / 2, (
        f"Muon+ 应该大幅改善低秩矩阵: simple={e_simple:.3f} full={e_full:.3f}")


def test_nor_muon_scale_reduces_neuron_imbalance():
    """
    回归测试（ch25）：NorMuon 的方差缩减应该降低「逐神经元更新量的差异」。

    做法：造一个某些列更新量特别大的梯度，看 NorMuon 缩放后
    各列能量是否变均匀。
    """
    torch.manual_seed(0)
    m, n = 64, 128
    G = torch.randn(m, n)
    G[:, :4] *= 50.0                     # 前 4 列能量特别大

    def imbalance(X):
        col_energy = X.float().square().mean(dim=0)
        return (col_energy.max() / col_energy.min()).item()

    before = imbalance(G)
    state = torch.zeros(m, 1)
    scale = nor_muon_scale(G, state, beta2=0.9, red_dim=-1)
    after = imbalance(G * scale)
    assert after < before, f"逐神经元能量应该更均匀: {before:.1f} -> {after:.1f}"


# ===========================================================================
# ch26：谨慎权重衰减
# ===========================================================================
def test_cautious_wd_only_decays_same_sign():
    """回归测试（ch26）：谨慎 WD 只在「梯度与参数同号」时衰减。

    谨慎 WD 实现在 muon_step 内，且仅在 flavor != "simple" 时启用。
    """
    def run(p0, g, flavor="advanced"):
        p = torch.full((8, 8), float(p0[0]))
        grad = torch.full((8, 8), float(g[0]))
        hp = _hp(step=1, lr=0.1, momentum=0.0, beta2=0.9, wd=0.5)
        buf = torch.zeros(8, 8); sec = torch.zeros(8, 1)
        muon_step(grad, p, buf, sec, hp, MuonConfig(flavor=flavor))
        return p[0, 0].item()

    # 谨慎 WD 的 mask = (g * p) >= 0，即「这一步会把参数往 0 拉」才衰减。
    # 所以：
    #   同号 (p=2, g=+1) -> mask=1，衰减照常发生（比普通 WD 略多）
    #   反号 (p=-2, g=+1) -> mask=0，**不衰减**
    same_c, same_plain = run([2.0], [1.0]), run([2.0], [1.0], flavor="simple")
    opp_c, opp_plain = run([-2.0], [1.0]), run([-2.0], [1.0], flavor="simple")

    # ★ 旧版算了 same_c / same_plain 却**没有断言**，等于半个测试是空转的。
    # 同号：mask 打开，衰减量应 >= 普通 WD
    assert abs(same_c) <= abs(same_plain) + 1e-6, (
        f"同号时谨慎 WD 应该照常衰减（|p| 不应大于普通 WD）: "
        f"{same_c:.4f} vs 普通 {same_plain:.4f}")
    # 反号：mask 关闭，衰减被抑制 -> |p| 应比普通 WD 大
    assert abs(opp_c) > abs(opp_plain), (
        f"反号时谨慎 WD 几乎不衰减: {opp_c:.4f} vs 普通 {opp_plain:.4f}")


# ===========================================================================
# ch27：参数分组
# ===========================================================================
def test_param_grouping_covers_all_params():
    """
    回归测试（ch27）：参数分组必须覆盖模型的**每一个**参数。

    漏掉一个参数是这类代码最容易出的 bug，而且症状隐蔽：
    训练看起来正常，但那个模块一直是随机初始化。
    所以 setup_optimizer 里必须有那条断言。
    """
    from common.config import make_run_config
    from model.gpt import build_model

    cfg = make_run_config("debug", vocab_size=64)
    cfg.model.sequence_len, cfg.model.n_embd = 32, 32
    cfg.model.n_head = cfg.model.n_kv_head = 2
    cfg.model.head_dim = 16
    for f in ("use_resid_lambdas", "use_x0_lambdas", "use_value_embeds",
              "use_smear", "use_backout"):
        setattr(cfg.model, f, True)
    model = build_model(cfg.model, device="cpu")

    opt = setup_optimizer(model, OptimConfig(), batch_lr_scale=1.0)

    n_model = sum(1 for _ in model.parameters())
    n_grouped = sum(len(g["params"]) for g in opt.param_groups)
    assert n_model == n_grouped, (
        f"参数分组不完整：分到 {n_grouped} 个，模型有 {n_model} 个")

    # 嵌入类必须在 AdamW 组，不能在 Muon 组
    names = {id(p): n for n, p in model.named_parameters()}
    for g in opt.param_groups:
        if g["kind"] == "muon":
            for p in g["params"]:
                assert p.ndim == 2, "Muon 组里出现了非 2D 参数"
        if g["kind"] == "adamw":
            for p in g["params"]:
                if p.ndim == 2:
                    assert names[id(p)].startswith(EMBED_PREFIXES), (
                        f"AdamW 组里的 2D 参数只能是嵌入类，实得 {names[id(p)]}")


# ===========================================================================
# ch28：0-D tensor 技巧
# ===========================================================================
def test_hyperparams_hot_update():
    """
    回归测试（ch28）：0-D CPU tensor 的值可以随时改，
    这正是「不触发 torch.compile 重编译」的关键。
    """
    hp = HyperParams(step=0, lr=0.0)
    hp.set(lr=0.1, step=5)
    assert abs(hp["lr"].item() - 0.1) < 1e-6
    assert hp["step"].item() == 5.0
    # 同一个 tensor 对象，值被原地改了（不是换了新对象）
    before = hp["lr"]
    hp.set(lr=0.2)
    assert before is hp["lr"], "应该是原地改值，不是换对象"
    assert abs(hp["lr"].item() - 0.2) < 1e-6


def test_hyperparams_rejects_unregistered():
    """未登记的超参名必须报错，否则打错字会静默失效。"""
    hp = HyperParams(step=0, lr=0.0)
    # ⚠ 必须用 pytest.raises。手写 try/except + raise AssertionError 的写法
    #   在这里是无效的：被测代码抛的正是 AssertionError，手抛的那个会被
    #   同一个 except 吞掉（消息里含 "未登记" -> 永不重抛）。
    #   实测：把 HyperParams.set 改成静默接受未登记超参，本测试照样 PASSED。
    with pytest.raises(AssertionError, match="未登记"):
        hp.set(lr2=0.1)


# ===========================================================================
# ch21-22 的粒度判据（此前与相邻章节共用 -k 判据）
# ===========================================================================
def test_adamw_decouples_weight_decay_from_the_gradient():
    """
    ch21 AdamW 的粒度判据：**解耦**权重衰减。

    经典（SGD-style）L2 正则把 wd 项混进梯度：g' = g + wd*p。
    AdamW 把它从梯度里拿出来，在更新之后单独作用：p -= lr*wd*p。

    为什么重要：混进梯度后，wd 会被 beta1/beta2 的自适应缩放**再乘一遍**，
    实际衰减量变成 wd 的若干倍，无法精确控制。

    验证方式：梯度恒为 0 时，两种写法结果必须不同；
      AdamW 下参数仍按 p *= (1 - lr*wd) 收缩。
    """
    lr, wd = 0.1, 0.5
    p0 = 2.0
    hp = _hp(step=1, lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, wd=wd)

    # 梯度为 0：经典 L2 会把 wd 加进梯度 -> 参数被拉向 0
    # AdamW 也会收缩（因为有独立的 p *= (1-lr*wd)），但幅度不同
    p_adamw = torch.full((4, 4), p0)
    adamw_step(p_adamw, torch.zeros(4, 4),
               torch.zeros(4, 4), torch.zeros(4, 4), hp)
    adamw_val = p_adamw[0, 0].item()
    expected_shrink = p0 * (1 - lr * wd)
    assert abs(adamw_val - expected_shrink) < 1e-5, (
        f"AdamW 的 wd 收缩量不对: {adamw_val:.6f}，"
        f"期望 p*(1-lr*wd) = {expected_shrink:.6f}")

    # 反例：把 wd 混进梯度（经典 L2）会得到明显不同的结果
    g_mixed = torch.full((4, 4), lr * wd * p0)     # g + wd*p 的效果
    p_l2 = torch.full((4, 4), p0)
    adamw_step(p_l2, g_mixed, torch.zeros(4, 4), torch.zeros(4, 4), hp)
    assert abs(p_l2[0, 0].item() - adamw_val) > 1e-4, (
        "把 wd 混进梯度后结果与解耦一致 -> 说明实现里 wd 进了梯度")

    # wd=0 时参数不应因为「衰减」而变化：只有梯度驱动的更新
    hp0 = _hp(step=1, lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.0)
    p_c = torch.full((4, 4), p0)
    adamw_step(p_c, torch.zeros(4, 4), torch.zeros(4, 4), torch.zeros(4, 4), hp0)
    assert abs(p_c[0, 0].item() - p0) < 1e-6, (
        f"wd=0 且梯度为 0 时参数不该变，实际 {p_c[0,0].item():.6f} vs {p0}")
    # 同一个 step 调两次必须得到相同结果（确定性，无隐藏状态）
    g = torch.randn(4, 4)
    r1, r2 = p0, p0
    p1, p2 = torch.full((4, 4), r1), torch.full((4, 4), r2)
    m1, m2 = torch.zeros(4, 4), torch.zeros(4, 4)
    v1, v2 = torch.zeros(4, 4), torch.zeros(4, 4)
    adamw_step(p1, g, m1, v1, hp0)
    adamw_step(p2, g, m2, v2, hp0)
    assert torch.allclose(p1, p2), "相同输入两次调用应完全一致（确定性）"


def test_muon_orthogonalization_beats_raw_gradient_update():
    """
    ch22「为什么矩阵参数适合 Muon」的粒度判据。

    核心主张：矩阵的梯度谱严重不均衡（不同方向尺度差几个数量级），
    直接用 SGD 更新会让「高频方向」主导；正交化把条件数从 O(κ) 压到 O(1)。

    验证：构造一个**各向异性**的梯度矩阵（谱差 100 倍），
      正交化之后所有奇异值都应落在 1 附近，而原始梯度不是。
    """
    torch.manual_seed(0)
    # 各向异性矩阵：对角线从 1 到 100，条件数很大
    U, _ = torch.linalg.qr(torch.randn(64, 64))
    scales = torch.logspace(0, 2, 64)              # 1 .. 100
    G = U @ torch.diag(scales) @ U.T
    before = condition_number(G)
    assert before > 20, f"构造的矩阵条件数应该很大，实际 {before:.1f}"

    O = orthogonalize_simple(G, steps=5)
    after = condition_number(O)
    assert after < before / 5, (
        f"正交化后条件数应大幅下降: {before:.1f} -> {after:.1f}")
    # 正交化后各奇异值应接近 1（半正交矩阵的谱）
    s = torch.linalg.svdvals(O.float())
    assert s.max() / s.min() < 3.0, (
        f"正交化后谱应接近平坦，实测 max/min = {s.max()/s.min():.2f}")
    # 而且总范数应约等于 sqrt(min(m,n))（半正交矩阵的 Frobenius 范数）
    target = min(O.shape[-2:]) ** 0.5
    got = O.norm()
    assert abs(float(got) - target) / target < 0.6, (
        f"正交化后 Frobenius 范数应 ≈ sqrt(min(m,n)) = {target:.1f}，实际 {float(got):.1f}")
