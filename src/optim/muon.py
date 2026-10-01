"""
MuonAdamW：矩阵参数用 Muon，其余用 AdamW。

▓▓ 这一章要你敲的部分 ▓▓
    adamw_step             卷4 第 21 章（AdamW 本身）
    muon_step              卷4 第 23 章（Muon 的一步）
    HyperParams            卷4 第 28 章（0-D tensor 传参技巧）
    compile_or_eager       卷4 第 28 章（torch.compile 兜底）
    MuonAdamW.step         卷4（AdamW 路径；Muon 路径已给）
    setup_optimizer        卷4 第 27 章（参数分组）
    分布式分支（world_size>1）-> 📖 只读，卷6 专章

────────────────────────────────────────────────────────────────
参数怎么分组？（卷4 第 26-27 章）
────────────────────────────────────────────────────────────────
    Muon  : transformer 里所有 2D 矩阵（c_q/c_k/c_v/c_proj/mlp.*）
    AdamW : token 嵌入 wte、反嵌入 lm_head、value_embeds
            以及所有逐层标量（resid_lambdas 等）

分界线是「形状 + 角色」：
  · 2D 矩阵参与大规模矩阵乘，梯度谱严重不均衡 -> Muon 的正交化有用
  · 嵌入是**查表**，梯度是稀疏的 one-hot 类的信号，正交化会把它毁掉
  · 标量只有 1 个元素，正交化无从谈起
  · 数值稳定性：AdamW 对每个参数独立归一化步长，不会炸；
    Muon 依赖整个矩阵的谱结构，单个元素出问题时无处可躲
"""

import torch
import torch.distributed as dist

from common import COMPUTE_DTYPE
from optim.orthogonalize import (
    orthogonalize_simple, orthogonalize_advanced, nor_muon_scale,
)


# ===========================================================================
# ❗ 1. 0-D CPU tensor 传参技巧（卷4 第 28 章）
# ===========================================================================
# torch.compile 会把「Python 标量」当成常量 bake 进编译产物。
# 于是改变 lr 或 beta2 就会触发重新编译（几秒钟）。
#
# 解法：把所有会变的超参做成 0 维的 **CPU tensor**，
# 传给编译函数后在图里用 fill_() 填值。tensor 的 **形状和 dtype**
# 是编译期常量，**数值不是** —— 所以改超参不重新编译。
#
# 下面是把这套机制封装一下，免得每次都写一堆 self._xxx_t。

class HyperParams:
    """一组可热更新的 0-D CPU tensor。"""

    def __init__(self, **kwargs):
        # ❗ 一个 for 循环，给每个 key 建一个 torch.tensor(0.0, float32, "cpu")
        raise NotImplementedError(
            "待实现：HyperParams.__init__ ——\n"
            "  self._t = {k: torch.tensor(0.0, dtype=torch.float32, device='cpu')\n"
            "            for k in kwargs}")

    def set(self, **kwargs):
        """填值。assert 传入的 key 是登记过的。"""
        # ❗ 一个 for 循环 + assert
        raise NotImplementedError(
            "待实现：HyperParams.set ——\n"
            "  for k, v in kwargs.items():\n"
            "      assert k in self._t, f'未登记的超参: {k}'\n"
            "      self._t[k].fill_(float(v))")

    def __getitem__(self, k):
        return self._t[k]


# ===========================================================================
# ❗ 2. AdamW 融合步（卷4 第 21 章）
# ===========================================================================
def adamw_step(p, grad, exp_avg, exp_avg_sq, hp):
    """
    一个 AdamW 更新步，全部融合在一个函数里。

        weight decay (解耦，先做) -> 动量更新 -> 偏差校正 -> 参数更新

    ── 为什么整段都在 fp32 里做？───────────────────────────
    我们把 wte / value_embeds 直接存成 COMPUTE_DTYPE（bf16）省显存。
    但 bf16 只有 8 位尾数，算 `1 - beta2`（beta2=0.999 时等于 0.001）
    会直接下溢成 0，动量就再也不衰减了。所以中途必须转 fp32。

    ── 为什么用 lerp_ 而不是 exp_avg.mul_(b1).add_(g, alpha=1-b1)？
    lerp_ 是单次融合操作，在 torch.compile 下明显更快。
    语义：out = a + (b - a) * w
    所以 exp_avg.lerp_(grad, w=1-beta1) 等价于 a*(beta1) + grad*(1-beta1)

    ── 注意优化器状态不在 state_dict 里！────────────────────
    这是本实现的一个刻意的简化：state_dict 只存参数，AdamW 的
    exp_avg / exp_avg_sq 和 Muon 的 momentum_buffer 全部在内存里，
    **不参与 checkpoint**。后果是断点续训会丢失优化器动量，
    头几十步会有一段小抖动。对本项目（几百到几千步）可接受。

    ── 你要写的 ──
    1) p32, g32 = p.float(), grad.float()
       m32, v32 = exp_avg.float(), exp_avg_sq.float()
    2) 解耦权重衰减：p32.mul_(1 - lr * wd)      ← 注意用的是 hp 里的 tensor
    3) m32.lerp_(g32, 1 - hp['beta1'])
       v32.lerp_(g32.square(), 1 - hp['beta2'])
    4) 偏差校正：
       denom = (v32 / (1 - hp['beta2']**hp['step'])).sqrt() + hp['eps']
    5) p32.add_(m32 / denom, alpha = -(hp['lr'] / (1 - hp['beta1']**hp['step'])))
    6) 写回：p.copy_(p32); exp_avg.copy_(m32); exp_avg_sq.copy_(v32)
    """
    raise NotImplementedError(
        "待实现：adamw_step —— 六步，见 docstring\n"
        "  参考实现：git show solution:src/optim/muon.py")


# ===========================================================================
# ❗ 3. Muon 融合步（卷4 第 23 章）
# ===========================================================================
def muon_step(stacked_grad, stacked_param, momentum_buf, second_moment_buf,
              hp, cfg, use_compiled_ortho=True):
    """
    一步 Muon 更新，作用在「堆叠起来」的参数上。

    为什么参数要 stack？
      同一个 Muon group 里的参数形状必须相同（这样才能一次通信、
      一次编译）。把它们 stack 成 (K, m, n) 后，
      正交化可以在 K 个矩阵上**同时**做（batch 维），大幅提高 GPU 利用率。
      小模型尤其需要这个 —— 单独处理一个 128×128 矩阵会让 GPU 闲着。

    参数
      stacked_grad    (K, m, n)  堆叠的梯度
      stacked_param   (K, m, n)  堆叠的参数
      momentum_buf    (K, m, n)  Nesterov 动量
      second_moment_buf          因子化二阶矩，(K,m,1) 或 (K,1,n)

    ── 五个阶段 ────────────────────────────────────────────
    1) Nesterov 动量
       g = lerp(grad, momentum, μ)  而不是传统 SGD 的 lerp(momentum, grad, μ)。
       区别在于用的是「前瞻」后的梯度方向，效果更好。
       写法：momentum_buf.lerp_(stacked_grad, 1 - hp['momentum'])
             g = stacked_grad.lerp_(momentum_buf, hp['momentum'])
       ⚠ 注意第二步是**原地**修改 stacked_grad 的，所以 g 和 stacked_grad
         是同一个对象。这是有意的（省一次分配），但要知道。

    2) 降到 bf16 做正交化（矩阵乘要吃 tensor core）
       fp16 不行：指数范围太小，正交化过程中的中间值容易溢出。

    3) 正交化
       cfg.flavor == "simple" -> orthogonalize_simple(X, cfg.ns_steps)
       否则                    -> orthogonalize_advanced(X, cfg.ns_steps,
                                     use_muon_eq=, use_muon_plus=, use_polar_express=)
       最后 cast 回 stacked_param.dtype。

    4) 方差缩减（仅 advanced 且 use_nor_muon）
       red_dim = -1 if m >= n else -2
       scale = nor_muon_scale(g, second_moment_buf, hp['beta2'], red_dim)
       g = g * scale

    5) 权重衰减 + 参数更新（★ 谨慎权重衰减，卷4 第 26 章）

       普通 WD：stacked_param.sub_(lr * g + lr * wd * stacked_param)
       谨慎 WD：只在「梯度与参数同号」时衰减
                mask = (g * stacked_param) >= 0
       听起来是废话，但 Muon 的梯度是正交化过的，
       这个符号一致性判据能把 WD 的伤害限制住：
       「同号」意味着这一步会把参数往 0 拉（因为更新是 -lr*g，
       g 与 p 同号 -> p 减小 -> 靠近 0），这时候才衰减。
       反号时更新把参数推离 0，衰减只会阻碍学习。

    ⚠ use_compiled_ortho 这个参数是给「把正交化单独编译」预留的。
      本项目默认 False（直接调用），因为 advanced 版图太大，
      fullgraph=True 容易编译失败。simple 版才值得编译。

    ★★★ 第 5 步有个坑，写之前务必读 ★★★
      谨慎 WD 的分支条件**不能**写成：
          if ... and hp["wd"] != 0:
      `hp["wd"]` 是一个 0-D **tensor**。拿它做 Python 的 if 判断，
      在 torch.compile(fullgraph=True) 下就是 dynamo 眼里的
      "Data-dependent branching" —— 直接拒绝编译，于是整个 muon_step
      **静默回落到 eager**。实测代价（d24 / trick 全开）：
          带这个判断   862 ms/micro  ->  full 档 55.3 小时
          去掉这个判断 710 ms/micro  ->  full 档 45.4 小时
      而且它**不报错**，只在日志里留一行「回落到 eager」，极易漏看。

      正确写法：只判断 cfg（静态的 Python 值），无条件算 mask：
          if cfg.flavor != "simple" and cfg.use_cautious_wd:
              mask = (g * stacked_param) >= 0
              stacked_param.sub_(lr * g + lr * hp["wd"] * stacked_param * mask)
      wd == 0 时那一项是 0*param*mask == 0，数值上完全等价。

      一般性教训：凡是「tensor 参与 Python 的 if / while / and / or」，
      在 fullgraph=True 下都会炸。超参要参与控制流，先把它 .item() 取出来
      （但那会重新引入重编译 —— 所以更好的办法是像这里一样改掉控制流本身）。
    """
    raise NotImplementedError(
        "待实现：muon_step —— 五个阶段，见 docstring\n"
        "  参考实现：git show solution:src/optim/muon.py\n"
        "  验证：uv run pytest -k optim -v")


# ===========================================================================
# ❗ 4. torch.compile 的兜底（卷4 第 28 章）
# ===========================================================================
_COMPILE_FAILED = False   # 全局只警告一次，避免刷屏


def compile_or_eager(fn, probe_args, enabled=True):
    """
    尝试把 fn 编译成 torch.compile 版本；探测失败就返回原函数。

    为什么需要这个兜底？
      torch.compile 不是编译期检查，而是**首次调用时**才真正编译。
      fullgraph=True 模式下如果图里有编译器不支持的算子，
      会在训练跑到一半时炸掉。训练到一半炸掉是最难受的失败方式，
      所以这里在第一次经过该路径时用真实形状的张量试跑一次，
      失败就永久切回 eager（慢一点，但至少能跑完）。

    probe_args 为 None 时直接返回原函数（eager）。

    ── 你要写的 ──
    1) global _COMPILE_FAILED；if 不 enabled / probe_args is None / _COMPILE_FAILED: return fn
    2) try: compiled = torch.compile(fn, dynamic=False, fullgraph=True)
            compiled(*probe_args)          ← 立刻试跑一次，验证能编译
            return compiled
       except Exception as e:
            _COMPILE_FAILED = True          ← 置位，后续不再尝试
            log0 一句友好的提示（怎么修：sudo apt install gcc）
            return fn

    验证：在本机（没装 gcc）跑一次训练，日志里应该出现
         「[optim] torch.compile 不可用（...），回落到 eager」
    """
    raise NotImplementedError(
        "待实现：compile_or_eager —— 见 docstring")


# ===========================================================================
# 优化器类
# ===========================================================================
class MuonAdamW(torch.optim.Optimizer):
    """
    组合优化器。param_groups 里每组带一个 'kind' 字段：
        kind='muon'  -> 走 Muon 路径
        kind='adamw' -> 走 AdamW 路径

    同一套类同时处理单卡和分布式。本项目只有单卡，
    所以 world_size 恒为 1，所有 dist 调用都被跳过。
    但代码保留完整的分布式形态，因为「为什么 nanochat 不用 DDP」
    这个问题只有读到那段代码才有答案（教程卷6）。

    下面三个方法里：
      __init__            🔶 已给
      _init_adamw_state   🔶 已给
      _init_muon_state    🔶 已给
      step                ❗ AdamW 路径要你敲（Muon 路径已给）
    """

    def __init__(self, param_groups, muon_cfg, use_compile=True):
        super().__init__(param_groups, defaults={})
        self.muon_cfg = muon_cfg
        self.use_compile = use_compile
        self._hp = HyperParams(step=0, lr=0.0, beta1=0.0, beta2=0.0,
                               eps=0.0, wd=0.0, momentum=0.0)
        self._fn_adamw = adamw_step
        self._fn_muon = muon_step
        self._probed_adamw = False
        self._probed_muon = False

    def _init_adamw_state(self, p):
        st = self.state[p]
        if "exp_avg" not in st:
            st["exp_avg"] = torch.zeros_like(p)
            st["exp_avg_sq"] = torch.zeros_like(p)
        st["step"] = st.get("step", 0) + 1
        return st

    def _init_muon_state(self, params):
        """Muon 状态挂在 group 的第一个参数上（同组参数共享一个状态容器）。"""
        p0 = params[0]
        st = self.state[p0]
        if "momentum_buffer" not in st:
            k = len(params)
            st["momentum_buffer"] = torch.zeros((k, *p0.shape),
                                                dtype=p0.dtype, device=p0.device)
            m, n = p0.shape[-2], p0.shape[-1]
            # 二阶矩只存「被规约维度」上的均值，不存完整矩阵
            shape = (k, m, 1) if m >= n else (k, 1, n)
            st["second_moment_buffer"] = torch.zeros(shape, dtype=p0.dtype, device=p0.device)
        st["step"] = st.get("step", 0) + 1
        return st

    @torch.no_grad()
    def step(self):
        # rank / world 在单卡下没人用，但**你要写的 AdamW 路径会用**：
        # 分布式分支要按 world 决定是 all_reduce 还是 reduce_scatter。
        # 这里先算好，提示里直接引用。lint 的 F841 是预期内的。
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
        else:
            rank, world = 0, 1   # ← 单卡退化点：下面所有 world>1 的分支都不执行

        for group in self.param_groups:
            kind = group["kind"]

            # ==================== AdamW 路径（你要敲）====================
            if kind == "adamw":
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self._init_adamw_state(p)
                    # ① 填超参（注意 betas 是二元组）
                    # ② 首次经过这条路径时探测编译：compile_or_eager(...)
                    # ③ world == 1 时直接 self._fn_adamw(p, p.grad, st[...], st[...], self._hp)
                    #
                    # ④ world > 1 的分支 📖 只读（卷6）：
                    #    · p.numel() < 1024（小参数/标量）：all_reduce 梯度，状态保持复制
                    #    · 否则（reduce_scatter_tensor + all_gather_into_tensor）
                    #      效果：优化器状态天然按 world_size 分片
                    #      前提：p.shape[0] % world == 0
                    raise NotImplementedError(
                        "待实现：MuonAdamW.step 的 AdamW 路径\n"
                        "  ① self._hp.set(step=st['step'], lr=group['lr'],\n"
                        "                 beta1=group['betas'][0], beta2=group['betas'][1],\n"
                        "                 eps=group['eps'], wd=group['weight_decay'])\n"
                        "  ② if not self._probed_adamw:\n"
                        "         self._probed_adamw = True\n"
                        "         self._fn_adamw = compile_or_eager(\n"
                        "             adamw_step, (p, p.grad, st['exp_avg'], st['exp_avg_sq'], self._hp),\n"
                        "             self.use_compile)\n"
                        "  ③ if world == 1:\n"
                        "         self._fn_adamw(p, p.grad, st['exp_avg'], st['exp_avg_sq'], self._hp)\n"
                        "         continue\n"
                        "  ④ 分布式分支见 tutorial/卷6\n"
                        "参考实现：git show solution:src/optim/muon.py")

            # ==================== Muon 路径（🔶 已给，读懂即可）====================
            elif kind == "muon":
                params = group["params"]
                st = self._init_muon_state(params)

                # 堆叠：同组参数形状相同，stack 后正交化能一次算 K 个矩阵，
                # 大幅提高 GPU 利用率（小模型尤其需要）
                stacked_grad = torch.stack([p.grad for p in params])
                stacked_param = torch.stack(params)

                # 形状修正：高矩阵（m>n）乘 √(m/n)。
                # 原因：正交化后每个元素量级是 1/√n，高矩阵列数少，
                # 不补偿的话每步实际位移偏小，等效于更小的学习率。
                m, n = params[0].shape[-2], params[0].shape[-1]
                self._hp.set(
                    lr=group["lr"] * max(1.0, m / n) ** 0.5,
                    momentum=group["momentum"],
                    wd=group["weight_decay"],
                    beta2=group.get("beta2", 0.9),
                )
                if not self._probed_muon:
                    self._probed_muon = True
                    self._fn_muon = compile_or_eager(
                        muon_step,
                        (stacked_grad, stacked_param, st["momentum_buffer"],
                         st["second_moment_buffer"], self._hp, self.muon_cfg),
                        self.use_compile)
                self._fn_muon(stacked_grad, stacked_param, st["momentum_buffer"],
                              st["second_moment_buffer"], self._hp, self.muon_cfg)

                # 把更新后的堆叠参数写回各个独立参数
                torch._foreach_copy_(params, list(stacked_param.unbind(0)))

            else:
                raise ValueError(f"未知的 kind: {kind}")


# ===========================================================================
# ❗ 5. 参数分组（卷4 第 26-27 章）
# ===========================================================================
def setup_optimizer(model, optim_cfg, batch_lr_scale: float = 1.0) -> MuonAdamW:
    """
    把模型参数分到 Muon / AdamW 两组，并套上 LR 缩放。

    ── 为什么 LR 要随模型维度缩放？───────────────────────────
    dmodel_lr_scale = (d_model / 768)^(-1/2)
    这就是 muP（maximal update parameterization）的思路：
    宽度变大时，单个参数的「分摊职责」变小，LR 要相应调小，
    否则有效步长会随宽度放大，训练发散。
    768 是当初调参时的参考宽度，除以它是为了让超参在任意深度下通用。

    ── 你要写的 ──
    1) 遍历 model.named_parameters()，按名字分流到：
         matrix_params（2D 且不是嵌入/lm_head）
         adam_groups["wte"] / ["ve"] / ["lm_head"] / ["scalar"]
    2) ★ 加一条断言：分到的参数个数必须等于模型总参数个数。
       漏掉一个参数是这类代码最容易出的 bug，而症状是「训练看起来正常，
       但某个模块一直是随机初始化」。断言能当场抓住。
       dmodel_lr_scale = (d / optim_cfg.dmodel_lr_ref) ** -0.5
    3) AdamW 组（4 个）：
         lm_head:  lr = unembedding_lr × batch_lr_scale × dmodel_lr_scale
                   betas=ac.betas_unembedding, wd=ac.weight_decay_unembedding
         wte:      lr = embedding_lr    × batch_lr_scale × dmodel_lr_scale
                   betas=ac.betas_embedding,  wd=ac.weight_decay_embedding
         ve:       lr = embedding_lr × 0.5 × ...（value_embeds 用更小的 lr）
         scalar:   lr = 0.5 × batch_lr_scale, betas=(0.8, 0.95), wd=0
    4) Muon 组：**按形状分组**（必须同形才能 stack）
       for shape in sorted({p.shape for p in matrix_params}):
           kind='muon', params=[同形状的那些], lr=mc.lr × batch_lr_scale,
           momentum=0.95, ns_steps=mc.ns_steps, beta2=0.9, weight_decay=mc.weight_decay
    5) 返回 MuonAdamW(param_groups, mc)

    验证：
      uv run pytest -k optim -v
      训练日志里应该打印出「Muon N 个张量 / K 种形状，AdamW M 个张量」
      冒烟检查：分组后所有参数都被覆盖（断言不报错）
    """
    raise NotImplementedError(
        "待实现：setup_optimizer —— 五步，见 docstring\n"
        "  参考实现：git show solution:src/optim/muon.py")
