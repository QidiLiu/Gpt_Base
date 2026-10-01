#!/usr/bin/env bash
# 评测基座模型：val bits-per-byte + 多选题准确率 + 固定 prompt 采样。
#
# 用法：
#   bash script/eval_base.sh            # 默认 smoke
#   bash script/eval_base.sh full
#   bash script/eval_base.sh smoke --eval-tokens 1048576    # 更精确的 bpb
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE="${1:-smoke}"; [ $# -gt 0 ] && shift || true

# ⚠ 这里必须用 compgen 而不是 `[ -f glob ]`。真实事故：
#   写成 `[ -f "runs/.../model_"*.pt ]` 时，只要那个目录下有 **2 个以上**
#   checkpoint（full 档 save_every=200 必然产生多个），glob 就展开成多个参数，
#   bash 内建 `[` 报 "binary operator expected" 并返回退出码 2，于是 `|| die`
#   触发 —— **明明有存档，却报「没有找到」**。compgen 不受参数个数影响。
compgen -G "runs/base_checkpoints/$TAG/model_*.pt" > /dev/null \
  || die "没有找到 $TAG 的 checkpoint。先跑：bash script/train_base.sh $MODE"

say "评测基座模型 $TAG"
python -m training.eval_base --mode "$MODE" --tag "$TAG" "$@"

say "评测完成"
cat <<EOF
  结果 json : runs/eval_base.json
  训练曲线  : runs/base_checkpoints/$TAG/curves.png

  怎么读 bpb：
    · bpb 与词表大小无关，是唯一能跨模型比较的语言建模指标
    · 参考量级：随机 ≈ 8，英文自然文本 ≈ 1.0-1.5
    · 消融实验里看的是「相对差异」（0.01-0.05 量级），不是绝对值
EOF
