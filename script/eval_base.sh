#!/usr/bin/env bash
# 评测基座模型：val bits-per-byte + 多选题准确率 + 固定 prompt 采样。
#
# 用法：
#   bash script/eval_base.sh            # 默认 smoke
#   bash script/eval_base.sh full
#   bash script/eval_base.sh smoke --eval-tokens 1048576    # 更精确的 bpb
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE="${1:-smoke}"; [ $# -gt 0 ] && shift || true

[ -f "runs/base_checkpoints/$TAG/model_"*.pt ] 2>/dev/null \
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
