#!/usr/bin/env bash
# SFT（监督微调）：把基座模型变成会对话的模型。
#
# 用法：
#   bash script/train_sft.sh            # 默认 smoke
#   bash script/train_sft.sh full
#   bash script/train_sft.sh smoke --num-iterations 500
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE="${1:-smoke}"; [ $# -gt 0 ] && shift || true

[ -f "runs/base_checkpoints/$TAG/model_"*.pt ] 2>/dev/null \
  || die "没有找到基座 $TAG。先跑：bash script/train_base.sh $MODE"

say "SFT：基座 $TAG"
info "数据混合：SmolTalk×1（多轮对话）+ MMLU×3（教多选格式）+ GSM8K×4（教数学与工具调用）"
info "核心机制：loss masking —— 只有 assistant 的 token 参与 loss"
info "首次运行会下载任务数据：SmolTalk 1 片(224MB) + MMLU(48MB) + GSM8K(3MB)"

# shellcheck disable=SC2086
python -m training.train_sft --mode "$MODE" --base-tag "$TAG" "$@"

say "SFT 完成"
cat <<EOF

下一步：
  bash script/eval_sft.sh $MODE      # 多选准确率 + GSM8K pass@1 + 多轮对话展示
  bash script/chat.sh    $MODE        # 交互式聊天

  SFT 权重 : runs/sft_checkpoints/$TAG/
EOF
