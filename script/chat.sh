#!/usr/bin/env bash
# 和模型聊天。
#
# 用法：
#   bash script/chat.sh                          # 交互式
#   bash script/chat.sh smoke -p "为什么天是蓝色的？"
#   bash script/chat.sh full -t 0                # 贪心解码（确定性）
#   bash script/chat.sh full -m 512 --no-tools   # 更长回复，关掉计算器
#
# 交互模式内的命令：
#   /reset   清空对话历史
#   /exit    退出
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE="$1"; shift || true

[ -f "runs/base_checkpoints/$TAG/model_"*.pt ] 2>/dev/null \
  || die "没有找到模型 $TAG。先跑：bash script/train_base.sh $MODE"

if [ -f "runs/sft_checkpoints/$TAG/model_"*.pt ] 2>/dev/null ]; then
  say "聊天：$TAG（SFT 模型）"
else
  warn "没有 SFT 模型，将使用基座模型。它会续写文本，但不会和你对话。"
  warn "想要会对话的模型：bash script/train_sft.sh $MODE"
  say "聊天：$TAG（基座模型）"
fi

python -m training.chat --mode "$MODE" --tag "$TAG" "$@"
