#!/usr/bin/env bash
# 评测 SFT 模型：多选题准确率 + GSM8K pass@1 + 多轮对话展示。
#
# 用法：
#   bash script/eval_sft.sh             # 默认 smoke
#   bash script/eval_sft.sh full
#   bash script/eval_sft.sh smoke --gsm8k-examples 100
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE="${1:-smoke}"; [ $# -gt 0 ] && shift || true

[ -f "runs/sft_checkpoints/$TAG/model_"*.pt ] 2>/dev/null \
  || die "没有找到 SFT 模型 $TAG。先跑：bash script/train_sft.sh $MODE"

say "评测 SFT 模型 $TAG"
info "注意：多选题这里用的是「生成式」评分（让模型直接吐一个字母），"
info "      而不是基座评测里的 loglikelihood 评分 —— 这正是 SFT 教会它的格式。"
info "      单卡小模型准确率通常接近随机（25%），看格式对不对比看准确率更有意义。"
python -m training.eval_sft --mode "$MODE" --tag "$TAG" "$@"

say "评测完成"
info "  结果 json : runs/eval_sft.json"
