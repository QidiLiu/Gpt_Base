#!/usr/bin/env bash
# 预训练基座模型。会自动串好：下载数据 -> 训 tokenizer -> 训练。
#
# 用法：
#   bash script/train_base.sh            # 默认 smoke
#   bash script/train_base.sh smoke
#   bash script/train_base.sh full       # 20-25 分钟
#   bash script/train_base.sh smoke --all-tricks --muon-advanced   # 透传消融开关
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE="$1"; shift || true
EXTRA_ARGS=("$@")

say "档位 $MODE  (tag=$TAG, shards=$SHARDS, vocab=$VOCAB)"
info "Python : $(python -V 2>&1)"
info "设备   : $(uv run python -c 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' 2>/dev/null || echo '?')"

# ── 1) 数据 ────────────────────────────────────────────────
say "步骤 1/3  下载预训练数据（$SHARDS 个训练 shard + 1 个验证 shard）"
info "每个 shard 约 92 MB。最后一个 shard 固定作为验证集，不参与训练。"
info "已经下过的会自动跳过，可以放心重复执行。"
if [ "$SKIP_DOWNLOAD" != "1" ]; then
  python -m data.dataset -n "$SHARDS" -w 4
fi

# ── 2) tokenizer ───────────────────────────────────────────
say "步骤 2/3  准备 BPE tokenizer (vocab=$VOCAB)"
info "tokenizer 只需训练一次，存在 ~/.cache/gpt_base/tokenizer/ 下会自动跳过。"
# 已有 tokenizer 但词表大小和档位不匹配时会提示重建
python -m training.train_tokenizer --vocab-size "$VOCAB" --shards 2

# ── 3) 预训练 ──────────────────────────────────────────────
say "步骤 3/3  预训练（会自动从已有 checkpoint 续训）"
info "只想深度？加 --depth 8。想看消融？加 --no-rope / --norm-type layer / --muon-advanced 等。"
info "完整开关列表：uv run python -m training.train_base --help"
python -m training.train_base --mode "$MODE" --model-tag "$TAG" \
  $EXTRA_TRAIN "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

say "预训练完成"
cat <<EOF

下一步：
  bash script/eval_base.sh $MODE     # 评测 bpb / 多选题 / 采样
  bash script/train_sft.sh $MODE     # 对话微调
  bash script/chat.sh    $MODE       # 直接聊天（没跑 SFT 也能聊，只是不会说人话）

  checkpoint : runs/base_checkpoints/$TAG/
  训练曲线   : runs/base_checkpoints/$TAG/curves.png
  历史数据   : runs/base_checkpoints/$TAG/history.csv
EOF
