#!/usr/bin/env bash
# 消融实验结果汇总。
#
# 用法：
#   bash scratch/ablation.sh              # 汇总全部（基线默认 d6_base）
#   bash scratch/ablation.sh d4base       # 指定基线 tag
#
# ★ 做消融实验时 train_base.sh 必须加 --no-resume，
#   否则同一个 tag 重跑会被 auto-resume 吞掉（0 步跑完），bpb 一模一样。
# ★ 必须用 full 档。debug 档只 20 步，模型还在初始化附近，
#   架构差异测不出来（实测五组 bpb 完全一致）。
#   smoke 档 896 步勉强能看出趋势，full 档才能得出可信结论。
cd "$(dirname "$0")/.."
BASE_TAG="${1:-d6_base}"

bpb_of() {  # $1 = meta json 路径
  python3 - "$1" <<'PY'
import json, sys
print(f"{json.load(open(sys.argv[1]))['best_val_bpb']:.4f}")
PY
}

delta_of() {  # $1 = bpb, $2 = base
  python3 - "$1" "$2" <<'PY'
import sys
print(f"{float(sys.argv[1]) - float(sys.argv[2]):+.4f}")
PY
}

printf "%-20s %-10s %-12s\n" "实验" "val_bpb" "Δ vs base"
printf "%s\n" "----------------------------------------------------------------"
base=""
for d in runs/base_checkpoints/*/; do
  t=$(basename "$d")
  [ "$t" = "$BASE_TAG" ] || continue
  f=$(ls -t "$d"meta_*.json 2>/dev/null | head -1)
  [ -n "$f" ] && base=$(bpb_of "$f")
done
if [ -z "$base" ]; then
  echo "  找不到基线 $BASE_TAG，Δ 列将为空"
  echo "  先跑：bash script/train_base.sh full --model-tag $BASE_TAG --no-resume"
fi
for d in runs/base_checkpoints/*/; do
  t=$(basename "$d")
  f=$(ls -t "$d"meta_*.json 2>/dev/null | head -1)
  [ -z "$f" ] && continue
  bpb=$(bpb_of "$f")
  if [ "$t" = "$BASE_TAG" ]; then
    printf "%-20s %-10s %-12s\n" "$t" "$bpb" "(基线)"
  elif [ -n "$base" ]; then
    printf "%-20s %-10s %-12s\n" "$t" "$bpb" "$(delta_of "$bpb" "$base")"
  else
    printf "%-20s %-10s %-12s\n" "$t" "$bpb" "?"
  fi
done
echo
echo "Δ < 0 表示比基线好（bpb 越低越好）"
echo "纪律："
echo "  1. 消融必须用 full 档 —— debug 档（20 步）测不出架构差异"
echo "  2. 一次只改一个变量"
echo "  3. |Δ| < 0.02 时重复跑一次，确认不是初始化随机性的波动"
