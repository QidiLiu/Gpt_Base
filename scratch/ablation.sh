#!/usr/bin/env bash
# 消融实验结果汇总。
#
# 用法：
#   bash scratch/ablation.sh              # 汇总全部（基线默认 d6_base）
#   bash scratch/ablation.sh d4base       # 指定基线 tag
#
# ★ 做消融实验时 train_base.sh 必须加 --no-resume，
#   否则同一个 tag 重跑会被 auto-resume 吞掉（0 步跑完），bpb 一模一样。
# ★ 必须用 ablation 档（d6）。debug 档只 20 步，模型还在初始化附近，
#   架构差异测不出来（实测五组 bpb 完全一致）。
#   smoke 档 896 步勉强能看出趋势，ablation 档才能得出可信结论。
#
#   档位为什么不是 full：full 档是 d24 + 5 个 trick 全开 + Muon advanced，
#   一次 128 小时。消融要跑 8 组以上，而且基线必须保持**中性**
#   （Muon simple + trick 全关），否则测出来的是「A 相对 A+B」，
#   单项贡献被稀释。这正是 ablation 档存在的理由。
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
  echo "  先跑：bash script/train_base.sh ablation --model-tag $BASE_TAG --no-resume"
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
echo "  1. 消融必须用 ablation 档（d6）—— debug 档（20 步）测不出架构差异"
echo "     full 档（d24，128 小时）不适合做消融，且基线非中性会稀释单项贡献"
echo "  2. 一次只改一个变量"
echo "  3. |Δ| < 0.001 时重复跑一次，确认不是初始化随机性的波动"
echo ""
echo "─── 阈值 0.001 是实测来的，不是拍的 ────────────────────────────"
echo "  早期版本这里写的是 0.02，那是个**从未验证过的经验值**"
echo "  （从项目最早的 commit 7319830起就在了）。"
echo ""
echo "  实测 ablation 档噪声底（3 次同配置 + 1 次换初始化）："
echo "    同种子样本标准差 σ   = 0.000134"
echo "    Δ 的合成噪声 σ_Δ     = √2·σ = 0.000190"
echo "    换初始化的实测偏移= 0.000437"
echo "  => 阈值取 0.001 ≈ 5.3σ"
echo ""
echo "  ⚠ 原来的 0.02 比实测噪声大 105 倍（分母 σ_Δ，判定 |Δ| 用它）。"
echo "    后果：13 项消融里有 11 项会被误判成「在噪声带内 / 测不出差异」，"
echo "    而它们其实是 7σ~279σ 的真实效应。"
echo ""
echo "  ⚠ 引用倍数务必连分母一起说 —— 三个分母都合法，含义不同："
echo "       σ_Δ=0.000190 判定 |Δ| 用这个 → 105 倍（项目口径）"
echo "       σ  =0.000134 单次跑的样本标准差  → 149 倍"
echo "       偏移=0.000437 换初始化的最大偏移  →  46 倍"
echo ""
echo "  ⚠ 而纪律第 1 条说的 smoke 档噪声具体多大，**至今没有测过**。"
