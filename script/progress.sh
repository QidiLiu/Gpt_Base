#!/usr/bin/env bash
# 进度检查：告诉你哪一章还没做完。
#
# 用法：
#   bash script/progress.sh          # 全部章节
#   bash script/progress.sh 03       # 只看第 03 章
#
# 原理：每章有若干 pytest 用例作为「完成判据」。
# 跑通 = 你那一章敲对了。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# 章节 -> pytest -k 表达式（多个用逗号或空格分隔的 -k 都行）
declare -A CHAPTERS=(
  ["00-01"]="tests/test_presets.py -k 'presets'"
  ["02"]="tests/test_data.py -k 'tokenizer or special or decode_bytes or parquet'"
  ["03"]="tests/test_data.py -k 'tokenizer or decode_bytes'"
  ["04"]="tests/test_data.py -k 'mask or python_output or truncate or alternation'"
  ["05"]="tests/test_data.py -k 'dataloader or targets_are or bos or padding or bestfit or split'"
  ["06"]="tests/test_data.py -k 'decode_bytes or token_bytes or bpb'"
  ["07"]="tests/test_core.py -k 'uniform or loss_reduction or ignore or causal'"
  ["08"]="tests/test_core.py -k 'rmsnorm'"
  ["09"]="tests/test_core.py -k 'rope'"
  ["10"]="tests/test_core.py -k 'attend or sliding'"
  ["11"]="tests/test_core.py -k 'sdpa'"
  ["12"]="tests/test_core.py -k 'sdpa or attend'"
  ["13"]="tests/test_core.py -k 'mlp_activation'"
  ["14"]="tests/test_core.py -k 'causal'"
  ["15"]="tests/test_core.py -k 'uniform or sampling or topk'"
  ["16"]="tests/test_core.py -k 'meta_device'"
  ["17"]="tests/test_core.py -k 'lambdas'"
  ["18"]="tests/test_core.py -k 'value_embeds'"
  ["19"]="tests/test_core.py -k 'smear or backout'"
  ["20"]="tests/test_core.py -k 'sliding'"
  ["21"]="tests/test_optim.py -k 'adamw_step or adamw_decouples'"
  ["22"]="tests/test_optim.py -k 'muon_orthogonalization'"
  ["23"]="tests/test_optim.py -k 'orthogonalize'"
  ["24"]="tests/test_optim.py -k 'polar'"
  ["25"]="tests/test_optim.py -k 'orthogonalize or muon_plus or nor_muon'"
  ["26"]="tests/test_optim.py -k 'cautious'"
  ["27"]="tests/test_optim.py -k 'grouping'"
  ["28"]="tests/test_optim.py -k 'hyperparams'"
)

WANT="${1:-}"

if [ -n "$WANT" ]; then
  for k in "${!CHAPTERS[@]}"; do
    case "$k" in "$WANT"*) echo "${CHAPTERS[$k]}";; esac
  done
  exit 0
fi

printf '\n\033[1m章节进度\033[0m\n'
printf '%-8s %-6s %s\n' "章节" "状态" "完成判据"
printf '%s\n' "------------------------------------------------------------------"
for k in $(echo "${!CHAPTERS[@]}" | tr ' ' '\n' | sort); do
  sel="${CHAPTERS[$k]}"
  # shellcheck disable=SC2086
  if eval "pytest -q $sel" >/dev/null 2>&1; then
    st=$'\033[32m通过\033[0m'
  else
    st=$'\033[33m未完成\033[0m'
  fi
  printf '%-8s %-18b %s\n' "$k" "$st" "$sel"
done
echo
echo "第 05 章的打包可视化：uv run python scratch/packing_demo.py"
echo "第 06 章的 bpb 验证  ：uv run python scratch/bpb_demo.py"
echo "第 23 章的正交化验证：uv run python scratch/ortho_demo.py"
echo
echo "只读章节（卷5-8）不需要敲代码，直接读 src/ 即可。"
echo "看完整答案：git checkout solution   回到你的版本：git checkout main"
echo
printf '\033[1;33m[重要]\033[0m 改完 src/ 或 tests/ 之后，先确认判据本身是可达的：\n'
echo "  git stash push -- src/ tests/    # 有改动才需要；工作区干净时会报错，可跳过"
echo "  git checkout solution && uv run pytest tests/ -q"
echo "  # 预期：243 passed, 2 skipped（2 个 skipped 是「只在骨架态有意义」的判据）"
echo "  git checkout main && git stash pop"
echo "  （判据若在完整答案上都过不了，你永远敲不到全绿 —— 见 tests/test_judging_soundness.py）"
