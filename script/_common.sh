#!/usr/bin/env bash
# 所有 script 共用的环境设置。由其他脚本 source，不要直接执行。
set -euo pipefail

# 切到项目根目录（不管从哪调用）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── 网络镜像 ───────────────────────────────────────────────
# 本机实测（2026-09）：
#   huggingface.co           不可达
#   raw.githubusercontent.com 不可达
#   pypi.org                 不可达（uv 已配腾讯镜像）
# 所以统一走镜像。想用官方源就 export HF_ENDPOINT=https://huggingface.co
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export GITHUB_RAW_MIRROR="${GITHUB_RAW_MIRROR:-https://ghproxy.net/https://raw.githubusercontent.com}"
# OMP 线程数设成 1：dataloader 自己做并行，OMP 再抢核只会互相拖慢
export OMP_NUM_THREADS=1
# 让 PyTorch 的显存分配器用可扩展段，减少碎片
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# ── 虚拟环境 ───────────────────────────────────────────────
if [ -d .venv ]; then
  source .venv/bin/activate
else
  echo "找不到 .venv。先跑：uv sync" >&2
  exit 1
fi

# ── 档位 ───────────────────────────────────────────────────
# debug : 极小模型 + 20 步，用来单步调试，秒级完成
# smoke : d4  + 约 900 步，2-3 分钟跑完全流程（每章验证用这个）
# full  : d6  + 约 3100 步，20-25 分钟
MODE="${1:-smoke}"
case "$MODE" in
  debug|smoke|full) ;;
  *) echo "档位必须是 debug / smoke / full，得到 '$MODE'" >&2; exit 1 ;;
esac

# 档位 -> 模型 tag（与 src/common/config.py 的 PRESETS 保持一致）
case "$MODE" in
  debug) TAG="d2";  SHARDS=1 ;;
  smoke) TAG="d4";  SHARDS=1 ;;
  full)  TAG="d6";  SHARDS=4 ;;
esac

# 档位 -> 期望的词表大小（与 config.py 的 PRESETS 一致，用于校验 tokenizer）
case "$MODE" in
  debug|smoke) VOCAB=8192 ;;
  full)        VOCAB=16384 ;;
esac

# 档位 -> 传给 python 脚本的参数
case "$MODE" in
  debug) EXTRA_TRAIN="--num-iterations 20" ;;
  smoke) EXTRA_TRAIN="" ;;
  full)  EXTRA_TRAIN="" ;;
esac

# 是否跳过数据下载（SMOKE_ALL_IN_ONE=1 时由 train_base.sh 自行串联）
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

# ── 输出小工具 ─────────────────────────────────────────────
say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
info() { printf '\033[0;37m      %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }
