"""
从 `runs/base_checkpoints/*/meta_*.json` 重算消融显著性。

★ 这个脚本存在的原因：纪律第 3 条的阈值曾经写的是 0.02，
  那是**从未被测量过的经验值**（项目最早的 commit `7319830` 里就有）。
  实测发现真实噪声只有 0.000134 —— 小 150 倍 ——
  于是 13 项消融里有 11 项被误判成「测不出差异」。

  「阈值本身是没测过的取值」是本项目反复出现的一类失败模式，
  所以这里把它变成**可复算**的：跑一次本脚本就能知道显著性怎么判的。

用法：
    python scratch/measure_noise_floor.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scratch" / "ablation_noise_floor.json"

# 与消融跑批的 tag 对应（见 doc/tutorial/README.md 的 ★ 消融总表）
ABLATION_FLAGS = {
    "d6_norope":     "--no-rope",
    "d6_tie":        "--tie-embeddings",
    "d6_alltricks":  "--all-tricks",
    "d6_ve":         "--use-value-embeds",
    "d6_resid":      "--use-resid-lambdas",
    "d6_noqk":       "--no-qk-norm",
    "d6_smear":      "--use-smear",
    "d6_nosoftcap":  "--no-softcap",
    "d6_x0":         "--use-x0-lambdas",
    "d6_backout":    "--use-backout",
    "d6_adv":        "--muon-advanced",
    "d6_layernorm":  "--norm-type layer",
    "d6_shve":       "--shared-value-embeds",
}
BASE_RUNS = ("d6_base", "base_r1", "base_r2")


def val_bpb(tag: str) -> float | None:
    files = sorted(glob.glob(str(ROOT / "runs" / "base_checkpoints" / tag / "meta_*.json")))
    if not files:
        return None
    return json.load(open(files[-1]))["best_val_bpb"]


def main() -> int:
    data = json.load(open(DATA))
    thresh = data["derived"]["threshold"]

    # ---- 1. 噪声底 ----
    same = [val_bpb(t) for t in BASE_RUNS]
    missing = [t for t, v in zip(BASE_RUNS, same) if v is None]
    if missing:
        print(f"缺少基线跑批：{missing} —— 先跑完再算显著性")
        print("  PYTHONPATH=src python -m training.train_base --mode ablation \\")
        print(f"      --model-tag {missing[0]} --no-resume")
        return 1
    mean, sd = st.mean(same), st.stdev(same)
    sd_delta = (2 ** 0.5) * sd

    print("=== 噪声底（实测，不是拍的）===")
    for t, v in zip(BASE_RUNS, same):
        print(f"  {t:10s} {v:.7f}   偏离均值 {v - mean:+.7f}")
    s7 = val_bpb("base_s7")
    if s7 is not None:
        print(f"  base_s7   {s7:.7f}   偏离均值 {s7 - mean:+.7f} "
              f"（换初始化，{(s7 - mean) / sd:.1f}σ）")
    print(f"\n  同种子样本标准差 σ  = {sd:.6f}")
    print(f"  σ_Δ = √2·σ          = {sd_delta:.6f}")
    print(f"  显著性阈值          = {thresh}  （{thresh / sd_delta:.1f}σ）")

    # ---- 2. 逐项显著性 ----
    print(f"\n=== 消融显著性（基线 = {mean:.4f}，阈值 {thresh}）===")
    print(f"{'实验':16s} {'val_bpb':>9s} {'Δ':>10s} {'σ 倍数':>9s}  判定")
    print("-" * 68)
    rows = []
    for tag, flag in ABLATION_FLAGS.items():
        v = val_bpb(tag)
        if v is None:
            print(f"{tag:16s} {'(未跑)':>9s}")
            continue
        d = v - mean
        sigma = d / sd_delta
        if abs(d) < thresh:
            verdict = "真的测不出"
        elif abs(sigma) < 10:
            verdict = "弱但真实"
        else:
            verdict = "显著"
        rows.append((tag, v, d, sigma, verdict))
    for tag, v, d, sigma, verdict in sorted(rows, key=lambda r: r[2]):
        print(f"{tag:16s} {v:9.4f} {d:+10.4f} {sigma:9.1f}  {verdict}")

    unmeasurable = [t for t, _, _, _, v in rows if v == "真的测不出"]
    print(f"\n{len(rows)} 项里真的测不出的：{unmeasurable or '（无）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())