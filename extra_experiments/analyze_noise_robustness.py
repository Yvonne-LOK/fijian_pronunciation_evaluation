"""
对已有 scores_detail.csv 做后处理：
  1) 用 norm_distance 重新计算每个 (feature, snr) 的均值和标准差
  2) 用线性映射重新计算分数：score_lin = clip(100*(1 - d / D_REF), 0, 100)
  3) 生成 4 张图：
      - distance_mean_vs_snr.png        (距离均值 vs SNR，越低越好)
      - distance_std_vs_snr.png         (距离标准差 vs SNR，越低越稳定)
      - score_drop_vs_snr.png           (重标后分数衰减，clean - noisy)
      - score_std_vs_snr.png            (重标后分数标准差)
  4) 写出 summary_v2.csv

输入：D:\\workdir\\extra_experiments\\noise_robustness\\results\\scores_detail.csv
输出：同目录下 summary_v2.csv 和 plots/ 下 4 个 png
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------
D_REF = 4.0  # 线性映射的"完全失配"距离上限，由数据分布决定（max d≈3.88）

if Path("/mnt/d/workdir").exists():
    OUT_ROOT = Path("/mnt/d/workdir/extra_experiments/noise_robustness")
else:
    OUT_ROOT = Path(r"D:\workdir\extra_experiments\noise_robustness")

DETAIL_CSV = OUT_ROOT / "results" / "scores_detail.csv"
SUMMARY_CSV = OUT_ROOT / "results" / "summary_v2.csv"
PLOT_DIR = OUT_ROOT / "results" / "plots"

SNR_ORDER = ["clean", "snr20", "snr10", "snr05"]
SNR_LABEL = {"clean": "Clean", "snr20": "20 dB", "snr10": "10 dB", "snr05": "5 dB"}
NOISY_ORDER = ["snr20", "snr10", "snr05"]
NOISY_LABEL = ["20 dB", "10 dB", "5 dB"]

FEATURE_STYLE = {
    "gfcc": dict(marker="o", color="#1f77b4", label="GFCC"),
    "mfcc": dict(marker="s", color="#d62728", label="MFCC"),
}


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def linear_score(d: float, d_ref: float = D_REF) -> float:
    return float(np.clip(100.0 * (1.0 - d / d_ref), 0.0, 100.0))


def main():
    print(f"reading {DETAIL_CSV}")
    df = pd.read_csv(DETAIL_CSV)
    print(f"  {len(df)} rows, columns = {list(df.columns)}")
    assert {"sentence_id", "feature", "snr", "norm_distance"}.issubset(df.columns)

    # 重新计算线性分数
    df["score_linear"] = df["norm_distance"].apply(linear_score)

    # 汇总
    grouped = df.groupby(["feature", "snr"])
    summary = grouped.agg(
        dist_mean=("norm_distance", "mean"),
        dist_std=("norm_distance", "std"),
        score_mean=("score_linear", "mean"),
        score_std=("score_linear", "std"),
    ).reset_index()

    # 把 SNR 排序
    summary["snr"] = pd.Categorical(summary["snr"], categories=SNR_ORDER, ordered=True)
    summary = summary.sort_values(["feature", "snr"]).reset_index(drop=True)

    # 加上"距离增量"和"分数衰减"两列（相对于 clean）
    summary["dist_increase"] = 0.0
    summary["score_drop"] = 0.0
    for feat in summary["feature"].unique():
        mask = summary["feature"] == feat
        d_clean = summary.loc[mask & (summary["snr"] == "clean"), "dist_mean"].iloc[0]
        s_clean = summary.loc[mask & (summary["snr"] == "clean"), "score_mean"].iloc[0]
        summary.loc[mask, "dist_increase"] = summary.loc[mask, "dist_mean"] - d_clean
        summary.loc[mask, "score_drop"] = s_clean - summary.loc[mask, "score_mean"]

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")
    print(f"\nwrote {SUMMARY_CSV}\n")
    print(summary.to_string(index=False))

    # ------------------------- 画图 -------------------------
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(NOISY_ORDER))

    def _line_plot(metric: str, ylabel: str, title: str, out_png: Path,
                   annotate_fmt: str = "{:.2f}"):
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=140)
        for feat in ("gfcc", "mfcc"):
            ys = [
                summary[(summary["feature"] == feat) & (summary["snr"] == s)][metric].iloc[0]
                for s in NOISY_ORDER
            ]
            sty = FEATURE_STYLE[feat]
            ax.plot(x, ys, marker=sty["marker"], color=sty["color"],
                    linewidth=2, label=sty["label"])
            for xi, yi in zip(x, ys):
                ax.annotate(annotate_fmt.format(yi), (xi, yi),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(NOISY_LABEL)
        ax.set_xlabel("SNR")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
        print(f"saved {out_png}")

    _line_plot(
        "dist_mean",
        "Mean normalized DTW distance",
        "Feature stability under noise (lower = more robust)",
        PLOT_DIR / "distance_mean_vs_snr.png",
        annotate_fmt="{:.3f}",
    )
    _line_plot(
        "dist_std",
        "Std. of normalized DTW distance (n=30)",
        "Cross-sentence variability of distance",
        PLOT_DIR / "distance_std_vs_snr.png",
        annotate_fmt="{:.3f}",
    )
    _line_plot(
        "score_drop",
        "Score drop (clean − noisy), linear mapping",
        "Score degradation under noise (lower = more robust)",
        PLOT_DIR / "score_drop_vs_snr.png",
        annotate_fmt="{:.2f}",
    )
    _line_plot(
        "score_std",
        "Std. of score across 30 sentences",
        "Score stability under noise (lower = more stable)",
        PLOT_DIR / "score_std_vs_snr.png",
        annotate_fmt="{:.2f}",
    )

    # ------------------------- 简短的文本结论打印 -------------------------
    print("\n=== quick check ===")
    for snr in NOISY_ORDER:
        g = summary[(summary["feature"] == "gfcc") & (summary["snr"] == snr)].iloc[0]
        m = summary[(summary["feature"] == "mfcc") & (summary["snr"] == snr)].iloc[0]
        print(
            f"{snr}: "
            f"GFCC dist={g['dist_mean']:.3f}±{g['dist_std']:.3f}, "
            f"MFCC dist={m['dist_mean']:.3f}±{m['dist_std']:.3f}; "
            f"MFCC/GFCC = {m['dist_mean'] / g['dist_mean']:.3f}"
        )

    print("\ndone.")


if __name__ == "__main__":
    main()
