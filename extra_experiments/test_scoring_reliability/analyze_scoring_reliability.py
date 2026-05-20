"""
评分复测信度分析脚本

读取 scoring_reliability_experiment.py 生成的 all_scores.csv，
计算以下指标：
  - ICC(2,1)：双向随机效应、绝对一致性、单次测量的组内相关系数
  - 95% 置信区间
  - 每个样本 5 次测量的标准差均值
  - 按 Koo & Li (2016) 标准的信度等级判定

输出：
  - results/icc_analysis/icc_report.csv
  - results/icc_analysis/per_sample_sd.csv
  - results/icc_analysis/report.md
  - results/icc_analysis/plots/

用法:
    python analyze_scoring_reliability.py
    python analyze_scoring_reliability.py --input results/all_scores.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
RESULTS_DIR  = SCRIPT_DIR / "results"
ANALYSIS_DIR = RESULTS_DIR / "icc_analysis"
PLOT_DIR     = ANALYSIS_DIR / "plots"

# ---------------------------------------------------------------------------
# ICC(2,1) 计算
# ---------------------------------------------------------------------------

def compute_icc21(data: np.ndarray) -> dict[str, float]:
    """
    计算 ICC(2,1)：双向随机效应、绝对一致性、单次测量。
    参考：Shrout & Fleiss (1979); Koo & Li (2016)。

    Parameters
    ----------
    data : np.ndarray, shape (n, k)
        n 个被试（样本对），k 次测量（重复运行）。

    Returns
    -------
    dict 包含：
        icc       : ICC(2,1) 点估计
        ci_lower  : 95% CI 下界
        ci_upper  : 95% CI 上界
        ms_between: 被试间均方
        ms_raters : 评分者（运行次数）均方
        ms_error  : 残差均方
        f_value   : 被试间 F 统计量
    """
    n, k = data.shape  # 20 samples × 5 runs

    grand_mean   = data.mean()
    row_means    = data.mean(axis=1)   # (n,)
    col_means    = data.mean(axis=0)   # (k,)

    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    ss_raters  = n * np.sum((col_means  - grand_mean) ** 2)
    ss_total   = np.sum((data - grand_mean) ** 2)
    ss_within  = ss_total - ss_between
    ss_error   = ss_within - ss_raters

    df_between = n - 1
    df_raters  = k - 1
    df_error   = (n - 1) * (k - 1)

    ms_between = ss_between / df_between
    ms_raters  = ss_raters  / df_raters
    ms_error   = ss_error   / df_error if df_error > 0 else 0.0

    # ICC(2,1) absolute agreement（Shrout & Fleiss 1979, eq. ICC(2,1)）
    denom = ms_between + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    icc   = (ms_between - ms_error) / denom if denom > 0 else float("nan")

    # 95% CI via F-distribution（McGraw & Wong, 1996 方法）
    alpha = 0.05
    f_val = ms_between / ms_error if ms_error > 0 else float("inf")
    f_lower = f_val / stats.f.ppf(1 - alpha / 2, df_between, df_error)
    f_upper = f_val * stats.f.ppf(1 - alpha / 2, df_error, df_between)

    ci_lower = (f_lower - 1) / (f_lower + k - 1)
    ci_upper = (f_upper - 1) / (f_upper + k - 1)
    ci_lower = float(np.clip(ci_lower, -1.0, 1.0))
    ci_upper = float(np.clip(ci_upper, -1.0, 1.0))

    return {
        "icc":        float(icc),
        "ci_lower":   ci_lower,
        "ci_upper":   ci_upper,
        "ms_between": float(ms_between),
        "ms_raters":  float(ms_raters),
        "ms_error":   float(ms_error),
        "f_value":    float(f_val),
        "n_subjects": n,
        "n_raters":   k,
    }


def interpret_icc(icc: float) -> str:
    """按 Koo & Li (2016) 标准判定信度等级。"""
    if math.isnan(icc):
        return "无法计算"
    if icc < 0.50:
        return "不佳 (Poor, < 0.50)"
    if icc < 0.75:
        return "中等 (Moderate, 0.50–0.75)"
    if icc < 0.90:
        return "良好 (Good, 0.75–0.90)"
    return "优秀 (Excellent, > 0.90)"


# ---------------------------------------------------------------------------
# 核心分析
# ---------------------------------------------------------------------------

def analyze_metric(
    df: pd.DataFrame,
    metric: str,
    label: str,
) -> dict[str, object]:
    """
    对单个评分指标进行 ICC(2,1) 分析。
    df 应为长表，含 pair_id、run、{metric} 三列。
    """
    pivot = df.pivot_table(index="pair_id", columns="run", values=metric)
    pivot = pivot.dropna()
    n_complete = len(pivot)

    data = pivot.values  # (n, k)
    result = compute_icc21(data)

    per_sample_sd = pivot.std(axis=1, ddof=1)
    mean_sd = float(per_sample_sd.mean())
    max_sd  = float(per_sample_sd.max())

    # 若 ICC ≈ 1 且 SD = 0，说明完全确定性
    is_deterministic = max_sd < 1e-8

    return {
        "metric":          metric,
        "label":           label,
        "n_samples":       n_complete,
        "n_runs":          int(data.shape[1]),
        "icc":             result["icc"],
        "ci_lower":        result["ci_lower"],
        "ci_upper":        result["ci_upper"],
        "ms_between":      result["ms_between"],
        "ms_raters":       result["ms_raters"],
        "ms_error":        result["ms_error"],
        "f_value":         result["f_value"],
        "mean_sd":         mean_sd,
        "max_sd":          max_sd,
        "interpretation":  interpret_icc(result["icc"]),
        "is_deterministic": is_deterministic,
        "pivot":           pivot,
        "per_sample_sd":   per_sample_sd,
    }


def run_analysis(all_scores_csv: Path) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读取评分数据: {all_scores_csv}", flush=True)
    df = pd.read_csv(all_scores_csv, encoding="utf-8-sig")
    print(f"    共 {len(df)} 行, {df['run'].nunique()} 次运行, {df['pair_id'].nunique()} 个样本对")

    # 只保留状态正常的行
    if "status" in df.columns:
        n_before = len(df)
        df = df[df["status"] == "ok"].copy()
        print(f"    过滤后 {len(df)} 行（排除 {n_before - len(df)} 行非 OK 状态）")

    metrics = [
        ("sentence_score",    "句级评分 (sentence_score)"),
        ("phone_mean_score",  "音素均值评分 (phone_mean_score)"),
        ("word_mean_score",   "词级均值评分 (word_mean_score)"),
        ("overall_score",     "综合评分 (overall_score)"),
    ]
    # 过滤掉数据中不存在的列
    metrics = [(m, lbl) for m, lbl in metrics if m in df.columns]

    all_results: list[dict] = []
    per_sample_sd_frames: list[pd.DataFrame] = []

    for metric, label in metrics:
        print(f"\n>>> 分析: {label}", flush=True)
        res = analyze_metric(df, metric, label)
        all_results.append(
            {
                "指标":       res["label"],
                "列名":       res["metric"],
                "样本数":     res["n_samples"],
                "重复次数":   res["n_runs"],
                "ICC(2,1)":   f"{res['icc']:.4f}",
                "95%CI下界":  f"{res['ci_lower']:.4f}",
                "95%CI上界":  f"{res['ci_upper']:.4f}",
                "均值SD":     f"{res['mean_sd']:.4f}",
                "最大SD":     f"{res['max_sd']:.4f}",
                "信度等级":   res["interpretation"],
                "完全确定性": "是" if res["is_deterministic"] else "否",
            }
        )
        print(f"    ICC(2,1) = {res['icc']:.4f}  [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]")
        print(f"    信度等级  = {res['interpretation']}")
        print(f"    均值 SD   = {res['mean_sd']:.4f}（最大 SD = {res['max_sd']:.4f}）")

        # 收集每样本 SD 表
        sd_frame = pd.DataFrame(
            {
                "pair_id": res["per_sample_sd"].index,
                f"sd_{metric}": res["per_sample_sd"].values,
            }
        )
        per_sample_sd_frames.append(sd_frame.set_index("pair_id"))

        # 生成图表
        make_metric_plots(res, metric)

    # 保存 ICC 汇总表
    icc_df = pd.DataFrame(all_results)
    icc_csv = ANALYSIS_DIR / "icc_report.csv"
    icc_df.to_csv(icc_csv, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] ICC 汇总表: {icc_csv}", flush=True)

    # 保存每样本 SD 表
    if per_sample_sd_frames:
        import functools
        sd_combined = functools.reduce(
            lambda a, b: a.join(b, how="outer"), per_sample_sd_frames
        )
        sd_csv = ANALYSIS_DIR / "per_sample_sd.csv"
        sd_combined.to_csv(sd_csv, encoding="utf-8-sig")
        print(f"[SAVE] 每样本 SD 表: {sd_csv}", flush=True)

    # 生成报告
    write_report(all_results, df)
    print(f"\n[DONE] 分析完成，结果目录: {ANALYSIS_DIR}", flush=True)


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------

def make_metric_plots(res: dict, metric: str) -> None:
    pivot: pd.DataFrame = res["pivot"]
    n_runs = res["n_runs"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    fig.suptitle(res["label"], fontsize=13, fontweight="bold")

    # 左图：每次运行的分数散点 + 连线（按样本对）
    ax = axes[0]
    run_cols = sorted(pivot.columns)
    x = np.arange(1, n_runs + 1)
    for _, row in pivot.iterrows():
        ax.plot(x, row[run_cols].values, color="steelblue", alpha=0.35, linewidth=0.8)
    # 均值线
    means = [pivot[c].mean() for c in run_cols]
    ax.plot(x, means, color="crimson", linewidth=2.2, marker="o", label="均值")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Run {r}" for r in run_cols])
    ax.set_xlabel("测量次数")
    ax.set_ylabel("评分")
    ax.set_ylim(0, 105)
    ax.set_title(f"各样本每次测量分数（n={res['n_samples']}）")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)

    # 右图：每个样本的 SD 条形图
    ax2 = axes[1]
    sd_vals = res["per_sample_sd"].sort_values(ascending=False)
    pair_labels = [str(p) for p in sd_vals.index]
    ax2.bar(range(len(sd_vals)), sd_vals.values, color="steelblue", alpha=0.75)
    ax2.axhline(res["mean_sd"], color="crimson", linestyle="--", linewidth=1.5,
                label=f"均值 SD = {res['mean_sd']:.4f}")
    ax2.axhline(0.5, color="orange", linestyle=":", linewidth=1.2,
                label="参考阈值 0.5")
    ax2.set_xticks(range(len(sd_vals)))
    ax2.set_xticklabels(pair_labels, rotation=70, ha="right", fontsize=7)
    ax2.set_xlabel("样本对")
    ax2.set_ylabel("标准差（SD）")
    ax2.set_title(
        f"各样本 SD   ICC(2,1)={res['icc']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]"
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y")

    fig.tight_layout()
    out_path = PLOT_DIR / f"{metric}_reliability.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    [PLOT] {out_path}", flush=True)


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def write_report(all_results: list[dict], df: pd.DataFrame) -> None:
    report_path = ANALYSIS_DIR / "report.md"
    n_pairs = df["pair_id"].nunique()
    n_runs  = df["run"].nunique()

    lines = [
        "# 评分复测信度实验报告",
        "",
        "## 实验概况",
        "",
        f"- **样本对数量**：{n_pairs} 对（从 6 位学习者 × 10 句中随机抽取，seed=42）",
        f"- **重复测量次数**：{n_runs} 次",
        f"- **总评分数**：{n_pairs} × {n_runs} = {n_pairs * n_runs}",
        "- **流水线**：MFA 强制对齐 → GFCC 特征提取（nfilts=40, nceps=13） → FastDTW 相似度评分",
        "",
        "## ICC(2,1) 结果",
        "",
        "信度判定标准（Koo & Li, 2016）：",
        "- ICC < 0.50：不佳（Poor）",
        "- 0.50 ≤ ICC < 0.75：中等（Moderate）",
        "- 0.75 ≤ ICC < 0.90：良好（Good）",
        "- ICC ≥ 0.90：优秀（Excellent）",
        "",
        "| 评分指标 | ICC(2,1) | 95% CI | 均值 SD | 信度等级 |",
        "|----------|----------|--------|---------|----------|",
    ]

    primary_icc: float | None = None
    for r in all_results:
        lines.append(
            f"| {r['指标']} | {r['ICC(2,1)']} "
            f"| [{r['95%CI下界']}, {r['95%CI上界']}] "
            f"| {r['均值SD']} | {r['信度等级']} |"
        )
        if r["列名"] == "sentence_score" and primary_icc is None:
            primary_icc = float(r["ICC(2,1)"])

    lines += [
        "",
        "## 解读",
        "",
    ]

    if primary_icc is not None:
        if primary_icc >= 0.99:
            lines.append(
                f"句级评分的 ICC(2,1) = {primary_icc:.4f}，接近或等于理论上限 1.0，"
                "说明 MFA-GFCC-DTW 流水线具备**完全确定性**，每次重复评测给出完全一致的分数。"
            )
        elif primary_icc >= 0.90:
            lines.append(
                f"句级评分的 ICC(2,1) = {primary_icc:.4f}（优秀），"
                "表明系统具有**优秀的复测信度**，重复评测结果高度一致。"
            )
        elif primary_icc >= 0.75:
            lines.append(
                f"句级评分的 ICC(2,1) = {primary_icc:.4f}（良好），"
                "系统具有良好信度，但仍有轻微波动，建议检查 MFA 对齐的随机性。"
            )
        else:
            lines.append(
                f"句级评分的 ICC(2,1) = {primary_icc:.4f}，信度未达优秀级别，"
                "建议核查 MFA 对齐过程是否引入了随机性（批处理顺序、并行线程数等）。"
            )

    lines += [
        "",
        "## 参考文献",
        "",
        "- Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting "
        "intraclass correlation coefficients for reliability research. "
        "*Journal of Chiropractic Medicine*, 15(2), 155–163.",
        "- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: "
        "Uses in assessing rater reliability. "
        "*Psychological Bulletin*, 86(2), 420–428.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] Markdown 报告: {report_path}", flush=True)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="评分复测信度分析")
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "all_scores.csv",
        help="all_scores.csv 路径（默认 results/all_scores.csv）",
    )
    args = parser.parse_args()

    if not args.input.exists():
        import sys
        sys.exit(
            f"[ERROR] 输入文件不存在: {args.input}\n"
            "请先运行 scoring_reliability_experiment.py 生成评分数据。"
        )

    run_analysis(args.input)


if __name__ == "__main__":
    main()
