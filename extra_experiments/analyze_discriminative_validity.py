"""分析区分效度实验结果：均值/SD、ANOVA、Tukey HSD、效应量 η²，并生成图与报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PRIMARY_METRIC = "sentence_score_new_relative"
SECONDARY_METRICS = ["sentence_score", "overall_score_new_relative"]

LEVEL_ORDER = ["A", "B", "C"]
LEVEL_LABEL = {
    "A": "Level A (Self-pair)",
    "B": "Level B (Real learner)",
    "C": "Level C (Mismatched)",
}
LEVEL_COLOR = {"A": "#2ca02c", "B": "#1f77b4", "C": "#d62728"}


def load_data(utterance_csv: Path, manifest_csv: Path) -> pd.DataFrame:
    scores = pd.read_csv(utterance_csv)
    manifest = pd.read_csv(manifest_csv)
    merged = manifest.merge(scores, on=["pair_id", "reference_id", "learner_id"], how="left")
    if merged[PRIMARY_METRIC].isna().any():
        missing = merged.loc[merged[PRIMARY_METRIC].isna(), ["level", "pair_id"]]
        print(f"[warn] {len(missing)} pair(s) have NaN {PRIMARY_METRIC}:\n{missing.to_string(index=False)}")
    return merged


def describe_by_level(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for level in LEVEL_ORDER:
        sub = df.loc[df["level"] == level, metric].dropna()
        rows.append({
            "level": level,
            "label": LEVEL_LABEL[level],
            "n": int(sub.size),
            "mean": float(sub.mean()) if sub.size else float("nan"),
            "std": float(sub.std(ddof=1)) if sub.size > 1 else float("nan"),
            "min": float(sub.min()) if sub.size else float("nan"),
            "max": float(sub.max()) if sub.size else float("nan"),
            "median": float(sub.median()) if sub.size else float("nan"),
        })
    return pd.DataFrame(rows)


def anova_eta_squared(df: pd.DataFrame, metric: str) -> dict:
    groups = [df.loc[df["level"] == lvl, metric].dropna().to_numpy() for lvl in LEVEL_ORDER]
    f_stat, p_value = stats.f_oneway(*groups)
    grand = np.concatenate(groups)
    grand_mean = float(grand.mean())
    ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups))
    ss_total = float(((grand - grand_mean) ** 2).sum())
    eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")
    df_between = len(groups) - 1
    df_within = sum(len(g) for g in groups) - len(groups)
    ss_within = ss_total - ss_between
    omega_sq = ((ss_between - df_between * (ss_within / df_within)) /
                (ss_total + ss_within / df_within)) if df_within > 0 and ss_total > 0 else float("nan")
    return {
        "F": float(f_stat),
        "p_value": float(p_value),
        "df_between": int(df_between),
        "df_within": int(df_within),
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "eta_squared": float(eta_sq),
        "omega_squared": float(omega_sq),
    }


def tukey_hsd(df: pd.DataFrame, metric: str) -> list[dict]:
    """Tukey HSD pairwise；优先用 statsmodels，缺失时回退到 scipy.stats.tukey_hsd。"""
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        sub = df[["level", metric]].dropna()
        res = pairwise_tukeyhsd(sub[metric].to_numpy(), sub["level"].to_numpy(), alpha=0.05)
        out = []
        for row in res._results_table.data[1:]:
            out.append({
                "group1": str(row[0]),
                "group2": str(row[1]),
                "mean_diff": float(row[2]),
                "p_adj": float(row[3]),
                "lower": float(row[4]),
                "upper": float(row[5]),
                "reject": bool(row[6]),
            })
        return out
    except ImportError:
        groups = {lvl: df.loc[df["level"] == lvl, metric].dropna().to_numpy() for lvl in LEVEL_ORDER}
        res = stats.tukey_hsd(*[groups[l] for l in LEVEL_ORDER])
        out = []
        for i, l1 in enumerate(LEVEL_ORDER):
            for j, l2 in enumerate(LEVEL_ORDER):
                if j <= i:
                    continue
                p = float(res.pvalue[i, j])
                out.append({
                    "group1": l1,
                    "group2": l2,
                    "mean_diff": float(groups[l2].mean() - groups[l1].mean()),
                    "p_adj": p,
                    "reject": bool(p < 0.05),
                })
        return out


def plot_box(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df.loc[df["level"] == lvl, metric].dropna().to_numpy() for lvl in LEVEL_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    labels=[LEVEL_LABEL[l] for l in LEVEL_ORDER])
    for patch, lvl in zip(bp["boxes"], LEVEL_ORDER):
        patch.set_facecolor(LEVEL_COLOR[lvl])
        patch.set_alpha(0.55)
    for i, (lvl, values) in enumerate(zip(LEVEL_ORDER, data), start=1):
        x = np.random.default_rng(0).normal(i, 0.04, size=len(values))
        ax.scatter(x, values, color=LEVEL_COLOR[lvl], edgecolor="black",
                   linewidth=0.4, s=20, zorder=3, alpha=0.85)
    ax.set_ylabel(metric)
    ax.set_title(f"Discriminative validity boxplot ({metric})")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_bar(summary: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(summary))
    means = summary["mean"].to_numpy()
    stds = summary["std"].to_numpy()
    colors = [LEVEL_COLOR[l] for l in summary["level"]]
    ax.bar(xs, means, yerr=stds, capsize=6, color=colors, alpha=0.75, edgecolor="black")
    for x, m in zip(xs, means):
        ax.text(x, m + 1.0, f"{m:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([LEVEL_LABEL[l] for l in summary["level"]])
    ax.set_ylabel(metric)
    ax.set_title(f"Mean ± SD by level ({metric})")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_report(output_dir: Path, summary: pd.DataFrame, anova: dict, tukey: list[dict],
                 metric: str, df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# 区分效度实验报告")
    lines.append("")
    lines.append(f"主指标：`{metric}`（句子级评分，0–100，越高越接近标准发音）")
    lines.append("")
    lines.append("## 1. 三组样本描述")
    lines.append("")
    lines.append("| Level | 含义 | n | mean | std | min | median | max |")
    lines.append("|-------|------|---|------|-----|-----|--------|-----|")
    for _, r in summary.iterrows():
        lines.append(f"| {r['level']} | {r['label']} | {r['n']} | {r['mean']:.2f} | "
                     f"{r['std']:.2f} | {r['min']:.2f} | {r['median']:.2f} | {r['max']:.2f} |")
    lines.append("")
    lines.append("## 2. 单因素 ANOVA")
    lines.append("")
    lines.append(f"- F({anova['df_between']}, {anova['df_within']}) = {anova['F']:.4f}")
    lines.append(f"- p = {anova['p_value']:.4e}")
    lines.append(f"- SS_between = {anova['ss_between']:.4f}")
    lines.append(f"- SS_within  = {anova['ss_within']:.4f}")
    lines.append(f"- η² = {anova['eta_squared']:.4f}（>0.14 视为大效应；预期 > 0.5）")
    lines.append(f"- ω² = {anova['omega_squared']:.4f}")
    lines.append("")
    lines.append("## 3. Tukey HSD 事后检验")
    lines.append("")
    lines.append("| group1 | group2 | mean_diff (g2-g1) | p_adj | reject H₀ |")
    lines.append("|--------|--------|-------------------|-------|-----------|")
    for t in tukey:
        lines.append(f"| {t['group1']} | {t['group2']} | {t['mean_diff']:+.3f} | "
                     f"{t['p_adj']:.4e} | {'YES' if t['reject'] else 'no'} |")
    lines.append("")
    lines.append("## 4. 与预期对照")
    lines.append("")
    lines.append("| Level | 预期区间 | 实测均值 | 是否落在预期内 |")
    lines.append("|-------|----------|----------|----------------|")
    expectations = {"A": (95, 100), "B": (60, 90), "C": (20, 50)}
    for _, r in summary.iterrows():
        lo, hi = expectations[r["level"]]
        in_range = "✓" if lo <= r["mean"] <= hi else "✗"
        lines.append(f"| {r['level']} | [{lo}, {hi}] | {r['mean']:.2f} | {in_range} |")
    lines.append("")
    lines.append("## 5. 结论模板")
    lines.append("")
    monotone = (summary.loc[summary['level'] == 'A', 'mean'].iloc[0] >
                summary.loc[summary['level'] == 'B', 'mean'].iloc[0] >
                summary.loc[summary['level'] == 'C', 'mean'].iloc[0])
    lines.append(f"- 三组均值梯度 A > B > C：{'成立' if monotone else '不成立'}")
    lines.append(f"- ANOVA 高度显著（p < 0.001）：{'是' if anova['p_value'] < 1e-3 else '否'}")
    lines.append(f"- 效应量 η² > 0.5（大效应）：{'是' if anova['eta_squared'] > 0.5 else '否'}")
    all_reject = all(t["reject"] for t in tukey)
    lines.append(f"- 所有 Tukey 两两比较均显著：{'是' if all_reject else '否'}")
    lines.append("")
    lines.append("## 6. 附：所有样本明细")
    lines.append("")
    lines.append("详见 `per_sample.csv`。")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utterance-csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metric", default=PRIMARY_METRIC)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.utterance_csv, args.manifest)
    per_sample_cols = ["level", "pair_id", "reference_id", "learner_id", args.metric] + [
        m for m in SECONDARY_METRICS if m in df.columns
    ]
    df[per_sample_cols].to_csv(args.output_dir / "per_sample.csv", index=False, encoding="utf-8-sig")

    summary = describe_by_level(df, args.metric)
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    anova = anova_eta_squared(df, args.metric)
    tukey = tukey_hsd(df, args.metric)

    stats_blob = {"metric": args.metric, "anova": anova, "tukey_hsd": tukey,
                  "summary": summary.to_dict(orient="records")}
    secondary_blob = {}
    for m in SECONDARY_METRICS:
        if m in df.columns:
            secondary_blob[m] = {
                "summary": describe_by_level(df, m).to_dict(orient="records"),
                "anova": anova_eta_squared(df, m),
            }
    stats_blob["secondary_metrics"] = secondary_blob

    (args.output_dir / "stats.json").write_text(
        json.dumps(stats_blob, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_box(df, args.metric, plots_dir / "boxplot.png")
    plot_bar(summary, args.metric, plots_dir / "bar_mean_sd.png")

    write_report(args.output_dir, summary, anova, tukey, args.metric, df)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nANOVA: F={anova['F']:.4f}, p={anova['p_value']:.4e}, η²={anova['eta_squared']:.4f}")
    print("Tukey HSD:")
    for t in tukey:
        print(f"  {t['group1']} vs {t['group2']}: diff={t['mean_diff']:+.3f}, p_adj={t['p_adj']:.4e}, reject={t['reject']}")
    print(f"\n[done] results -> {args.output_dir}")


if __name__ == "__main__":
    main()
