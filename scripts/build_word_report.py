"""
生成 Word 文档报告：特征提取的抗噪稳健性对比实验

读取 summary_v2.csv + plots/*.png，输出 report.docx 到 extra_experiments/noise_robustness/

依赖：pip install python-docx
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ------------------------------------------------------------------
# 路径
# ------------------------------------------------------------------
if Path("/mnt/d/workdir").exists():
    OUT_ROOT = Path("/mnt/d/workdir/extra_experiments/noise_robustness")
else:
    OUT_ROOT = Path(r"D:\workdir\extra_experiments\noise_robustness")

SUMMARY_CSV = OUT_ROOT / "results" / "summary_v2.csv"
PLOT_DIR = OUT_ROOT / "results" / "plots"
OUT_DOCX = OUT_ROOT / "report.docx"

# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------
def set_cell_shading(cell, fill_hex: str):
    """给单元格加底色（python-docx 没有原生支持，要写 XML）"""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tc_pr.append(shd)


def add_code_block(doc: Document, text: str):
    """段落使用等宽字体 + 浅灰底色，模拟代码块"""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(10)
    # 段落底色
    p_pr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F2F2F2")
    p_pr.append(shd)
    return p


def style_table_header(row, fill_hex="D5E8F0"):
    for cell in row.cells:
        set_cell_shading(cell, fill_hex)
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True


def add_table_from_rows(doc: Document, rows, col_widths_cm=None,
                        header_fill="D5E8F0"):
    """
    rows[0] 为表头，后续行为数据。
    col_widths_cm: 列宽列表（cm），None 则均分。
    """
    tbl = doc.add_table(rows=len(rows), cols=len(rows[0]))
    tbl.style = "Light Grid Accent 1"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(10)
                    if i == 0:
                        r.bold = True
    if col_widths_cm:
        for j, w in enumerate(col_widths_cm):
            for i in range(len(rows)):
                tbl.cell(i, j).width = Cm(w)
    style_table_header(tbl.rows[0], header_fill)
    return tbl


def heading_with_size(doc: Document, text: str, level: int = 1):
    h = doc.add_heading(text, level=level)
    # 强制把字号设大些，避免默认字体太小
    for r in h.runs:
        r.font.name = "Microsoft YaHei"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        if level == 0:
            r.font.size = Pt(20)
        elif level == 1:
            r.font.size = Pt(16)
        elif level == 2:
            r.font.size = Pt(14)
        r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    return h


def add_para(doc: Document, text: str, *, italic=False, bold=False,
             align=None, font_size=11):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(font_size)
    run.italic = italic
    run.bold = bold
    return p


def add_image(doc: Document, png_path: Path, width_cm: float = 14.0,
              caption: str | None = None):
    if not png_path.exists():
        add_para(doc, f"[图片缺失: {png_path}]", italic=True)
        return
    doc.add_picture(str(png_path), width=Cm(width_cm))
    # 居中
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        cap = add_para(doc, caption, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
                       font_size=10)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"找不到 {SUMMARY_CSV}，请先跑 analyze_noise_robustness.py")
    df = pd.read_csv(SUMMARY_CSV)

    def row(feat, snr):
        r = df[(df["feature"] == feat) & (df["snr"] == snr)].iloc[0]
        return r

    doc = Document()

    # 设默认字体
    style = doc.styles["Normal"]
    style.font.name = "Microsoft YaHei"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    # 页边距
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # =============== 标题 ===============
    heading_with_size(doc, "特征提取的抗噪稳健性对比实验：GFCC vs MFCC", level=0)
    add_para(doc, "实验报告 · 北外多语语音评测研究 · 2026-05",
             italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, font_size=10)
    doc.add_paragraph()

    # =============== 1. 实验目的 ===============
    heading_with_size(doc, "1. 实验目的", level=1)
    add_para(doc,
        "验证在移动学习真实噪声环境下，GFCC（Gammatone Frequency Cepstral Coefficients）"
        "相比 MFCC（Mel Frequency Cepstral Coefficients）能否更稳定地维持 DTW 句子级评分，"
        "从而为口语评测系统选择特征提取器提供经验依据。"
    )

    # =============== 2. 数据与流程 ===============
    heading_with_size(doc, "2. 数据与流程", level=1)

    add_para(doc, "数据来源", bold=True)
    add_para(doc,
        "从北外多语语音评测项目 Fijian 母语者朗读语料库（static/fijian/u1_l1_XXX.wav）"
        "随机抽取 30 个句子作为标准音频。随机种子 seed=42，保证可复现。"
    )

    add_para(doc, "噪声生成", bold=True)
    add_para(doc,
        "对每条句子的副本（即“完美朗读”的学习者音频）注入加性高斯白噪声，"
        "SNR 取 clean / 20 dB / 10 dB / 5 dB 四档，每条产生 4 个版本，共 120 个测试音频。公式："
    )
    add_code_block(doc,
        "signal_power = mean(x ** 2)\n"
        "noise_power  = signal_power / 10 ** (SNR_dB / 10)\n"
        "noise        = sqrt(noise_power) * randn(len(x))\n"
        "x_noisy      = x + noise"
    )
    add_para(doc, "噪声随机种子 numpy.default_rng(2026)。")

    add_para(doc, "特征与评分", bold=True)
    add_para(doc, "两种特征采用对称实现，保证比较公平：")

    add_table_from_rows(doc, [
        ["参数", "GFCC", "MFCC"],
        ["采样率", "16 kHz", "16 kHz"],
        ["窗长 / 帧移", "25 ms / 10 ms", "25 ms / 10 ms"],
        ["滤波器组通道数", "32（gammatone）", "32（mel）"],
        ["倒谱系数维数", "13", "13"],
        ["后处理", "log + DCT + z-norm", "log + DCT + z-norm"],
        ["距离", "DTW + 欧氏距离（路径归一化）", "同左"],
    ], col_widths_cm=[3.5, 5.5, 5.5])
    doc.add_paragraph()

    add_para(doc, "评分映射", bold=True)
    add_para(doc,
        "原参考脚本的指数映射 100·exp(−8·d/0.6) 是基于词级片段标定的，"
        "在句子级 DTW 距离区间（典型 d∈[2.5, 4.0]）会指数下溢到零，无法分辨不同 SNR。"
        "本实验改用线性截断："
    )
    add_code_block(doc, "score = clip(100 · (1 − d / D_REF), 0, 100)，D_REF = 4.0")
    add_para(doc,
        "D_REF=4.0 略大于观测最大值（max d ≈ 3.88），使 clean 自比≈100、"
        "噪声条件下分数仍有可读动态范围。GFCC 与 MFCC 共用同一套映射参数。"
    )

    # =============== 3. 评测指标 ===============
    heading_with_size(doc, "3. 评测指标", level=1)
    add_para(doc,
        "主指标采用归一化 DTW 距离（norm_distance，直接反映特征空间内"
        "“标准 vs 加噪学习者”的差异，不受映射形状影响）；"
        "辅助指标为线性映射后的分数及其衰减/标准差。"
    )

    add_table_from_rows(doc, [
        ["指标", "含义", "方向"],
        ["距离均值", "30 句的 norm_distance 均值", "越低 → 越稳健"],
        ["距离标准差", "30 句的 norm_distance 标准差", "越低 → 跨句子越稳定"],
        ["分数衰减", "clean_score − noisy_score 的均值", "越小 → 越稳健"],
        ["分数标准差", "30 句的线性分数标准差", "越小 → 越稳定"],
    ], col_widths_cm=[3.0, 7.0, 4.5])
    doc.add_paragraph()

    # =============== 4. 结果 ===============
    heading_with_size(doc, "4. 结果", level=1)

    # ---- 4.1 ----
    heading_with_size(doc, "4.1 主指标：归一化 DTW 距离", level=2)

    def fmt(x, n=3):
        return f"{float(x):.{n}f}"

    table_rows = [["SNR", "GFCC dist_mean", "GFCC dist_std",
                   "MFCC dist_mean", "MFCC dist_std", "MFCC / GFCC"]]
    for snr, label in [("clean", "Clean"), ("snr20", "20 dB"),
                       ("snr10", "10 dB"), ("snr05", "5 dB")]:
        g = row("gfcc", snr)
        m = row("mfcc", snr)
        ratio = m["dist_mean"] / g["dist_mean"]
        table_rows.append([
            label, fmt(g["dist_mean"]), fmt(g["dist_std"]),
            fmt(m["dist_mean"]), fmt(m["dist_std"]), fmt(ratio, n=3),
        ])
    add_table_from_rows(doc, table_rows,
                        col_widths_cm=[2.0, 2.7, 2.7, 2.7, 2.7, 2.2])
    doc.add_paragraph()

    add_image(doc, PLOT_DIR / "distance_mean_vs_snr.png", width_cm=14,
              caption="图 1：归一化 DTW 距离均值随 SNR 变化（越低越稳健）")
    add_image(doc, PLOT_DIR / "distance_std_vs_snr.png", width_cm=14,
              caption="图 2：30 句子的距离标准差随 SNR 变化")

    # ---- 4.2 ----
    heading_with_size(doc, "4.2 辅助指标：线性映射分数与分数衰减", level=2)
    table_rows = [["SNR", "GFCC score_mean", "GFCC score_std",
                   "MFCC score_mean", "MFCC score_std",
                   "GFCC drop", "MFCC drop"]]
    for snr, label in [("clean", "Clean"), ("snr20", "20 dB"),
                       ("snr10", "10 dB"), ("snr05", "5 dB")]:
        g = row("gfcc", snr)
        m = row("mfcc", snr)
        table_rows.append([
            label,
            fmt(g["score_mean"], 2), fmt(g["score_std"], 2),
            fmt(m["score_mean"], 2), fmt(m["score_std"], 2),
            "—" if snr == "clean" else fmt(g["score_drop"], 2),
            "—" if snr == "clean" else fmt(m["score_drop"], 2),
        ])
    add_table_from_rows(doc, table_rows,
                        col_widths_cm=[1.8, 2.4, 2.4, 2.4, 2.4, 1.9, 1.9])
    doc.add_paragraph()

    add_image(doc, PLOT_DIR / "score_drop_vs_snr.png", width_cm=14,
              caption="图 3：线性分数衰减（clean − noisy）随 SNR 变化")
    add_image(doc, PLOT_DIR / "score_std_vs_snr.png", width_cm=14,
              caption="图 4：30 句子分数标准差随 SNR 变化")

    # =============== 5. 分析与结论 ===============
    heading_with_size(doc, "5. 分析与结论", level=1)

    add_para(doc, "1. clean 自比验证。", bold=True)
    add_para(doc,
        "clean 条件下，GFCC 和 MFCC 的分数均接近 100（97.88 / 98.04），"
        "归一化距离均小于 0.09。残差距离来自加噪管线中 PCM_16 重存的量化噪声，"
        "并不影响后续比较——两种特征面对该量化噪声的响应基本对称。"
    )

    add_para(doc, "2. GFCC 在所有 SNR 下都比 MFCC 更稳健。", bold=True)
    add_para(doc,
        "在 20 / 10 / 5 dB 三种噪声条件下，MFCC 的归一化 DTW 距离均比 GFCC 高出 "
        "4.9% / 6.7% / 7.7%，即“特征空间内偏离干净参考的程度”更大。"
        "对应线性分数衰减：MFCC 比 GFCC 多衰减 3.9 / 5.6 / 6.6 分。"
        "该差异随噪声变强而单调放大，说明 GFCC 的优势不是噪声水平依赖的偶然现象，"
        "而是在 SNR 越差时越显著——这正是移动学习场景（地铁、室外、教室回声）"
        "最需要的特性。"
    )

    add_para(doc, "3. 跨句子的稳定性几近持平，GFCC 略优于 MFCC。", bold=True)
    add_para(doc,
        "噪声条件下两特征的距离标准差都在 0.10–0.15 区间，无显著差距；"
        "分数标准差也都在 2.6–3.8 分区间。这意味着 GFCC 的优势主要来自"
        "更小的“平均偏移”而非“更小的方差”"
        "——它把噪声带来的扰动整体压低，而不是只对部分句子有效。"
    )

    add_para(doc, "4. 关于评分映射。", bold=True)
    add_para(doc,
        "原参考脚本的指数映射 α=8, β=0.6 是为词级片段标定的，在本句子级实验中"
        "完全无法区分 20/10/5 dB 三档；改用 D_REF=4 的线性映射后，"
        "分数动态范围被合理保留，clean 自比≈98。"
        "该映射可作为后续句子级评测的默认值。"
    )

    add_para(doc, "5. 实践含义。", bold=True)
    add_para(doc,
        "对于移动学习应用中的口语评分系统，建议优先采用 GFCC 作为前端特征："
        "在中等到较强的环境噪声下（10–5 dB），它能把分数误差压低 5–7 分，"
        "对应“勉强及格 vs 不及格”这种关键阈值上的判断稳定性。"
        "若需要在更广噪声条件下部署，还可考虑在 GFCC 基础上叠加语音增强或"
        "噪声鲁棒训练，但仅特征选择本身已能带来可观收益。"
    )

    # =============== 6. 文件清单 ===============
    heading_with_size(doc, "6. 文件清单", level=1)
    add_code_block(doc,
        "extra_experiments/noise_robustness/\n"
        "├── audio/                                  # 120 wav (30 句 × 4 SNR)\n"
        "│   ├── clean/   snr20/   snr10/   snr05/\n"
        "├── results/\n"
        "│   ├── scores_detail.csv                   # 240 行评分明细\n"
        "│   ├── summary_v2.csv                      # 本报告主表\n"
        "│   ├── selected_sentences.txt              # 选中的 30 个句子\n"
        "│   └── plots/\n"
        "│       ├── distance_mean_vs_snr.png\n"
        "│       ├── distance_std_vs_snr.png\n"
        "│       ├── score_drop_vs_snr.png\n"
        "│       └── score_std_vs_snr.png\n"
        "├── report.md\n"
        "└── report.docx"
    )

    add_para(doc, "复现命令", bold=True)
    add_code_block(doc,
        "python scripts/noise_robustness_experiment.py   "
        "# 一次性产生 120 wav + scores_detail.csv\n"
        "python scripts/analyze_noise_robustness.py      "
        "# 从 scores_detail 重算 summary_v2 + plots\n"
        "python scripts/build_word_report.py             "
        "# 生成 report.docx（本文档）"
    )

    # 保存
    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_DOCX))
    print(f"saved {OUT_DOCX}")


if __name__ == "__main__":
    main()
