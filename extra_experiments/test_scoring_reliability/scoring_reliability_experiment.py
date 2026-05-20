"""
评分复测信度实验 - 主实验脚本

从 6 位斐济语学习者的 60 条录音中随机抽取 20 条样本，
对每个样本独立运行完整 MFA-GFCC-DTW 流水线 5 次，
得到 20 × 5 = 100 个评分供后续 ICC 分析。

用法:
    # 完整流水线（每次重跑 MFA 对齐，需要 MFA 已安装）
    python scoring_reliability_experiment.py

    # 跳过 MFA，使用已有 TextGrid（适用于 MFA 未安装的环境）
    python scoring_reliability_experiment.py --skip-mfa

    # 指定重复次数
    python scoring_reliability_experiment.py --runs 5
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 路径配置（相对于项目根目录自动定位，兼容 Windows/WSL）
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = SCRIPT_DIR / "results"

REFERENCE_CORPUS_DIR    = PROJECT_ROOT / "data" / "fijian" / "raw_corpus"
LEARNER_PCM_ROOT        = PROJECT_ROOT / "data" / "fijian" / "learner_pcm"
REFERENCE_ALIGNMENT_DIR = PROJECT_ROOT / "data" / "fijian" / "aligned" / "reference"
EXISTING_LEARNER_ALIGNED_ROOT = PROJECT_ROOT / "data" / "fijian" / "aligned" / "learner"

DICTIONARY_PATH   = PROJECT_ROOT / "data" / "fijian" / "fijian_bootstrap_arpa.dict"
ACOUSTIC_MODEL    = "english_us_arpa"

PIPELINE_RUN_DIR  = PROJECT_ROOT / "scripts" / "mfa_pipeline" / "run"
PIPELINE_SCRIPT   = PIPELINE_RUN_DIR / "batch_gfcc_dtw_eval.py"

# ---------------------------------------------------------------------------
# 实验参数
# ---------------------------------------------------------------------------

N_SAMPLES    = 20    # 随机抽取的样本对数量
N_RUNS       = 5     # 默认重复测量次数
SEED_SAMPLE  = 42    # 抽样随机种子，保证结果可复现

SAMPLE_RATE  = 16000
GFCC_NFILTS  = 40
GFCC_NCEPS   = 13
DTW_ALPHA    = 8.0
DTW_BETA     = 0.6

# pairs.csv 中的对应关系：学习者问题编号 → 参考音频 ID
QUESTION_TO_REFERENCE: dict[str, str] = {
    "001": "u1_l1_001",
    "002": "u1_l1_003",
    "003": "u1_l1_004",
    "004": "u1_l1_007",
    "005": "u1_l1_008",
    "006": "u1_l1_009",
    "007": "u1_l1_010",
    "008": "u1_l1_011",
    "009": "u1_l1_012",
    "010": "u1_l1_013",
}

LEARNERS = [
    "guoziyu",
    "jiangshuman",
    "luoyumeng",
    "xuhaiyue",
    "zengjianbin",
    "zhangle",
]

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def get_python_bin() -> str:
    return sys.executable


def get_mfa_bin() -> str:
    for candidate in ["mfa", "mfa.exe"]:
        found = shutil.which(candidate)
        if found:
            return found
    return "mfa"


def build_all_pairs() -> list[dict[str, str]]:
    """枚举 6×10=60 个候选样本对，只保留 WAV、参考 TextGrid 均存在的条目。"""
    pairs: list[dict[str, str]] = []
    for learner in LEARNERS:
        for q_num, ref_id in QUESTION_TO_REFERENCE.items():
            learner_id  = f"{learner}_question_{q_num}"
            learner_wav = LEARNER_PCM_ROOT / f"{learner}_recordings" / f"{learner_id}.wav"
            ref_wav     = REFERENCE_CORPUS_DIR / f"{ref_id}.wav"
            ref_tg      = REFERENCE_ALIGNMENT_DIR / f"{ref_id}.TextGrid"
            if learner_wav.exists() and ref_wav.exists() and ref_tg.exists():
                pairs.append(
                    {
                        "pair_id":      f"{learner}__q{q_num}",
                        "learner":      learner,
                        "question":     q_num,
                        "learner_id":   learner_id,
                        "reference_id": ref_id,
                    }
                )
    return pairs


def sample_pairs(
    all_pairs: list[dict[str, str]],
    n: int,
    seed: int,
) -> list[dict[str, str]]:
    """按固定种子随机抽取 n 对，并按学习者+问题编号排序，便于阅读。"""
    rng = random.Random(seed)
    selected = rng.sample(all_pairs, n)
    return sorted(selected, key=lambda x: (x["learner"], x["question"]))


# ---------------------------------------------------------------------------
# MFA 对齐
# ---------------------------------------------------------------------------

def prepare_mfa_corpus(
    selected_pairs: list[dict[str, str]],
    learner: str,
    corpus_dir: Path,
) -> None:
    """
    为指定学习者准备 MFA 语料目录。
    每条录音需要: {learner_id}.wav（来自 learner_pcm）+ {learner_id}.txt（来自参考语料）。
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    learner_pcm_dir = LEARNER_PCM_ROOT / f"{learner}_recordings"
    for pair in selected_pairs:
        if pair["learner"] != learner:
            continue
        learner_id  = pair["learner_id"]
        reference_id = pair["reference_id"]
        src_wav = learner_pcm_dir / f"{learner_id}.wav"
        src_txt = REFERENCE_CORPUS_DIR / f"{reference_id}.txt"
        if not src_wav.exists():
            raise FileNotFoundError(f"学习者 WAV 不存在: {src_wav}")
        if not src_txt.exists():
            raise FileNotFoundError(f"参考文本 TXT 不存在: {src_txt}")
        shutil.copy2(src_wav, corpus_dir / f"{learner_id}.wav")
        shutil.copy2(src_txt, corpus_dir / f"{learner_id}.txt")


def run_mfa_align(corpus_dir: Path, output_dir: Path) -> bool:
    """调用 MFA 对齐，返回是否成功。"""
    mfa = get_mfa_bin()
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        mfa, "align",
        str(corpus_dir),
        str(DICTIONARY_PATH),
        ACOUSTIC_MODEL,
        str(output_dir),
        "--clean",
    ]
    print(f"    [MFA] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"    [WARNING] MFA 退出码非零: {result.returncode}", flush=True)
        return False
    return True


# ---------------------------------------------------------------------------
# GFCC + DTW 评分
# ---------------------------------------------------------------------------

def build_scoring_pair_csv(
    selected_pairs: list[dict[str, str]],
    learner_aligned_root: Path,
    output_path: Path,
) -> list[dict[str, str]]:
    """
    构造传给 batch_gfcc_dtw_eval.py 的 pairs CSV，
    同时返回实际有 TextGrid 的有效对列表。
    """
    valid_pairs: list[dict[str, str]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "reference_id", "learner_id"])
        writer.writeheader()
        for pair in selected_pairs:
            tg_path = (
                learner_aligned_root / pair["learner"] / f"{pair['learner_id']}.TextGrid"
            )
            if not tg_path.exists():
                print(
                    f"    [SKIP] 缺少 TextGrid，跳过: {tg_path}",
                    flush=True,
                )
                continue
            writer.writerow(
                {
                    "pair_id":      pair["pair_id"],
                    "reference_id": pair["reference_id"],
                    "learner_id":   pair["learner_id"],
                }
            )
            valid_pairs.append(pair)
    return valid_pairs


def build_merged_learner_dirs(
    valid_pairs: list[dict[str, str]],
    learner_aligned_root: Path,
    learner_wav_dir: Path,
    merged_aligned_dir: Path,
) -> None:
    """
    将各学习者的 WAV 和 TextGrid 汇聚到统一目录，
    batch_gfcc_dtw_eval.py 依赖单一的 learner_dir 和 learner_aligned_dir。
    """
    learner_wav_dir.mkdir(parents=True, exist_ok=True)
    merged_aligned_dir.mkdir(parents=True, exist_ok=True)
    for pair in valid_pairs:
        learner    = pair["learner"]
        learner_id = pair["learner_id"]
        # WAV
        src_wav = LEARNER_PCM_ROOT / f"{learner}_recordings" / f"{learner_id}.wav"
        dst_wav = learner_wav_dir / f"{learner_id}.wav"
        if src_wav.exists() and not dst_wav.exists():
            shutil.copy2(src_wav, dst_wav)
        # TextGrid
        src_tg = learner_aligned_root / learner / f"{learner_id}.TextGrid"
        dst_tg = merged_aligned_dir / f"{learner_id}.TextGrid"
        if src_tg.exists():
            shutil.copy2(src_tg, dst_tg)


def run_gfcc_dtw_scoring(
    selected_pairs: list[dict[str, str]],
    learner_aligned_root: Path,
    run_dir: Path,
) -> Path:
    """
    调用 batch_gfcc_dtw_eval.py 对本次所有选定样本对评分，
    返回 utterance_scores.csv 的路径。
    """
    pair_csv      = run_dir / "pairs_for_scoring.csv"
    learner_wav_dir  = run_dir / "learner_wavs"
    merged_tg_dir    = run_dir / "learner_aligned_merged"
    scores_dir       = run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    valid_pairs = build_scoring_pair_csv(selected_pairs, learner_aligned_root, pair_csv)
    if not valid_pairs:
        raise RuntimeError("没有找到任何有效的 TextGrid，无法评分")

    build_merged_learner_dirs(valid_pairs, learner_aligned_root, learner_wav_dir, merged_tg_dir)

    python = get_python_bin()
    cmd = [
        python, str(PIPELINE_SCRIPT),
        "--reference-dir",      str(REFERENCE_CORPUS_DIR),
        "--learner-dir",        str(learner_wav_dir),
        "--reference-aligned-dir", str(REFERENCE_ALIGNMENT_DIR),
        "--learner-aligned-dir",   str(merged_tg_dir),
        "--output-dir",         str(scores_dir),
        "--pair-csv",           str(pair_csv),
        "--sample-rate",        str(SAMPLE_RATE),
        "--nfilts",             str(GFCC_NFILTS),
        "--nceps",              str(GFCC_NCEPS),
        "--alpha",              str(DTW_ALPHA),
        "--beta",               str(DTW_BETA),
    ]
    print(f"    [DTW] 评分 {len(valid_pairs)} 对...", flush=True)

    # 把 pipeline run 目录加入 PYTHONPATH，使 cut_phone_segments 等可被 import
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PIPELINE_RUN_DIR)
        + (os.pathsep + existing_pp if existing_pp else "")
    )
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"    [WARNING] 评分脚本退出码非零: {result.returncode}", flush=True)

    return scores_dir / "utterance_scores.csv"


# ---------------------------------------------------------------------------
# 单次迭代
# ---------------------------------------------------------------------------

def run_one_iteration(
    run_idx: int,
    total_runs: int,
    selected_pairs: list[dict[str, str]],
    run_dir: Path,
    skip_mfa: bool,
) -> Path:
    """
    执行一次完整的 MFA + GFCC + DTW 评测。
    返回本次 utterance_scores.csv 的路径。
    """
    sep = "=" * 60
    print(f"\n{sep}", flush=True)
    print(f"Run {run_idx}/{total_runs}", flush=True)
    print(f"{sep}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    if skip_mfa:
        learner_aligned_root = EXISTING_LEARNER_ALIGNED_ROOT
        print("  [MFA] 已跳过，使用现有 TextGrid", flush=True)
    else:
        learner_aligned_root = run_dir / "aligned" / "learner"
        learners_in_run = sorted({p["learner"] for p in selected_pairs})
        for learner in learners_in_run:
            corpus_dir  = run_dir / "mfa_input" / learner
            aligned_dir = learner_aligned_root / learner
            print(f"  [MFA] 对齐: {learner}", flush=True)
            prepare_mfa_corpus(selected_pairs, learner, corpus_dir)
            run_mfa_align(corpus_dir, aligned_dir)

    scores_csv = run_gfcc_dtw_scoring(selected_pairs, learner_aligned_root, run_dir)
    print(f"  [DONE] 评分已保存: {scores_csv}", flush=True)
    return scores_csv


# ---------------------------------------------------------------------------
# 合并结果
# ---------------------------------------------------------------------------

def merge_all_runs(
    run_score_csvs: list[tuple[int, Path]],
) -> pd.DataFrame:
    """
    把 N 次运行的 utterance_scores.csv 合并成一张长表，
    增加 run 列标识第几次测量。
    """
    keep_cols = [
        "pair_id", "reference_id", "learner_id", "run", "status",
        "sentence_score", "phone_mean_score", "word_mean_score", "overall_score",
        "sentence_dtw_norm_distance", "sentence_dtw_norm_distance_dimnorm",
    ]
    frames: list[pd.DataFrame] = []
    for run_idx, csv_path in run_score_csvs:
        if not csv_path.exists():
            print(f"[WARNING] Run {run_idx} 缺少评分 CSV: {csv_path}", flush=True)
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df["run"] = run_idx
        frames.append(df)

    if not frames:
        raise RuntimeError("没有任何运行产生有效的评分 CSV")

    merged = pd.concat(frames, ignore_index=True)
    available = [c for c in keep_cols if c in merged.columns]
    return merged[available].copy()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="评分复测信度实验：对 20 个样本对重复评测 N 次",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-mfa",
        action="store_true",
        help="跳过 MFA 对齐步骤，使用已有 TextGrid（MFA 未安装时使用）",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=N_RUNS,
        help=f"重复测量次数（默认 {N_RUNS}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_SAMPLE,
        help=f"抽样随机种子（默认 {SEED_SAMPLE}）",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 项目根目录: {PROJECT_ROOT}", flush=True)
    print(f"[INFO] 结果目录:   {RESULTS_DIR}", flush=True)
    if args.skip_mfa:
        print("[INFO] 模式: 跳过 MFA（使用已有 TextGrid）", flush=True)
    else:
        print("[INFO] 模式: 完整流水线（每次重跑 MFA 对齐）", flush=True)

    # Step 1: 构建候选对
    print("\n>>> Step 1: 枚举所有可用样本对", flush=True)
    all_pairs = build_all_pairs()
    print(f"    共找到 {len(all_pairs)} 个有效样本对", flush=True)
    if len(all_pairs) < N_SAMPLES:
        sys.exit(
            f"[ERROR] 有效样本对不足 {N_SAMPLES} 个（仅 {len(all_pairs)} 个），请检查数据路径"
        )

    # Step 2: 随机抽取 20 对
    print(f"\n>>> Step 2: 随机抽取 {N_SAMPLES} 对（seed={args.seed}）", flush=True)
    selected = sample_pairs(all_pairs, N_SAMPLES, args.seed)
    selected_csv = RESULTS_DIR / "selected_pairs.csv"
    pd.DataFrame(selected).to_csv(selected_csv, index=False, encoding="utf-8-sig")
    print(f"    抽样结果已保存: {selected_csv}", flush=True)
    for i, p in enumerate(selected, 1):
        print(
            f"    {i:2d}. {p['pair_id']:30s}  ref={p['reference_id']}",
            flush=True,
        )

    # Step 3: 重复测量 N 次
    print(f"\n>>> Step 3: 开始 {args.runs} 次重复测量", flush=True)
    run_score_csvs: list[tuple[int, Path]] = []
    for run_idx in range(1, args.runs + 1):
        run_dir = RESULTS_DIR / f"run_{run_idx}"
        scores_csv = run_one_iteration(
            run_idx=run_idx,
            total_runs=args.runs,
            selected_pairs=selected,
            run_dir=run_dir,
            skip_mfa=args.skip_mfa,
        )
        run_score_csvs.append((run_idx, scores_csv))

    # Step 4: 合并结果
    print("\n>>> Step 4: 合并所有运行结果", flush=True)
    merged_df = merge_all_runs(run_score_csvs)
    all_scores_csv = RESULTS_DIR / "all_scores.csv"
    merged_df.to_csv(all_scores_csv, index=False, encoding="utf-8-sig")
    n_ok = (merged_df["status"] == "ok").sum() if "status" in merged_df.columns else len(merged_df)
    print(f"    合并评分已保存: {all_scores_csv}", flush=True)
    print(
        f"    共 {len(merged_df)} 行（{args.runs} runs × {N_SAMPLES} pairs），"
        f"其中 {n_ok} 行状态 OK",
        flush=True,
    )

    print(f"\n[DONE] 主实验完成。", flush=True)
    print(f"    下一步: python analyze_scoring_reliability.py", flush=True)


if __name__ == "__main__":
    main()
