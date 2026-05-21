"""区分效度实验（Discriminative Validity Experiment）

构造三个发音质量梯度（每级 20 条句子样本），用 MFA-GFCC-DTW 流水线打分：
  Level A: 标准音频与自身配对（自比，预期 ≈ 100）
  Level B: 真实学习者按正确文本朗读（预期分布较广）
  Level C: 学习者音频与不同句子的标准音频错配（预期显著偏低）

关键约束：batch_gfcc_dtw_eval.py 的 sentence_score_new_relative 走的是
batch 内 10/90 分位相对映射，必须把 60 个 pair 合并到一次评测中才能跨 level 比较。

用法（任意目录下都可）：
    python extra_experiments/discriminative_validity_experiment.py
    python extra_experiments/discriminative_validity_experiment.py --skip-batch  # 仅做分析

WSL / Windows 兼容：路径全部用 pathlib，相对仓库根定位，不写死盘符。
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_CORPUS_DIR = REPO_ROOT / "data" / "fijian" / "raw_corpus"
REFERENCE_ALIGNMENT_DIR = REPO_ROOT / "data" / "fijian" / "aligned" / "reference"
LEARNER_AUDIO_ROOT = REPO_ROOT / "data" / "fijian" / "learner_pcm"
LEARNER_ALIGNMENT_ROOT = REPO_ROOT / "data" / "fijian" / "aligned" / "learner"

BATCH_EVAL_SCRIPT = REPO_ROOT / "scripts" / "mfa_pipeline" / "run" / "batch_gfcc_dtw_eval.py"

EXP_ROOT = REPO_ROOT / "extra_experiments" / "discriminative_validity"
STAGING_DIR = EXP_ROOT / "staging"
STAGING_REF_DIR = STAGING_DIR / "reference"
STAGING_LEARNER_DIR = STAGING_DIR / "learner"
RUN_DIR = EXP_ROOT / "raw_run"
RESULT_DIR = EXP_ROOT / "results"

LEARNERS = ["guoziyu", "jiangshuman", "luoyumeng", "xuhaiyue", "zengjianbin", "zhangle"]

# learner question_NNN -> reference_id（与 scripts/mfa_pipeline/run/pairs.csv 一致）
QUESTION_TO_REF: dict[str, str] = {
    "question_001": "u1_l1_001",
    "question_002": "u1_l1_003",
    "question_003": "u1_l1_004",
    "question_004": "u1_l1_007",
    "question_005": "u1_l1_008",
    "question_006": "u1_l1_009",
    "question_007": "u1_l1_010",
    "question_008": "u1_l1_011",
    "question_009": "u1_l1_012",
    "question_010": "u1_l1_013",
}
LEARNER_QUESTIONS = list(QUESTION_TO_REF.keys())  # 10 questions
LEARNER_REF_IDS = list(QUESTION_TO_REF.values())  # 10 refs

# Level A 用 20 条 reference 全集
LEVEL_A_REF_IDS = [f"u1_l1_{i:03d}" for i in range(1, 21)]

SEED = 42

LEVEL_A_TAG = "levelA"
LEVEL_B_TAG = "levelB"
LEVEL_C_TAG = "levelC"


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------
def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def stage_reference_assets() -> None:
    """把所有需要的 reference wav+TextGrid 拷贝到 staging/reference/。"""
    STAGING_REF_DIR.mkdir(parents=True, exist_ok=True)
    for ref_id in LEVEL_A_REF_IDS:
        wav_src = REFERENCE_CORPUS_DIR / f"{ref_id}.wav"
        tg_src = REFERENCE_ALIGNMENT_DIR / f"{ref_id}.TextGrid"
        if not wav_src.exists():
            raise FileNotFoundError(f"missing reference wav: {wav_src}")
        if not tg_src.exists():
            raise FileNotFoundError(f"missing reference TextGrid: {tg_src}")
        _safe_copy(wav_src, STAGING_REF_DIR / f"{ref_id}.wav")
        _safe_copy(tg_src, STAGING_REF_DIR / f"{ref_id}.TextGrid")


def stage_level_a_learner_assets(ref_ids: list[str]) -> list[tuple[str, str, str]]:
    """Level A：自比。把每条 ref 复制为 learner-side 文件，文件名加 _selfA 后缀。

    返回 (pair_id, reference_id, learner_id) 列表。
    """
    rows: list[tuple[str, str, str]] = []
    for ref_id in ref_ids:
        learner_id = f"{ref_id}_selfA"
        wav_src = REFERENCE_CORPUS_DIR / f"{ref_id}.wav"
        tg_src = REFERENCE_ALIGNMENT_DIR / f"{ref_id}.TextGrid"
        _safe_copy(wav_src, STAGING_LEARNER_DIR / f"{learner_id}.wav")
        _safe_copy(tg_src, STAGING_LEARNER_DIR / f"{learner_id}.TextGrid")
        pair_id = f"{LEVEL_A_TAG}__{ref_id}"
        rows.append((pair_id, ref_id, learner_id))
    return rows


def stage_learner_assets(learner_questions: list[tuple[str, str]]) -> None:
    """把指定 (learner_name, question_id) 的 wav+TG 拷贝到 staging/learner/。

    使用扁平命名 {learner_name}_{question_id}.wav，与原始命名一致，不会冲突。
    """
    seen: set[str] = set()
    for learner_name, question_id in learner_questions:
        key = f"{learner_name}_{question_id}"
        if key in seen:
            continue
        seen.add(key)
        wav_src = LEARNER_AUDIO_ROOT / f"{learner_name}_recordings" / f"{key}.wav"
        tg_src = LEARNER_ALIGNMENT_ROOT / learner_name / f"{key}.TextGrid"
        if not wav_src.exists():
            raise FileNotFoundError(f"missing learner wav: {wav_src}")
        if not tg_src.exists():
            raise FileNotFoundError(f"missing learner TextGrid: {tg_src}")
        _safe_copy(wav_src, STAGING_LEARNER_DIR / f"{key}.wav")
        _safe_copy(tg_src, STAGING_LEARNER_DIR / f"{key}.TextGrid")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_level_b_pairs(rng: random.Random, per_question: int = 2) -> list[tuple[str, str, str]]:
    """Level B：句子均匀。每个 question 从 6 个学习者里随机选 per_question 个。

    返回 (pair_id, reference_id, learner_id) 列表。
    """
    rows: list[tuple[str, str, str]] = []
    counter = 0
    for question_id in LEARNER_QUESTIONS:
        chosen = rng.sample(LEARNERS, per_question)
        ref_id = QUESTION_TO_REF[question_id]
        for learner_name in chosen:
            counter += 1
            learner_id = f"{learner_name}_{question_id}"
            pair_id = f"{LEVEL_B_TAG}__{counter:02d}__{learner_id}"
            rows.append((pair_id, ref_id, learner_id))
    return rows


def sample_level_c_pairs(rng: random.Random, per_question: int = 2) -> list[tuple[str, str, str]]:
    """Level C：内容错配。学习者音频 X 配一条不同句子的 reference Y (Y ≠ X)。

    句子均匀：每个 question 选 per_question 个学习者，每条配一条不同的 reference。
    """
    rows: list[tuple[str, str, str]] = []
    counter = 0
    for question_id in LEARNER_QUESTIONS:
        true_ref = QUESTION_TO_REF[question_id]
        wrong_pool = [r for r in LEVEL_A_REF_IDS if r != true_ref]
        chosen_learners = rng.sample(LEARNERS, per_question)
        chosen_wrong_refs = rng.sample(wrong_pool, per_question)
        for learner_name, wrong_ref in zip(chosen_learners, chosen_wrong_refs):
            counter += 1
            learner_id = f"{learner_name}_{question_id}"
            pair_id = f"{LEVEL_C_TAG}__{counter:02d}__{learner_id}__vs__{wrong_ref}"
            rows.append((pair_id, wrong_ref, learner_id))
    return rows


# ---------------------------------------------------------------------------
# Pairs CSV + manifest
# ---------------------------------------------------------------------------
def write_pairs_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "reference_id", "learner_id"])
        for r in rows:
            writer.writerow(r)


def write_manifest(path: Path, all_rows: dict[str, list[tuple[str, str, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "pair_id", "reference_id", "learner_id"])
        for level, rows in all_rows.items():
            for r in rows:
                writer.writerow([level, *r])


# ---------------------------------------------------------------------------
# Run batch evaluation
# ---------------------------------------------------------------------------
def run_batch_eval(pair_csv: Path, output_dir: Path, python_bin: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(BATCH_EVAL_SCRIPT),
        "--reference-dir", str(STAGING_REF_DIR),
        "--learner-dir", str(STAGING_LEARNER_DIR),
        "--reference-aligned-dir", str(STAGING_REF_DIR),
        "--learner-aligned-dir", str(STAGING_LEARNER_DIR),
        "--output-dir", str(output_dir),
        "--pair-csv", str(pair_csv),
        "--sample-rate", "16000",
        "--nfilts", "40",
        "--nceps", "13",
        "--alpha", "8.0",
        "--beta", "0.6",
    ]
    print("[run] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(BATCH_EVAL_SCRIPT.parent))
    if proc.returncode != 0:
        raise SystemExit(f"batch_gfcc_dtw_eval.py exited with {proc.returncode}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter used to invoke batch_gfcc_dtw_eval.py.")
    parser.add_argument("--skip-staging", action="store_true", help="Skip rebuilding the staging directory.")
    parser.add_argument("--skip-batch", action="store_true", help="Skip running batch_gfcc_dtw_eval.py (reuse existing results).")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip the stats + plot + report step.")
    parser.add_argument("--per-question", type=int, default=2, help="Samples per question for Level B and Level C (default 2 -> 20 pairs each).")
    args = parser.parse_args()

    rng_b = random.Random(SEED)
    rng_c = random.Random(SEED + 1)

    level_a_rows = [(f"{LEVEL_A_TAG}__{rid}", rid, f"{rid}_selfA") for rid in LEVEL_A_REF_IDS]
    level_b_rows = sample_level_b_pairs(rng_b, per_question=args.per_question)
    level_c_rows = sample_level_c_pairs(rng_c, per_question=args.per_question)

    print(f"[info] Level A: {len(level_a_rows)} pairs")
    print(f"[info] Level B: {len(level_b_rows)} pairs")
    print(f"[info] Level C: {len(level_c_rows)} pairs")

    pair_csv_path = RUN_DIR / "pairs.csv"
    manifest_path = RUN_DIR / "manifest.csv"

    if not args.skip_staging:
        _ensure_clean_dir(STAGING_REF_DIR)
        _ensure_clean_dir(STAGING_LEARNER_DIR)
        stage_reference_assets()
        stage_level_a_learner_assets(LEVEL_A_REF_IDS)
        learner_questions_needed = sorted({
            (lid.rsplit("_question_", 1)[0], "question_" + lid.rsplit("_question_", 1)[1])
            for _, _, lid in level_b_rows + level_c_rows
        })
        stage_learner_assets(learner_questions_needed)

    all_rows = level_a_rows + level_b_rows + level_c_rows
    write_pairs_csv(pair_csv_path, all_rows)
    write_manifest(manifest_path, {
        "A": level_a_rows,
        "B": level_b_rows,
        "C": level_c_rows,
    })

    if not args.skip_batch:
        run_batch_eval(pair_csv_path, RUN_DIR, args.python_bin)
    else:
        if not (RUN_DIR / "utterance_scores.csv").exists():
            raise SystemExit(f"--skip-batch set but {RUN_DIR / 'utterance_scores.csv'} does not exist.")

    if not args.skip_analysis:
        analyze_script = Path(__file__).with_name("analyze_discriminative_validity.py")
        cmd = [
            args.python_bin,
            str(analyze_script),
            "--utterance-csv", str(RUN_DIR / "utterance_scores.csv"),
            "--manifest", str(manifest_path),
            "--output-dir", str(RESULT_DIR),
        ]
        print("[run] " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise SystemExit(f"analyze_discriminative_validity.py exited with {proc.returncode}")

    print(f"[done] results -> {RESULT_DIR}")


if __name__ == "__main__":
    main()
