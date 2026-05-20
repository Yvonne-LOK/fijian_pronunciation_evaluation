#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from compare_dtw_across_learners import (
    build_row_order,
    extract_distance,
    find_latest_run_per_learner,
    load_sentence_json,
    print_selected_runs,
    resolve_path,
)


QUESTION_JSON_PATTERN = re.compile(r"^dtw_(?P<learner>.+)_question_(?P<sentence_no>\d{3})\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DTW distances across learners for all available sentences and save them into one CSV."
    )
    parser.add_argument(
        "--eval-runs-dir",
        type=str,
        default="/mnt/d/workdir/outputs/eval_runs",
        help="Root eval_runs directory in WSL path format.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional output csv path. Defaults to <eval_runs_dir>/dtw_comparison_all_sentences.csv",
    )
    parser.add_argument(
        "--score-output-csv",
        type=str,
        default=None,
        help="Optional output csv path for score comparison. Defaults to <eval_runs_dir>/score_comparison_all_sentences.csv",
    )
    return parser.parse_args()


def collect_sentence_numbers(latest_runs: dict[str, tuple[str, Path]]) -> list[str]:
    sentence_numbers: set[str] = set()
    for _, run_dir in latest_runs.values():
        dtw_dir = run_dir / "dtw_distances"
        if not dtw_dir.exists():
            continue
        for json_path in dtw_dir.glob("dtw_*.json"):
            match = QUESTION_JSON_PATTERN.match(json_path.name)
            if match:
                sentence_numbers.add(match.group("sentence_no"))
    return sorted(sentence_numbers)


def extract_metric(
    sentence_json: dict,
    row_type: str,
    word_key: str | None,
    phone_key: str | None,
    *,
    metric_kind: str,
) -> float | None:
    if metric_kind == "distance":
        if row_type == "sentence":
            return sentence_json.get("sentence_distance")
        if row_type == "word" and word_key is not None:
            return sentence_json["words"].get(word_key, {}).get("word_distance")
        if row_type == "phone" and word_key is not None and phone_key is not None:
            return (
                sentence_json["words"]
                .get(word_key, {})
                .get("phones_distances", {})
                .get(phone_key, {})
                .get("phone_distance")
            )
        return None

    if metric_kind == "score":
        if row_type == "sentence":
            return sentence_json.get("sentence_score")
        if row_type == "word" and word_key is not None:
            return sentence_json["words"].get(word_key, {}).get("word_score")
        if row_type == "phone" and word_key is not None and phone_key is not None:
            return (
                sentence_json["words"]
                .get(word_key, {})
                .get("phones_distances", {})
                .get(phone_key, {})
                .get("phone_score")
            )
        return None

    raise ValueError(f"Unsupported metric kind: {metric_kind}")


def build_sentence_dataframe(
    sentence_no: str,
    sentence_jsons: dict[str, dict],
    *,
    metric_kind: str,
) -> pd.DataFrame:
    reference_learner = sorted(sentence_jsons.keys())[0]
    reference_json = sentence_jsons[reference_learner]
    row_order = build_row_order(reference_json)

    table: dict[str, list] = {
        "sentence_no": [sentence_no] * len(row_order),
        "sentence_text": [reference_json["sentence_text"]] * len(row_order),
        "row_type": [row_type for row_type, _, _, _ in row_order],
        "segment": [row_name for _, _, _, row_name in row_order],
    }
    for learner in sorted(sentence_jsons.keys()):
        sentence_json = sentence_jsons[learner]
        table[learner] = [
            extract_metric(
                sentence_json,
                row_type,
                word_key,
                phone_key,
                metric_kind=metric_kind,
            )
            for row_type, word_key, phone_key, _ in row_order
        ]
    return pd.DataFrame(table)


def sentence_json_has_scores(sentence_json: dict) -> bool:
    if sentence_json.get("sentence_score") is None:
        return False

    for word_entry in sentence_json.get("words", {}).values():
        if not isinstance(word_entry, dict):
            continue
        if word_entry.get("word_score") is None:
            return False
        for phone_entry in word_entry.get("phones_distances", {}).values():
            if not isinstance(phone_entry, dict):
                continue
            if phone_entry.get("phone_score") is None:
                return False
    return True


def main() -> None:
    args = parse_args()
    eval_runs_dir = resolve_path(args.eval_runs_dir)
    output_csv = (
        resolve_path(args.output_csv)
        if args.output_csv
        else eval_runs_dir / "dtw_comparison_all_sentences.csv"
    )
    score_output_csv = (
        resolve_path(args.score_output_csv)
        if args.score_output_csv
        else eval_runs_dir / "score_comparison_all_sentences.csv"
    )

    latest_runs = find_latest_run_per_learner(eval_runs_dir)
    if not latest_runs:
        raise SystemExit(f"No learner run directories found in {eval_runs_dir}")

    print_selected_runs(latest_runs)

    sentence_numbers = collect_sentence_numbers(latest_runs)
    if not sentence_numbers:
        raise SystemExit(f"No DTW sentence JSON files found under latest runs in {eval_runs_dir}")

    print(f"[INFO] Found {len(sentence_numbers)} sentence numbers: {', '.join(sentence_numbers)}")

    all_distance_frames: list[pd.DataFrame] = []
    all_score_frames: list[pd.DataFrame] = []
    score_available_for_all_sentences = True
    for sentence_no in sentence_numbers:
        sentence_jsons: dict[str, dict] = {}
        missing: list[str] = []

        for learner, (_, run_dir) in sorted(latest_runs.items()):
            try:
                sentence_jsons[learner] = load_sentence_json(run_dir, learner, sentence_no)
            except FileNotFoundError as exc:
                missing.append(str(exc))

        if not sentence_jsons:
            print(f"[WARN] Skip sentence {sentence_no}: no learner JSON found")
            continue

        if missing:
            for item in missing:
                print(f"[WARN] {item}")

        all_distance_frames.append(
            build_sentence_dataframe(sentence_no, sentence_jsons, metric_kind="distance")
        )

        if all(sentence_json_has_scores(sentence_json) for sentence_json in sentence_jsons.values()):
            all_score_frames.append(
                build_sentence_dataframe(sentence_no, sentence_jsons, metric_kind="score")
            )
        else:
            score_available_for_all_sentences = False

    if not all_distance_frames:
        raise SystemExit("No comparison rows were generated.")

    distance_df = pd.concat(all_distance_frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    distance_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved all-sentence DTW comparison CSV to: {output_csv}")

    if score_available_for_all_sentences and all_score_frames:
        score_df = pd.concat(all_score_frames, ignore_index=True)
        score_output_csv.parent.mkdir(parents=True, exist_ok=True)
        score_df.to_csv(score_output_csv, index=False, encoding="utf-8-sig")
        print(f"Saved all-sentence score comparison CSV to: {score_output_csv}")
    else:
        print("[INFO] Score fields were not complete across all selected JSON files, so score comparison CSV was skipped.")


if __name__ == "__main__":
    main()
