#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
import os
from pathlib import Path

import pandas as pd


RUN_DIR_PATTERN = re.compile(r"^learner_(?P<learner>.+)_(?P<stamp>\d{8}_\d{6})$")


def resolve_path(path_str: str) -> Path:
    raw = path_str.strip()
    if os.name != "nt" and re.match(r"^[A-Za-z]:\\", raw):
        drive = raw[0].lower()
        suffix = raw[2:].replace("\\", "/")
        return Path(f"/mnt/{drive}{suffix}")
    if os.name == "nt" and raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5].upper()
        suffix = raw[6:].replace("/", "\\")
        return Path(f"{drive}:{suffix}")
    return Path(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DTW distances across learners for one sentence number."
    )
    parser.add_argument(
        "sentence_no",
        help="Sentence number like 004, used to read dtw_<learner>_question_004.json from each learner's latest run.",
    )
    parser.add_argument(
        "--eval-runs-dir",
        type=str,
        default="/mnt/d/workdir/outputs/eval_runs" if os.name != "nt" else r"d:\workdir\outputs\eval_runs",
        help="Root eval_runs directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional output csv path. Defaults to <eval_runs_dir>/dtw_comparison_<sentence_no>.csv",
    )
    return parser.parse_args()


def normalize_sentence_no(sentence_no: str) -> str:
    digits = sentence_no.strip()
    if not digits.isdigit():
        raise ValueError(f"Sentence number must be digits, got: {sentence_no}")
    return digits.zfill(3)


def find_latest_run_per_learner(eval_runs_dir: Path) -> dict[str, tuple[str, Path]]:
    latest: dict[str, tuple[str, Path]] = {}
    for path in eval_runs_dir.iterdir():
        if not path.is_dir():
            continue
        match = RUN_DIR_PATTERN.match(path.name)
        if not match:
            continue
        learner = match.group("learner")
        stamp = match.group("stamp")
        current = latest.get(learner)
        if current is None or stamp > current[0]:
            latest[learner] = (stamp, path)
    return latest


def print_selected_runs(latest_runs: dict[str, tuple[str, Path]]) -> None:
    print("[INFO] Selected latest run per learner:")
    for learner, (stamp, run_dir) in sorted(latest_runs.items()):
        print(f"[INFO]   learner={learner} stamp={stamp} run_dir={run_dir}")


def load_sentence_json(run_dir: Path, learner: str, sentence_no: str) -> dict:
    json_path = run_dir / "dtw_distances" / f"dtw_{learner}_question_{sentence_no}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing DTW json: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def build_phone_row_labels(sentence_json: dict) -> list[tuple[str, str, str]]:
    phones: list[tuple[str, str, str]] = []
    for word_key, word_entry in sentence_json["words"].items():
        for phone_key, phone_entry in word_entry["phones_distances"].items():
            phones.append((word_key, phone_key, str(phone_entry.get("mapping", "")).strip()))

    arpa_base_counts = Counter(phone_key.split("__", 1)[0] for _, phone_key, _ in phones)
    seen: Counter[str] = Counter()
    row_labels: list[tuple[str, str, str]] = []
    for word_key, phone_key, mapping in phones:
        arpa_base = phone_key.split("__", 1)[0]
        seen[arpa_base] += 1
        if arpa_base_counts[arpa_base] > 1:
            row_name = f"{arpa_base}_{seen[arpa_base]}"
        else:
            row_name = arpa_base
        if mapping:
            row_name = f"{row_name}({mapping})"
        row_labels.append((word_key, phone_key, row_name))
    return row_labels


def build_row_order(reference_json: dict) -> list[tuple[str, str | None, str | None, str]]:
    rows: list[tuple[str, str | None, str | None, str]] = []
    rows.append(("sentence", None, None, reference_json["sentence_text"]))

    for word_key in reference_json["words"].keys():
        rows.append(("word", word_key, None, word_key))

    for word_key, phone_key, row_name in build_phone_row_labels(reference_json):
        rows.append(("phone", word_key, phone_key, row_name))

    return rows


def extract_distance(sentence_json: dict, row_type: str, word_key: str | None, phone_key: str | None) -> float | None:
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


def build_comparison_dataframe(sentence_jsons: dict[str, dict]) -> pd.DataFrame:
    reference_learner = sorted(sentence_jsons.keys())[0]
    row_order = build_row_order(sentence_jsons[reference_learner])

    table = {"segment": [row_name for _, _, _, row_name in row_order]}
    for learner in sorted(sentence_jsons.keys()):
        sentence_json = sentence_jsons[learner]
        table[learner] = [
            extract_distance(sentence_json, row_type, word_key, phone_key)
            for row_type, word_key, phone_key, _ in row_order
        ]
    return pd.DataFrame(table)


def main() -> None:
    args = parse_args()
    sentence_no = normalize_sentence_no(args.sentence_no)
    eval_runs_dir = resolve_path(args.eval_runs_dir)
    output_csv = resolve_path(args.output_csv) if args.output_csv else (eval_runs_dir / f"dtw_comparison_{sentence_no}.csv")

    latest_runs = find_latest_run_per_learner(eval_runs_dir)
    if not latest_runs:
        raise SystemExit(f"No learner run directories found in {eval_runs_dir}")
    print_selected_runs(latest_runs)

    sentence_jsons: dict[str, dict] = {}
    missing = []
    for learner, (_, run_dir) in sorted(latest_runs.items()):
        try:
            sentence_jsons[learner] = load_sentence_json(run_dir, learner, sentence_no)
        except FileNotFoundError as exc:
            missing.append(str(exc))

    if not sentence_jsons:
        raise SystemExit("No DTW sentence JSON files were found for the requested sentence number.")
    if missing:
        for item in missing:
            print(f"[WARN] {item}")

    df = build_comparison_dataframe(sentence_jsons)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved DTW comparison CSV to: {output_csv}")


if __name__ == "__main__":
    main()
