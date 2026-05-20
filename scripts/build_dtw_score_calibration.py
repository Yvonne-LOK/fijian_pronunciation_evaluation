#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean, pstdev


def iter_input_paths(input_globs: list[str]) -> list[Path]:
    paths: dict[Path, None] = {}
    for pattern in input_globs:
        for matched in glob.glob(pattern, recursive=True):
            path = Path(matched)
            if path.is_file():
                paths[path.resolve()] = None
    return sorted(paths.keys())


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_distances(records: list[dict]) -> tuple[list[float], list[float], list[float]]:
    sentence_distances: list[float] = []
    word_distances: list[float] = []
    phone_distances: list[float] = []

    for record in records:
        sentence_distance = safe_float(record.get("sentence_distance"))
        if sentence_distance is not None:
            sentence_distances.append(sentence_distance)

        words = record.get("words", {})
        if not isinstance(words, dict):
            continue

        for word_entry in words.values():
            if not isinstance(word_entry, dict):
                continue

            word_distance = safe_float(word_entry.get("word_distance"))
            if word_distance is not None:
                word_distances.append(word_distance)

            phones = word_entry.get("phones_distances", {})
            if not isinstance(phones, dict):
                continue

            for phone_entry in phones.values():
                if not isinstance(phone_entry, dict):
                    continue
                phone_distance = safe_float(phone_entry.get("phone_distance"))
                if phone_distance is not None:
                    phone_distances.append(phone_distance)

    return sentence_distances, word_distances, phone_distances


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "count": 0,
        }
    if len(values) == 1:
        return {
            "mean": float(values[0]),
            "std": 0.0,
            "count": 1,
        }
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "count": int(len(values)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build calibration statistics for DTW distance JSONL files."
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        required=True,
        help="Glob pattern for dtw_distances.jsonl files. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON path for calibration statistics.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    input_paths = iter_input_paths(args.input_glob)
    if not input_paths:
        raise FileNotFoundError("No input files matched the provided --input-glob patterns.")

    sentence_distances: list[float] = []
    word_distances: list[float] = []
    phone_distances: list[float] = []

    for path in input_paths:
        records = load_jsonl(path)
        sentence_values, word_values, phone_values = collect_distances(records)
        sentence_distances.extend(sentence_values)
        word_distances.extend(word_values)
        phone_distances.extend(phone_values)

    output = {
        "sentence": summarize(sentence_distances),
        "word": summarize(word_distances),
        "phone": summarize(phone_distances),
        "source_files": [str(path) for path in input_paths],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved DTW calibration stats to: {args.output_json}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
