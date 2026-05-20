#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def safe_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = value.strip()
    if not value:
        return None
    return float(value)


def safe_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    value = value.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def load_calibration(path: Path | None) -> dict[str, dict[str, float | int | None]] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for level in ("sentence", "word", "phone"):
        if level not in data or not isinstance(data[level], dict):
            raise ValueError(f"Calibration JSON must contain object field: {level}")
    return data


def compute_score(
    distance: float | None,
    calibration: dict[str, float | int | None] | None,
    *,
    intercept: float,
    slope: float,
) -> float | None:
    if distance is None or calibration is None:
        return None

    mean_value = safe_float(calibration.get("mean"))
    std_value = safe_float(calibration.get("std"))
    if mean_value is None:
        return None
    if std_value is None or std_value <= 0.0:
        return float(max(0.0, min(float(intercept), 100.0)))

    z_score = (float(distance) - mean_value) / std_value
    score = float(intercept - slope * z_score)
    return float(max(0.0, min(score, 100.0)))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_reference_text(reference_dir: Path, reference_id: str) -> str:
    txt_path = reference_dir / f"{reference_id}.txt"
    if not txt_path.exists():
        return ""
    return txt_path.read_text(encoding="utf-8").strip()


def sanitize_filename(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        return "EMPTY"
    for ch in '/\\:*?"<>|':
        cleaned = cleaned.replace(ch, "_")
    return cleaned


def group_rows(rows: list[dict[str, str]], key_fields: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped.setdefault(key, []).append(row)
    return grouped


def make_unique_key(label: str, counts: dict[str, int]) -> str:
    base = label if label else "EMPTY"
    counts[base] = counts.get(base, 0) + 1
    if counts[base] == 1:
        return base
    return f"{base}__{counts[base]}"


MULTI_CHAR_GRAPHEMES = [
    "nq",
    "ng",
    "nd",
    "mb",
    "dr",
    "au",
    "ai",
    "ei",
    "oi",
    "ou",
    "iu",
    "ua",
    "ia",
    "io",
    "ie",
    "oa",
    "oe",
    "ui",
    "uo",
    "ā",
    "ē",
    "ī",
    "ō",
    "ū",
]


def tokenize_word_for_mapping(word: str) -> list[str]:
    lowered = word.strip().lower()
    tokens: list[str] = []
    idx = 0
    while idx < len(lowered):
        matched = None
        for candidate in MULTI_CHAR_GRAPHEMES:
            if lowered.startswith(candidate, idx):
                matched = candidate
                break
        if matched is None:
            matched = lowered[idx]
        tokens.append(matched)
        idx += len(matched)
    return tokens


def distribute_tokens(tokens: list[str], target_count: int) -> list[str]:
    if target_count <= 0:
        return []
    current = list(tokens)
    while len(current) < target_count:
        split_index = next((i for i, token in enumerate(current) if len(token) > 1), None)
        if split_index is None:
            break
        token = current.pop(split_index)
        for ch in reversed(list(token)):
            current.insert(split_index, ch)
    if len(current) < target_count:
        current.extend([""] * (target_count - len(current)))
        return current
    if len(current) == target_count:
        return current

    result: list[str] = []
    start = 0
    total = len(current)
    for i in range(target_count):
        end = round((i + 1) * total / target_count)
        result.append("".join(current[start:end]))
        start = end
    return result


def build_phone_mappings(word: str, phone_labels: list[str]) -> list[str]:
    return distribute_tokens(tokenize_word_for_mapping(word), len(phone_labels))


def word_span_contains_phone(word_row: dict[str, str], phone_row: dict[str, str]) -> bool:
    word_start = safe_float(word_row.get("ref_start"))
    word_end = safe_float(word_row.get("ref_end"))
    phone_start = safe_float(phone_row.get("ref_start"))
    phone_end = safe_float(phone_row.get("ref_end"))
    if None in (word_start, word_end, phone_start, phone_end):
        return False
    phone_mid = (phone_start + phone_end) / 2.0
    return word_start <= phone_mid <= word_end


def build_word_entry(
    word_row: dict[str, str],
    phone_rows: list[dict[str, str]],
    calibration: dict[str, dict[str, float | int | None]] | None,
) -> dict:
    matched_phones = [
        row for row in phone_rows
        if word_span_contains_phone(word_row, row)
    ]
    matched_phones.sort(key=lambda row: int(row["seg_idx"]) if row.get("seg_idx") else -1)
    phone_mappings = build_phone_mappings(
        word_row.get("ref_label", ""),
        [row.get("ref_label", "") for row in matched_phones],
    )

    word_distance = safe_float(word_row.get("dtw_norm_distance_dimnorm"))
    word_entry = {
        "learner_word": word_row.get("learner_label", ""),
        "word_distance": word_distance,
        "word_score": compute_score(
            word_distance,
            None if calibration is None else calibration.get("word"),
            intercept=82.0,
            slope=11.0,
        ),
        "phones_distances": {},
    }
    mismatch = {
        "word_label_mismatch": safe_bool(word_row.get("labels_match")) is False,
        "phone_label_mismatches": [],
        "non_ok_phone_segments": [],
    }

    phone_key_counts: dict[str, int] = {}
    for index, phone_row in enumerate(matched_phones):
        phone_key = make_unique_key(phone_row.get("ref_label", ""), phone_key_counts)
        phone_distance = safe_float(phone_row.get("dtw_norm_distance_dimnorm"))
        phone_entry = {
            "mapping": phone_mappings[index] if index < len(phone_mappings) else "",
            "learner_phone": phone_row.get("learner_label", ""),
            "phone_distance": phone_distance,
            "phone_score": compute_score(
                phone_distance,
                None if calibration is None else calibration.get("phone"),
                intercept=80.0,
                slope=9.0,
            ),
        }
        word_entry["phones_distances"][phone_key] = phone_entry

        if safe_bool(phone_row.get("labels_match")) is False:
            mismatch["phone_label_mismatches"].append(
                {
                    "phone_key": phone_key,
                    "reference_phone": phone_row.get("ref_label", ""),
                    "learner_phone": phone_row.get("learner_label", ""),
                }
            )
        if phone_row.get("status") != "ok":
            mismatch["non_ok_phone_segments"].append(
                {
                    "phone_key": phone_key,
                    "status": phone_row.get("status", ""),
                }
            )

    if mismatch["word_label_mismatch"] or mismatch["phone_label_mismatches"] or mismatch["non_ok_phone_segments"]:
        word_entry["mismatch"] = mismatch
    return word_entry


def build_records(
    reference_dir: Path,
    utterance_rows: list[dict[str, str]],
    word_rows: list[dict[str, str]],
    phone_rows: list[dict[str, str]],
    calibration: dict[str, dict[str, float | int | None]] | None,
) -> list[dict]:
    key_fields = ("pair_id", "reference_id", "learner_id")
    words_by_pair = group_rows(word_rows, key_fields)
    phones_by_pair = group_rows(phone_rows, key_fields)

    records: list[dict] = []
    for utt_row in utterance_rows:
        key = tuple(utt_row[field] for field in key_fields)
        pair_word_rows = sorted(
            words_by_pair.get(key, []),
            key=lambda row: int(row["seg_idx"]) if row.get("seg_idx") else -1,
        )
        pair_phone_rows = phones_by_pair.get(key, [])

        words: dict[str, dict] = {}
        word_key_counts: dict[str, int] = {}
        word_label_mismatches = []
        non_ok_word_segments = []
        for word_row in pair_word_rows:
            word_key = make_unique_key(word_row.get("ref_label", ""), word_key_counts)
            words[word_key] = build_word_entry(word_row, pair_phone_rows, calibration)
            if safe_bool(word_row.get("labels_match")) is False:
                word_label_mismatches.append(
                    {
                        "word_key": word_key,
                        "reference_word": word_row.get("ref_label", ""),
                        "learner_word": word_row.get("learner_label", ""),
                        "seg_idx": int(word_row["seg_idx"]) if word_row.get("seg_idx") else None,
                    }
                )
            if word_row.get("status", "") != "ok":
                non_ok_word_segments.append(
                    {
                        "word_key": word_key,
                        "status": word_row.get("status", ""),
                        "seg_idx": int(word_row["seg_idx"]) if word_row.get("seg_idx") else None,
                    }
                )

        sentence_distance = safe_float(utt_row.get("sentence_dtw_norm_distance_dimnorm"))
        records.append(
            {
                "pair_id": utt_row.get("pair_id", ""),
                "sentence_text": load_reference_text(reference_dir, utt_row.get("reference_id", "")),
                "sentence_distance": sentence_distance,
                "sentence_score": compute_score(
                    sentence_distance,
                    None if calibration is None else calibration.get("sentence"),
                    intercept=86.0,
                    slope=14.0,
                ),
                "words": words,
            }
        )
        summary = {
            "mismatched_word_labels": int(float(utt_row.get("mismatched_word_labels", "0") or 0)),
            "mismatched_phone_labels": int(float(utt_row.get("mismatched_phone_labels", "0") or 0)),
            "missing_word_segments": int(float(utt_row.get("missing_word_segments", "0") or 0)),
            "missing_phone_segments": int(float(utt_row.get("missing_phone_segments", "0") or 0)),
        }
        if (
            utt_row.get("status", "") != "ok"
            or word_label_mismatches
            or non_ok_word_segments
            or any(summary.values())
        ):
            records[-1]["mismatch"] = {
                "sentence_status": utt_row.get("status", ""),
                "word_label_mismatches": word_label_mismatches,
                "non_ok_word_segments": non_ok_word_segments,
                "summary": summary,
            }
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build JSONL with normalized DTW distances for sentence, word, and phone levels."
    )
    parser.add_argument("--reference-dir", type=Path, required=True, help="Directory containing reference txt files")
    parser.add_argument("--utterance-scores", type=Path, required=True, help="Path to utterance_scores.csv")
    parser.add_argument("--word-scores", type=Path, required=True, help="Path to word_scores.csv")
    parser.add_argument("--phone-scores", type=Path, required=True, help="Path to phone_scores.csv")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=None,
        help="Optional calibration JSON with sentence/word/phone mean and std.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    utterance_rows = load_csv_rows(args.utterance_scores)
    word_rows = load_csv_rows(args.word_scores)
    phone_rows = load_csv_rows(args.phone_scores)
    calibration = load_calibration(args.calibration_json)

    records = build_records(
        reference_dir=args.reference_dir,
        utterance_rows=utterance_rows,
        word_rows=word_rows,
        phone_rows=phone_rows,
        calibration=calibration,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for record in records:
        pair_id = str(record.get("pair_id", ""))
        sentence_id = pair_id.split("__", 1)[1] if "__" in pair_id else pair_id
        output_json = args.output_jsonl.parent / f"dtw_{sanitize_filename(sentence_id)}.json"
        output_json.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Saved {len(records)} DTW distance records to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
