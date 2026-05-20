#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def safe_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def sanitize_filename(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        return "EMPTY"
    for ch in '/\\:*?"<>|':
        cleaned = cleaned.replace(ch, "_")
    return cleaned


def load_reference_text(reference_dir: Path, reference_id: str) -> str:
    txt_path = reference_dir / f"{reference_id}.txt"
    if not txt_path.exists():
        return ""
    return txt_path.read_text(encoding="utf-8").strip()


def relative_score_within_group(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(index=series.index, dtype=float)
    min_value = float(valid.min())
    max_value = float(valid.max())
    if max_value <= min_value:
        return pd.Series(100.0, index=series.index, dtype=float)
    scores = 100.0 * (max_value - numeric) / (max_value - min_value)
    return scores.clip(lower=0.0, upper=100.0)


def normalize_segment_scores(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["relative_score_in_utterance"] = (
        out.groupby(group_keys)["dtw_norm_distance_dimnorm"]
        .transform(relative_score_within_group)
    )
    return out


def pick_weak_words(word_df: pd.DataFrame, max_words: int, score_threshold: float) -> list[dict]:
    usable = word_df[
        (word_df["status"] == "ok")
        & (word_df["labels_match"].fillna(False))
    ].copy()
    if usable.empty:
        return []

    usable = usable.sort_values(
        ["relative_score_in_utterance", "dtw_norm_distance_dimnorm"],
        ascending=[True, False],
    )
    weak = usable[usable["relative_score_in_utterance"] < score_threshold]
    if weak.empty:
        return []
    else:
        weak = weak.head(max_words)

    rows = []
    for _, row in weak.iterrows():
        rows.append(
            {
                "word": str(row["ref_label"]),
                "word_label_learner": str(row["learner_label"]),
                "seg_idx": safe_int(row["seg_idx"]),
                "word_score_relative": safe_float(row["relative_score_in_utterance"]),
                "word_dtw_norm_distance_dimnorm": safe_float(row["dtw_norm_distance_dimnorm"]),
                "ref_start": safe_float(row["ref_start"]),
                "ref_end": safe_float(row["ref_end"]),
                "learner_start": safe_float(row["learner_start"]),
                "learner_end": safe_float(row["learner_end"]),
            }
        )
    return rows


def pick_strong_words(word_df: pd.DataFrame, max_words: int) -> list[dict]:
    usable = word_df[
        (word_df["status"] == "ok")
        & (word_df["labels_match"].fillna(False))
    ].copy()
    if usable.empty:
        return []
    usable = usable.sort_values(
        ["relative_score_in_utterance", "dtw_norm_distance_dimnorm"],
        ascending=[False, True],
    ).head(max_words)

    rows = []
    for _, row in usable.iterrows():
        rows.append(
            {
                "word": str(row["ref_label"]),
                "word_score_relative": safe_float(row["relative_score_in_utterance"]),
                "word_dtw_norm_distance_dimnorm": safe_float(row["dtw_norm_distance_dimnorm"]),
            }
        )
    return rows


def attach_weak_phones(
    weak_words: list[dict],
    phone_df: pd.DataFrame,
    max_phones_per_word: int,
    phone_score_threshold: float,
) -> list[dict]:
    if not weak_words:
        return weak_words

    usable = phone_df[
        (phone_df["status"] == "ok")
        & (phone_df["labels_match"].fillna(False))
    ].copy()
    if usable.empty:
        for word in weak_words:
            word["weak_phones"] = []
        return weak_words

    usable["phone_mid"] = (usable["ref_start"].astype(float) + usable["ref_end"].astype(float)) / 2.0

    for word in weak_words:
        start = word.get("ref_start")
        end = word.get("ref_end")
        if start is None or end is None:
            word["weak_phones"] = []
            continue

        phones_in_word = usable[
            (usable["phone_mid"] >= float(start))
            & (usable["phone_mid"] <= float(end))
        ].copy()

        if phones_in_word.empty:
            word["weak_phones"] = []
            continue

        phones_in_word["relative_score_in_word"] = relative_score_within_group(
            phones_in_word["dtw_norm_distance_dimnorm"]
        )
        phones_in_word = phones_in_word.sort_values(
            ["relative_score_in_word", "dtw_norm_distance_dimnorm"],
            ascending=[True, False],
        )

        weak = phones_in_word[phones_in_word["relative_score_in_word"] < phone_score_threshold]
        if weak.empty:
            word["weak_phones"] = []
            continue
        else:
            weak = weak.head(max_phones_per_word)

        phone_rows = []
        for _, row in weak.iterrows():
            phone_rows.append(
                {
                    "phone": str(row["ref_label"]),
                    "phone_label_learner": str(row["learner_label"]),
                    "seg_idx": safe_int(row["seg_idx"]),
                    "phone_score_relative": safe_float(row["relative_score_in_word"]),
                    "phone_dtw_norm_distance_dimnorm": safe_float(row["dtw_norm_distance_dimnorm"]),
                    "ref_start": safe_float(row["ref_start"]),
                    "ref_end": safe_float(row["ref_end"]),
                }
            )
        word["weak_phones"] = phone_rows
    return weak_words


def build_record(
    utterance_row: pd.Series,
    word_df: pd.DataFrame,
    phone_df: pd.DataFrame,
    reference_dir: Path,
    sentence_threshold: float,
    word_threshold: float,
    phone_threshold: float,
    max_words: int,
    max_phones_per_word: int,
) -> dict:
    reference_id = str(utterance_row["reference_id"])
    learner_id = str(utterance_row["learner_id"])
    sentence_score = safe_float(utterance_row.get("sentence_score_new_relative"))
    sentence_needs_diagnosis = sentence_score is not None and sentence_score < sentence_threshold

    pair_words = word_df[
        (word_df["reference_id"] == reference_id)
        & (word_df["learner_id"] == learner_id)
    ].copy()
    pair_phones = phone_df[
        (phone_df["reference_id"] == reference_id)
        & (phone_df["learner_id"] == learner_id)
    ].copy()

    weak_words = []
    if sentence_needs_diagnosis:
        weak_words = pick_weak_words(pair_words, max_words=max_words, score_threshold=word_threshold)
        weak_words = attach_weak_phones(
            weak_words,
            pair_phones,
            max_phones_per_word=max_phones_per_word,
            phone_score_threshold=phone_threshold,
        )

    strong_words = pick_strong_words(pair_words, max_words=2)

    notes: list[str] = []
    if sentence_needs_diagnosis:
        notes.append("Sentence score is below the diagnosis threshold; inspect weak words and weak phones.")
        if not weak_words:
            notes.append("Sentence needs review, but no stable weak-word evidence was extracted; check audio manually.")
    else:
        notes.append("Sentence score passed the threshold; do not drill down for negative diagnosis by default.")

    return {
        "pair_id": str(utterance_row["pair_id"]),
        "reference_id": reference_id,
        "learner_id": learner_id,
        "reference_text": load_reference_text(reference_dir, reference_id),
        "status": str(utterance_row["status"]),
        "diagnosis_policy": {
            "sentence_threshold": sentence_threshold,
            "word_threshold": word_threshold,
            "phone_threshold": phone_threshold,
            "sentence_metric": "sentence_score_new_relative",
            "word_metric": "relative_score_in_utterance",
            "phone_metric": "relative_score_in_word",
        },
        "sentence": {
            "score_relative": sentence_score,
            "dtw_norm_distance_dimnorm": safe_float(utterance_row.get("sentence_dtw_norm_distance_dimnorm")),
            "needs_diagnosis": sentence_needs_diagnosis,
            "matched_word_segments": safe_int(utterance_row.get("matched_word_segments")),
            "matched_phone_segments": safe_int(utterance_row.get("matched_phone_segments")),
        },
        "drill_down_level": "word_phone" if sentence_needs_diagnosis else "sentence_only",
        "strong_words": strong_words,
        "weak_words": weak_words,
        "notes": notes,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build structured JSONL input for LLM-based pronunciation diagnosis."
    )
    parser.add_argument("--utterance-scores", type=Path, required=True, help="Path to utterance_scores.csv")
    parser.add_argument("--word-scores", type=Path, required=True, help="Path to word_scores.csv")
    parser.add_argument("--phone-scores", type=Path, required=True, help="Path to phone_scores.csv")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Directory containing reference .txt files")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--sentence-threshold", type=float, default=85.0, help="Sentence score threshold for further diagnosis")
    parser.add_argument("--word-threshold", type=float, default=75.0, help="Word relative score threshold")
    parser.add_argument("--phone-threshold", type=float, default=70.0, help="Phone relative score threshold")
    parser.add_argument("--max-words", type=int, default=2, help="Maximum weak words per utterance")
    parser.add_argument("--max-phones-per-word", type=int, default=2, help="Maximum weak phones per weak word")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    utterance_df = pd.read_csv(args.utterance_scores, encoding="utf-8-sig")
    word_df = pd.read_csv(args.word_scores, encoding="utf-8-sig")
    phone_df = pd.read_csv(args.phone_scores, encoding="utf-8-sig")

    word_df = normalize_segment_scores(word_df, ["reference_id", "learner_id"])

    records = []
    for _, row in utterance_df.iterrows():
        records.append(
            build_record(
                row,
                word_df=word_df,
                phone_df=phone_df,
                reference_dir=args.reference_dir,
                sentence_threshold=args.sentence_threshold,
                word_threshold=args.word_threshold,
                phone_threshold=args.phone_threshold,
                max_words=args.max_words,
                max_phones_per_word=args.max_phones_per_word,
            )
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for record in records:
        learner_id = sanitize_filename(str(record.get("learner_id", "")))
        output_json = args.output_jsonl.parent / f"diagnosis_{learner_id}.json"
        output_json.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Saved {len(records)} diagnosis records to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
