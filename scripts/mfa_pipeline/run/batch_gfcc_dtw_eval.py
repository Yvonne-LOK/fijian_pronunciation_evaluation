#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
from fastdtw import fastdtw

from cut_phone_segments import cut_intervals_from_tier as cut_phone_intervals
from cut_word_segments import cut_intervals_from_tier as cut_word_intervals

def load_wav(wav_path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {wav_path}")
    max_abs = np.max(np.abs(y)) + 1e-8
    y = y / max_abs
    return y.astype(np.float32), sr


def z_norm(feats: np.ndarray) -> np.ndarray:
    mean = np.mean(feats, axis=0, keepdims=True)
    std = np.std(feats, axis=0, keepdims=True) + 1e-8
    return (feats - mean) / std


def time_to_sample(t: float, sr: int) -> int:
    return max(0, int(round(t * sr)))


def slice_signal(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    start_idx = time_to_sample(start, sr)
    end_idx = min(len(y), time_to_sample(end, sr))
    if end_idx <= start_idx:
        raise ValueError(f"Invalid slice range: start={start}, end={end}")
    return y[start_idx:end_idx]


def extract_gfcc_from_signal(
    y: np.ndarray,
    sr: int,
    nfilts: int,
    nceps: int,
) -> np.ndarray:
    from spafe.features.gfcc import gfcc

    gfcc_params = inspect.signature(gfcc).parameters
    kwargs = {
        "fs": sr,
        "nfilts": nfilts,
    }
    if "nceps" in gfcc_params:
        kwargs["nceps"] = nceps
    elif "num_ceps" in gfcc_params:
        kwargs["num_ceps"] = nceps
    else:
        raise TypeError("Unsupported spafe.gfcc signature: missing nceps/num_ceps")

    feats = gfcc(y, **kwargs)
    if feats.size == 0:
        raise ValueError("Empty GFCC feature matrix")
    return z_norm(np.asarray(feats, dtype=np.float32))


def extract_gfcc_features(
    wav_path: Path,
    sample_rate: int,
    nfilts: int,
    nceps: int,
) -> np.ndarray:
    y, sr = load_wav(wav_path, target_sr=sample_rate)
    return extract_gfcc_from_signal(y, sr, nfilts, nceps)


def compute_dtw_metrics(
    ref_feats: np.ndarray,
    learner_feats: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[float, float, int, float]:
    distance, path = fastdtw(
        ref_feats,
        learner_feats,
        dist=lambda x, y: np.linalg.norm(x - y),
    )
    path_len = max(len(path), 1)
    norm_distance = float(distance) / path_len
    score = 100.0 * math.exp(-alpha * norm_distance / max(beta, 1e-8))
    return float(distance), norm_distance, path_len, float(np.clip(score, 0.0, 100.0))


def compute_dtw_metrics_dimnorm(
    ref_feats: np.ndarray,
    learner_feats: np.ndarray,
) -> tuple[float, float, int]:
    distance, path = fastdtw(
        ref_feats,
        learner_feats,
        dist=lambda x, y: np.linalg.norm(x - y) / max(len(x), 1),
    )
    path_len = max(len(path), 1)
    norm_distance = float(distance) / path_len
    return float(distance), norm_distance, path_len


def linear_distance_to_score(
    norm_distance: float,
    good_threshold: float = 0.8,
    bad_threshold: float = 4.0,
) -> float:
    if pd.isna(norm_distance):
        return float("nan")
    if bad_threshold <= good_threshold:
        raise ValueError("bad_threshold must be greater than good_threshold")
    score = 100.0 * (bad_threshold - float(norm_distance)) / (bad_threshold - good_threshold)
    return float(np.clip(score, 0.0, 100.0))


def relative_distance_to_score_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    good_threshold = float(valid.quantile(0.10))
    bad_threshold = float(valid.quantile(0.90))
    if bad_threshold <= good_threshold:
        min_value = float(valid.min())
        max_value = float(valid.max())
        if max_value <= min_value:
            return pd.Series(100.0, index=series.index, dtype=float)
        good_threshold = min_value
        bad_threshold = max_value

    def map_one(value: float) -> float:
        if pd.isna(value):
            return float("nan")
        score = 100.0 * (bad_threshold - float(value)) / (bad_threshold - good_threshold)
        return float(np.clip(score, 0.0, 100.0))

    return numeric.apply(map_one)


def score_segment_pair(
    ref_wav: Path,
    learner_wav: Path,
    sample_rate: int,
    nfilts: int,
    nceps: int,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    ref_feats = extract_gfcc_features(ref_wav, sample_rate, nfilts, nceps)
    learner_feats = extract_gfcc_features(learner_wav, sample_rate, nfilts, nceps)
    raw_distance, norm_distance, path_len, score = compute_dtw_metrics(
        ref_feats,
        learner_feats,
        alpha=alpha,
        beta=beta,
    )
    raw_distance_dimnorm, norm_distance_dimnorm, path_len_dimnorm = compute_dtw_metrics_dimnorm(
        ref_feats,
        learner_feats,
    )
    score_new_linear = linear_distance_to_score(norm_distance_dimnorm)
    return {
        "ref_num_frames": int(ref_feats.shape[0]),
        "learner_num_frames": int(learner_feats.shape[0]),
        "feat_dim": int(ref_feats.shape[1]),
        "dtw_raw_distance": raw_distance,
        "dtw_norm_distance": norm_distance,
        "dtw_path_len": path_len,
        "score": score,
        "score_old": score,
        "dtw_raw_distance_dimnorm": raw_distance_dimnorm,
        "dtw_norm_distance_dimnorm": norm_distance_dimnorm,
        "dtw_path_len_dimnorm": path_len_dimnorm,
        "score_new_linear": score_new_linear,
    }


def score_signal_pair(
    ref_signal: np.ndarray,
    learner_signal: np.ndarray,
    sample_rate: int,
    nfilts: int,
    nceps: int,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    ref_feats = extract_gfcc_from_signal(ref_signal, sample_rate, nfilts, nceps)
    learner_feats = extract_gfcc_from_signal(learner_signal, sample_rate, nfilts, nceps)
    raw_distance, norm_distance, path_len, score = compute_dtw_metrics(
        ref_feats,
        learner_feats,
        alpha=alpha,
        beta=beta,
    )
    raw_distance_dimnorm, norm_distance_dimnorm, path_len_dimnorm = compute_dtw_metrics_dimnorm(
        ref_feats,
        learner_feats,
    )
    score_new_linear = linear_distance_to_score(norm_distance_dimnorm)
    return {
        "ref_num_frames": int(ref_feats.shape[0]),
        "learner_num_frames": int(learner_feats.shape[0]),
        "feat_dim": int(ref_feats.shape[1]),
        "dtw_raw_distance": raw_distance,
        "dtw_norm_distance": norm_distance,
        "dtw_path_len": path_len,
        "score": score,
        "score_old": score,
        "dtw_raw_distance_dimnorm": raw_distance_dimnorm,
        "dtw_norm_distance_dimnorm": norm_distance_dimnorm,
        "dtw_path_len_dimnorm": path_len_dimnorm,
        "score_new_linear": score_new_linear,
    }


def build_sentence_metrics(
    ref_wav: Path,
    learner_wav: Path,
    ref_phone_df: pd.DataFrame,
    learner_phone_df: pd.DataFrame,
    sample_rate: int,
    nfilts: int,
    nceps: int,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    ref_signal, _ = load_wav(ref_wav, target_sr=sample_rate)
    learner_signal, _ = load_wav(learner_wav, target_sr=sample_rate)

    raw_metrics = score_signal_pair(
        ref_signal,
        learner_signal,
        sample_rate=sample_rate,
        nfilts=nfilts,
        nceps=nceps,
        alpha=alpha,
        beta=beta,
    )

    ref_trimmed, _ = librosa.effects.trim(ref_signal, top_db=30)
    learner_trimmed, _ = librosa.effects.trim(learner_signal, top_db=30)
    trim_metrics = score_signal_pair(
        ref_trimmed,
        learner_trimmed,
        sample_rate=sample_rate,
        nfilts=nfilts,
        nceps=nceps,
        alpha=alpha,
        beta=beta,
    )

    ref_active = slice_signal(
        ref_signal,
        sample_rate,
        float(ref_phone_df["start"].min()),
        float(ref_phone_df["end"].max()),
    )
    learner_active = slice_signal(
        learner_signal,
        sample_rate,
        float(learner_phone_df["start"].min()),
        float(learner_phone_df["end"].max()),
    )
    active_metrics = score_signal_pair(
        ref_active,
        learner_active,
        sample_rate=sample_rate,
        nfilts=nfilts,
        nceps=nceps,
        alpha=alpha,
        beta=beta,
    )

    return {
        "sentence_score_raw_old": raw_metrics["score_old"],
        "sentence_score_trimmed_old": trim_metrics["score_old"],
        "sentence_score_active_old": active_metrics["score_old"],
        "sentence_score_raw_new_linear": raw_metrics["score_new_linear"],
        "sentence_score_trimmed_new_linear": trim_metrics["score_new_linear"],
        "sentence_score_active_new_linear": active_metrics["score_new_linear"],
        "sentence_score_raw": raw_metrics["score_old"],
        "sentence_score_trimmed": trim_metrics["score_old"],
        "sentence_score_active": active_metrics["score_old"],
        "sentence_score": active_metrics["score_old"],
        "sentence_score_new_linear": active_metrics["score_new_linear"],
        "sentence_dtw_raw_distance": active_metrics["dtw_raw_distance"],
        "sentence_dtw_norm_distance": active_metrics["dtw_norm_distance"],
        "sentence_dtw_path_len": active_metrics["dtw_path_len"],
        "sentence_dtw_raw_distance_dimnorm": active_metrics["dtw_raw_distance_dimnorm"],
        "sentence_dtw_norm_distance_dimnorm": active_metrics["dtw_norm_distance_dimnorm"],
        "sentence_dtw_path_len_dimnorm": active_metrics["dtw_path_len_dimnorm"],
        "sentence_ref_num_frames": active_metrics["ref_num_frames"],
        "sentence_learner_num_frames": active_metrics["learner_num_frames"],
    }


def default_pairs(reference_dir: Path, learner_dir: Path) -> list[dict[str, str]]:
    learner_ids = sorted(wav.stem for wav in learner_dir.glob("*.wav"))
    return [
        {
            "pair_id": learner_id,
            "reference_id": learner_id,
            "learner_id": learner_id,
        }
        for learner_id in learner_ids
    ]


def load_pairs(pair_csv: Path | None, reference_dir: Path, learner_dir: Path) -> list[dict[str, str]]:
    if pair_csv is None:
        return default_pairs(reference_dir, learner_dir)

    df = pd.read_csv(pair_csv)
    required = {"reference_id", "learner_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{pair_csv} must contain columns: {sorted(required)}")

    pairs: list[dict[str, str]] = []
    for _, row in df.iterrows():
        pair_id = str(row["pair_id"]) if "pair_id" in df.columns and pd.notna(row["pair_id"]) else f'{row["reference_id"]}__{row["learner_id"]}'
        pairs.append(
            {
                "pair_id": pair_id,
                "reference_id": str(row["reference_id"]),
                "learner_id": str(row["learner_id"]),
            }
        )
    return pairs


def align_segment_tables(
    ref_df: pd.DataFrame,
    learner_df: pd.DataFrame,
) -> pd.DataFrame:
    ref_df = ref_df.rename(
        columns={
            "utt_id": "ref_utt_id",
            "label": "ref_label",
            "start": "ref_start",
            "end": "ref_end",
            "duration": "ref_duration",
            "wav_path": "ref_wav_path",
        }
    )
    learner_df = learner_df.rename(
        columns={
            "utt_id": "learner_utt_id",
            "label": "learner_label",
            "start": "learner_start",
            "end": "learner_end",
            "duration": "learner_duration",
            "wav_path": "learner_wav_path",
        }
    )
    merged = pd.merge(ref_df, learner_df, on="seg_idx", how="outer", indicator=True)
    merged["labels_match"] = merged["ref_label"].fillna("") == merged["learner_label"].fillna("")
    return merged.sort_values("seg_idx").reset_index(drop=True)


def mean_or_nan(values: Iterable[float]) -> float:
    clean_values = [float(v) for v in values if pd.notna(v)]
    return float(np.mean(clean_values)) if clean_values else float("nan")


def normalize_score(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(np.clip(float(value), 0.0, 100.0))


def score_to_grade(value: float) -> str:
    if pd.isna(value):
        return ""
    score = normalize_score(value)
    if score < 60.0:
        return "不合格"
    if score < 70.0:
        return "合格"
    if score < 85.0:
        return "良好"
    return "优秀"


def percent_string(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{normalize_score(value):.2f}%"


def evaluate_one_pair(
    pair_id: str,
    reference_id: str,
    learner_id: str,
    reference_dir: Path,
    learner_dir: Path,
    reference_aligned_dir: Path,
    learner_aligned_dir: Path,
    output_dir: Path,
    sample_rate: int,
    nfilts: int,
    nceps: int,
    alpha: float,
    beta: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    ref_wav = reference_dir / f"{reference_id}.wav"
    learner_wav = learner_dir / f"{learner_id}.wav"
    ref_tg = reference_aligned_dir / f"{reference_id}.TextGrid"
    learner_tg = learner_aligned_dir / f"{learner_id}.TextGrid"

    missing_inputs = [
        str(path)
        for path in (ref_wav, learner_wav, ref_tg, learner_tg)
        if not path.exists()
    ]
    if missing_inputs:
        return (
            {
                "pair_id": pair_id,
                "reference_id": reference_id,
                "learner_id": learner_id,
                "status": "missing_input",
                "missing_inputs": ";".join(missing_inputs),
            },
            [],
            [],
        )

    phone_segment_root = output_dir / "segments" / "phones"
    word_segment_root = output_dir / "segments" / "words"

    ref_phone_df = cut_phone_intervals(ref_tg, ref_wav, "phones", phone_segment_root / "reference")
    learner_phone_df = cut_phone_intervals(learner_tg, learner_wav, "phones", phone_segment_root / "learner")
    ref_word_df = cut_word_intervals(ref_tg, ref_wav, "words", word_segment_root / "reference")
    learner_word_df = cut_word_intervals(learner_tg, learner_wav, "words", word_segment_root / "learner")

    phone_rows = build_scored_rows(
        pair_id=pair_id,
        reference_id=reference_id,
        learner_id=learner_id,
        tier_name="phone",
        aligned_df=align_segment_tables(ref_phone_df, learner_phone_df),
        sample_rate=sample_rate,
        nfilts=nfilts,
        nceps=nceps,
        alpha=alpha,
        beta=beta,
    )
    word_rows = build_scored_rows(
        pair_id=pair_id,
        reference_id=reference_id,
        learner_id=learner_id,
        tier_name="word",
        aligned_df=align_segment_tables(ref_word_df, learner_word_df),
        sample_rate=sample_rate,
        nfilts=nfilts,
        nceps=nceps,
        alpha=alpha,
        beta=beta,
    )

    phone_df = pd.DataFrame(phone_rows)
    word_df = pd.DataFrame(word_rows)
    sentence_metrics = build_sentence_metrics(
        ref_wav=ref_wav,
        learner_wav=learner_wav,
        ref_phone_df=ref_phone_df,
        learner_phone_df=learner_phone_df,
        sample_rate=sample_rate,
        nfilts=nfilts,
        nceps=nceps,
        alpha=alpha,
        beta=beta,
    )

    phone_mean_score = normalize_score(mean_or_nan(phone_df.get("score_old", pd.Series(dtype=float))))
    word_mean_score = normalize_score(mean_or_nan(word_df.get("score_old", pd.Series(dtype=float))))
    phone_mean_score_new = normalize_score(mean_or_nan(phone_df.get("score_new_linear", pd.Series(dtype=float))))
    word_mean_score_new = normalize_score(mean_or_nan(word_df.get("score_new_linear", pd.Series(dtype=float))))
    phone_mean_dtw_norm_distance_dimnorm = mean_or_nan(phone_df.get("dtw_norm_distance_dimnorm", pd.Series(dtype=float)))
    word_mean_dtw_norm_distance_dimnorm = mean_or_nan(word_df.get("dtw_norm_distance_dimnorm", pd.Series(dtype=float)))

    summary = {
        "pair_id": pair_id,
        "reference_id": reference_id,
        "learner_id": learner_id,
        "status": "ok",
        "phone_mean_score": phone_mean_score,
        "word_mean_score": word_mean_score,
        "phone_mean_score_old": phone_mean_score,
        "word_mean_score_old": word_mean_score,
        "phone_mean_score_new_linear": phone_mean_score_new,
        "word_mean_score_new_linear": word_mean_score_new,
        "phone_mean_dtw_norm_distance_dimnorm": phone_mean_dtw_norm_distance_dimnorm,
        "word_mean_dtw_norm_distance_dimnorm": word_mean_dtw_norm_distance_dimnorm,
        "matched_phone_segments": int((phone_df.get("status") == "ok").sum()) if not phone_df.empty else 0,
        "matched_word_segments": int((word_df.get("status") == "ok").sum()) if not word_df.empty else 0,
        "mismatched_phone_labels": int((phone_df.get("labels_match") == False).sum()) if not phone_df.empty else 0,
        "mismatched_word_labels": int((word_df.get("labels_match") == False).sum()) if not word_df.empty else 0,
        "missing_phone_segments": int((phone_df.get("status") != "ok").sum()) if not phone_df.empty else 0,
        "missing_word_segments": int((word_df.get("status") != "ok").sum()) if not word_df.empty else 0,
    }
    summary.update(sentence_metrics)
    summary["sentence_score_raw"] = normalize_score(summary["sentence_score_raw"])
    summary["sentence_score_trimmed"] = normalize_score(summary["sentence_score_trimmed"])
    summary["sentence_score_active"] = normalize_score(summary["sentence_score_active"])
    summary["sentence_score"] = normalize_score(summary["sentence_score"])
    summary["sentence_score_raw_old"] = normalize_score(summary["sentence_score_raw_old"])
    summary["sentence_score_trimmed_old"] = normalize_score(summary["sentence_score_trimmed_old"])
    summary["sentence_score_active_old"] = normalize_score(summary["sentence_score_active_old"])
    summary["sentence_score_new_linear"] = normalize_score(summary["sentence_score_new_linear"])
    summary["sentence_score_raw_new_linear"] = normalize_score(summary["sentence_score_raw_new_linear"])
    summary["sentence_score_trimmed_new_linear"] = normalize_score(summary["sentence_score_trimmed_new_linear"])
    summary["sentence_score_active_new_linear"] = normalize_score(summary["sentence_score_active_new_linear"])

    summary["sentence_score_percent"] = percent_string(summary["sentence_score"])
    summary["sentence_score_grade"] = score_to_grade(summary["sentence_score"])
    summary["sentence_score_raw_percent"] = percent_string(summary["sentence_score_raw"])
    summary["sentence_score_trimmed_percent"] = percent_string(summary["sentence_score_trimmed"])
    summary["sentence_score_active_percent"] = percent_string(summary["sentence_score_active"])

    summary["phone_mean_score_percent"] = percent_string(summary["phone_mean_score"])
    summary["phone_mean_score_grade"] = score_to_grade(summary["phone_mean_score"])
    summary["word_mean_score_percent"] = percent_string(summary["word_mean_score"])
    summary["word_mean_score_grade"] = score_to_grade(summary["word_mean_score"])
    summary["phone_mean_score_new_linear_percent"] = percent_string(summary["phone_mean_score_new_linear"])
    summary["phone_mean_score_new_linear_grade"] = score_to_grade(summary["phone_mean_score_new_linear"])
    summary["word_mean_score_new_linear_percent"] = percent_string(summary["word_mean_score_new_linear"])
    summary["word_mean_score_new_linear_grade"] = score_to_grade(summary["word_mean_score_new_linear"])

    overall_components = [summary["sentence_score"], summary["phone_mean_score"], summary["word_mean_score"]]
    summary["overall_score"] = normalize_score(mean_or_nan(overall_components))
    summary["overall_score_percent"] = percent_string(summary["overall_score"])
    summary["overall_score_grade"] = score_to_grade(summary["overall_score"])
    overall_components_new = [
        summary["sentence_score_new_linear"],
        summary["phone_mean_score_new_linear"],
        summary["word_mean_score_new_linear"],
    ]
    summary["overall_score_new_linear"] = normalize_score(mean_or_nan(overall_components_new))
    summary["overall_score_new_linear_percent"] = percent_string(summary["overall_score_new_linear"])
    summary["overall_score_new_linear_grade"] = score_to_grade(summary["overall_score_new_linear"])
    return summary, phone_rows, word_rows


def build_scored_rows(
    pair_id: str,
    reference_id: str,
    learner_id: str,
    tier_name: str,
    aligned_df: pd.DataFrame,
    sample_rate: int,
    nfilts: int,
    nceps: int,
    alpha: float,
    beta: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in aligned_df.iterrows():
        base_row = {
            "pair_id": pair_id,
            "reference_id": reference_id,
            "learner_id": learner_id,
            "tier": tier_name,
            "seg_idx": int(row["seg_idx"]),
            "ref_label": row.get("ref_label"),
            "learner_label": row.get("learner_label"),
            "labels_match": bool(row.get("labels_match", False)),
            "ref_start": row.get("ref_start"),
            "ref_end": row.get("ref_end"),
            "learner_start": row.get("learner_start"),
            "learner_end": row.get("learner_end"),
            "ref_wav_path": row.get("ref_wav_path"),
            "learner_wav_path": row.get("learner_wav_path"),
        }
        if row["_merge"] != "both":
            base_row["status"] = str(row["_merge"])
            rows.append(base_row)
            continue

        try:
            metrics = score_segment_pair(
                ref_wav=Path(str(row["ref_wav_path"])),
                learner_wav=Path(str(row["learner_wav_path"])),
                sample_rate=sample_rate,
                nfilts=nfilts,
                nceps=nceps,
                alpha=alpha,
                beta=beta,
            )
            base_row.update(metrics)
            base_row["status"] = "ok"
        except Exception as exc:
            base_row["status"] = "error"
            base_row["error"] = str(exc)

        rows.append(base_row)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch evaluation for MFA-aligned Fijian utterances using GFCC + DTW."
    )
    parser.add_argument("--reference-dir", required=True, type=Path, help="Directory with reference WAV files.")
    parser.add_argument("--learner-dir", required=True, type=Path, help="Directory with learner WAV files.")
    parser.add_argument("--reference-aligned-dir", required=True, type=Path, help="Directory with reference TextGrid files.")
    parser.add_argument("--learner-aligned-dir", required=True, type=Path, help="Directory with learner TextGrid files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write segments and CSV reports.")
    parser.add_argument("--pair-csv", type=Path, default=None, help="Optional CSV with columns reference_id, learner_id, and optional pair_id.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate for loading audio.")
    parser.add_argument("--nfilts", type=int, default=40, help="GFCC filterbank count.")
    parser.add_argument("--nceps", type=int, default=13, help="GFCC cepstral coefficient count.")
    parser.add_argument("--alpha", type=float, default=8.0, help="Score mapping alpha parameter.")
    parser.add_argument("--beta", type=float, default=0.6, help="Score mapping beta parameter.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pair_csv, args.reference_dir, args.learner_dir)
    if not pairs:
        raise FileNotFoundError(f"No learner wav files found in {args.learner_dir}")

    utterance_rows: list[dict[str, object]] = []
    phone_rows: list[dict[str, object]] = []
    word_rows: list[dict[str, object]] = []

    for pair in pairs:
        summary, pair_phone_rows, pair_word_rows = evaluate_one_pair(
            pair_id=pair["pair_id"],
            reference_id=pair["reference_id"],
            learner_id=pair["learner_id"],
            reference_dir=args.reference_dir,
            learner_dir=args.learner_dir,
            reference_aligned_dir=args.reference_aligned_dir,
            learner_aligned_dir=args.learner_aligned_dir,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            nfilts=args.nfilts,
            nceps=args.nceps,
            alpha=args.alpha,
            beta=args.beta,
        )
        utterance_rows.append(summary)
        phone_rows.extend(pair_phone_rows)
        word_rows.extend(pair_word_rows)
        print(
            f"[{summary['status']}] pair={pair['pair_id']} "
            f"sentence_d={summary.get('sentence_dtw_norm_distance_dimnorm', float('nan'))} "
            f"phone_d={summary.get('phone_mean_dtw_norm_distance_dimnorm', float('nan'))} "
            f"word_d={summary.get('word_mean_dtw_norm_distance_dimnorm', float('nan'))}"
        )

    utterance_df = pd.DataFrame(utterance_rows)
    phone_df = pd.DataFrame(phone_rows)
    word_df = pd.DataFrame(word_rows)

    utterance_df["sentence_score_new_relative"] = relative_distance_to_score_series(
        utterance_df.get("sentence_dtw_norm_distance_dimnorm", pd.Series(dtype=float))
    )
    utterance_df["phone_mean_score_new_relative"] = relative_distance_to_score_series(
        utterance_df.get("phone_mean_dtw_norm_distance_dimnorm", pd.Series(dtype=float))
    )
    utterance_df["word_mean_score_new_relative"] = relative_distance_to_score_series(
        utterance_df.get("word_mean_dtw_norm_distance_dimnorm", pd.Series(dtype=float))
    )
    utterance_df["overall_score_new_relative"] = utterance_df[
        ["sentence_score_new_relative", "phone_mean_score_new_relative", "word_mean_score_new_relative"]
    ].mean(axis=1, skipna=True)
    utterance_df["sentence_score_new_relative_percent"] = utterance_df["sentence_score_new_relative"].apply(percent_string)
    utterance_df["sentence_score_new_relative_grade"] = utterance_df["sentence_score_new_relative"].apply(score_to_grade)
    utterance_df["phone_mean_score_new_relative_percent"] = utterance_df["phone_mean_score_new_relative"].apply(percent_string)
    utterance_df["phone_mean_score_new_relative_grade"] = utterance_df["phone_mean_score_new_relative"].apply(score_to_grade)
    utterance_df["word_mean_score_new_relative_percent"] = utterance_df["word_mean_score_new_relative"].apply(percent_string)
    utterance_df["word_mean_score_new_relative_grade"] = utterance_df["word_mean_score_new_relative"].apply(score_to_grade)
    utterance_df["overall_score_new_relative_percent"] = utterance_df["overall_score_new_relative"].apply(percent_string)
    utterance_df["overall_score_new_relative_grade"] = utterance_df["overall_score_new_relative"].apply(score_to_grade)

    utterance_df.to_csv(args.output_dir / "utterance_scores.csv", index=False, encoding="utf-8-sig")
    phone_df.to_csv(args.output_dir / "phone_scores.csv", index=False, encoding="utf-8-sig")
    word_df.to_csv(args.output_dir / "word_scores.csv", index=False, encoding="utf-8-sig")
    comparison_columns = [
        "pair_id",
        "reference_id",
        "learner_id",
        "status",
        "sentence_dtw_norm_distance",
        "sentence_dtw_norm_distance_dimnorm",
        "sentence_score",
        "sentence_score_new_relative",
        "phone_mean_score",
        "phone_mean_score_new_relative",
        "word_mean_score",
        "word_mean_score_new_relative",
        "overall_score",
        "overall_score_new_relative",
    ]
    available_columns = [col for col in comparison_columns if col in utterance_df.columns]
    utterance_df[available_columns].to_csv(
        args.output_dir / "utterance_score_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    run_summary = {
        "num_pairs": int(len(utterance_df)),
        "num_ok_pairs": int((utterance_df["status"] == "ok").sum()) if not utterance_df.empty else 0,
        "mean_sentence_score": normalize_score(mean_or_nan(utterance_df.get("sentence_score", pd.Series(dtype=float)))),
        "mean_sentence_score_new_relative": normalize_score(mean_or_nan(utterance_df.get("sentence_score_new_relative", pd.Series(dtype=float)))),
        "mean_phone_score": normalize_score(mean_or_nan(utterance_df.get("phone_mean_score", pd.Series(dtype=float)))),
        "mean_phone_score_new_relative": normalize_score(mean_or_nan(utterance_df.get("phone_mean_score_new_relative", pd.Series(dtype=float)))),
        "mean_word_score": normalize_score(mean_or_nan(utterance_df.get("word_mean_score", pd.Series(dtype=float)))),
        "mean_word_score_new_relative": normalize_score(mean_or_nan(utterance_df.get("word_mean_score_new_relative", pd.Series(dtype=float)))),
        "mean_overall_score": normalize_score(mean_or_nan(utterance_df.get("overall_score", pd.Series(dtype=float)))),
        "mean_overall_score_new_relative": normalize_score(mean_or_nan(utterance_df.get("overall_score_new_relative", pd.Series(dtype=float)))),
        "output_dir": str(args.output_dir.resolve()),
    }
    run_summary["mean_sentence_score_percent"] = percent_string(run_summary["mean_sentence_score"])
    run_summary["mean_sentence_score_new_relative_percent"] = percent_string(run_summary["mean_sentence_score_new_relative"])
    run_summary["mean_phone_score_percent"] = percent_string(run_summary["mean_phone_score"])
    run_summary["mean_phone_score_new_relative_percent"] = percent_string(run_summary["mean_phone_score_new_relative"])
    run_summary["mean_word_score_percent"] = percent_string(run_summary["mean_word_score"])
    run_summary["mean_word_score_new_relative_percent"] = percent_string(run_summary["mean_word_score_new_relative"])
    run_summary["mean_overall_score_percent"] = percent_string(run_summary["mean_overall_score"])
    run_summary["mean_overall_score_new_relative_percent"] = percent_string(run_summary["mean_overall_score_new_relative"])
    with open(args.output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("\n=== Run summary ===")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
