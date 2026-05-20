#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from fastdtw import fastdtw
from praatio import textgrid


@dataclass
class Interval:
    label: str
    start: float
    end: float


def resolve_input_path(path_str: str) -> Path:
    raw = path_str.strip()
    if os.name == "nt" and raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5].upper()
        suffix = raw[6:].replace("/", "\\")
        return Path(f"{drive}:{suffix}")
    return Path(raw)


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


def extract_gfcc_from_signal(y: np.ndarray, sr: int, nfilts: int, nceps: int) -> np.ndarray:
    from spafe.features.gfcc import gfcc

    gfcc_params = inspect.signature(gfcc).parameters
    kwargs = {"fs": sr, "nfilts": nfilts}
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


def compute_dimnorm_distance(ref_feats: np.ndarray, learner_feats: np.ndarray) -> float:
    distance, path = fastdtw(
        ref_feats,
        learner_feats,
        dist=lambda x, y: np.linalg.norm(x - y) / max(len(x), 1),
    )
    path_len = max(len(path), 1)
    return float(distance) / path_len


def score_signal_pair(
    ref_signal: np.ndarray,
    learner_signal: np.ndarray,
    sample_rate: int,
    nfilts: int,
    nceps: int,
) -> float:
    ref_feats = extract_gfcc_from_signal(ref_signal, sample_rate, nfilts, nceps)
    learner_feats = extract_gfcc_from_signal(learner_signal, sample_rate, nfilts, nceps)
    return compute_dimnorm_distance(ref_feats, learner_feats)


def time_to_sample(t: float, sr: int) -> int:
    return max(0, int(round(t * sr)))


def slice_signal(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    start_idx = time_to_sample(start, sr)
    end_idx = min(len(y), time_to_sample(end, sr))
    if end_idx <= start_idx:
        raise ValueError(f"Invalid slice range: start={start}, end={end}")
    return y[start_idx:end_idx]


def load_tier_intervals(tg_path: Path, tier_name: str, skip_silence: bool) -> list[Interval]:
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    if tier_name not in tg.tierNames:
        raise ValueError(f"Tier '{tier_name}' not found in {tg_path}. Available: {tg.tierNames}")

    intervals: list[Interval] = []
    for start, end, label in tg.getTier(tier_name).entries:
        clean = label.strip()
        if not clean:
            continue
        if skip_silence and clean.lower() in {"sp", "sil", "spn"}:
            continue
        intervals.append(Interval(label=clean, start=float(start), end=float(end)))
    return intervals


def group_phones_by_words(words: list[Interval], phones: list[Interval]) -> list[list[Interval]]:
    grouped: list[list[Interval]] = []
    phone_index = 0
    for word in words:
        current: list[Interval] = []
        while phone_index < len(phones):
            phone = phones[phone_index]
            phone_mid = (phone.start + phone.end) / 2.0
            if phone_mid < word.start:
                phone_index += 1
                continue
            if phone_mid > word.end:
                break
            current.append(phone)
            phone_index += 1
        grouped.append(current)
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


def pair_by_index(ref_items: list[Interval], learner_items: list[Interval]) -> tuple[list[tuple[Interval, Interval]], dict]:
    pair_count = min(len(ref_items), len(learner_items))
    pairs = list(zip(ref_items[:pair_count], learner_items[:pair_count]))
    mismatch = {
        "reference_count": len(ref_items),
        "learner_count": len(learner_items),
        "count_match": len(ref_items) == len(learner_items),
        "reference_only": [item.label for item in ref_items[pair_count:]],
        "learner_only": [item.label for item in learner_items[pair_count:]],
    }
    return pairs, mismatch


def build_output(
    sentence_text: str,
    reference_tg: Path,
    learner_tg: Path,
    reference_wav: Path,
    learner_wav: Path,
    sample_rate: int,
    nfilts: int,
    nceps: int,
) -> dict:
    ref_signal, sr = load_wav(reference_wav, sample_rate)
    learner_signal, _ = load_wav(learner_wav, sample_rate)

    ref_words = load_tier_intervals(reference_tg, "words", skip_silence=False)
    learner_words = load_tier_intervals(learner_tg, "words", skip_silence=False)
    ref_phones = load_tier_intervals(reference_tg, "phones", skip_silence=True)
    learner_phones = load_tier_intervals(learner_tg, "phones", skip_silence=True)

    word_pairs, word_pair_mismatch = pair_by_index(ref_words, learner_words)
    ref_phones_by_word = group_phones_by_words(ref_words, ref_phones)
    learner_phones_by_word = group_phones_by_words(learner_words, learner_phones)

    sentence_distance = score_signal_pair(
        ref_signal,
        learner_signal,
        sample_rate=sr,
        nfilts=nfilts,
        nceps=nceps,
    )

    words_output = {}
    word_key_counts: dict[str, int] = {}
    sentence_mismatch = {
        "word_alignment": word_pair_mismatch,
        "word_label_mismatches": [],
        "phone_alignment": [],
    }
    global_phone_idx = 0
    for word_idx, (ref_word, learner_word) in enumerate(word_pairs):
        ref_word_signal = slice_signal(ref_signal, sr, ref_word.start, ref_word.end)
        learner_word_signal = slice_signal(learner_signal, sr, learner_word.start, learner_word.end)
        word_distance = score_signal_pair(
            ref_word_signal,
            learner_word_signal,
            sample_rate=sr,
            nfilts=nfilts,
            nceps=nceps,
        )

        ref_word_phones = ref_phones_by_word[word_idx]
        learner_word_phones = learner_phones_by_word[word_idx]
        phone_pairs, phone_pair_mismatch = pair_by_index(ref_word_phones, learner_word_phones)
        phone_mappings = build_phone_mappings(ref_word.label, [phone.label for phone in ref_word_phones[:len(phone_pairs)]])

        phones_output = {}
        phone_key_counts: dict[str, int] = {}
        word_mismatch = {
            "word_label_mismatch": ref_word.label != learner_word.label,
            "phone_alignment": phone_pair_mismatch,
            "phone_label_mismatches": [],
        }
        for phone_idx, (ref_phone, learner_phone) in enumerate(phone_pairs):
            ref_phone_signal = slice_signal(ref_signal, sr, ref_phone.start, ref_phone.end)
            learner_phone_signal = slice_signal(learner_signal, sr, learner_phone.start, learner_phone.end)
            phone_distance = score_signal_pair(
                ref_phone_signal,
                learner_phone_signal,
                sample_rate=sr,
                nfilts=nfilts,
                nceps=nceps,
            )
            phone_key = make_unique_key(ref_phone.label, phone_key_counts)
            phone_entry = {
                "mapping": phone_mappings[phone_idx] if phone_idx < len(phone_mappings) else "",
                "learner_phone": learner_phone.label,
                "phone_distance": phone_distance,
            }
            phones_output[phone_key] = phone_entry
            if ref_phone.label != learner_phone.label:
                word_mismatch["phone_label_mismatches"].append(
                    {
                        "phone_key": phone_key,
                        "reference_phone": ref_phone.label,
                        "learner_phone": learner_phone.label,
                    }
                )
            global_phone_idx += 1

        word_key = make_unique_key(ref_word.label, word_key_counts)
        if word_mismatch["word_label_mismatch"]:
            sentence_mismatch["word_label_mismatches"].append(
                {
                    "word_key": word_key,
                    "reference_word": ref_word.label,
                    "learner_word": learner_word.label,
                    "seg_idx": word_idx,
                }
            )
        sentence_mismatch["phone_alignment"].append(
            {
                "word_key": word_key,
                **phone_pair_mismatch,
            }
        )

        words_output[word_key] = {
            "learner_word": learner_word.label,
            "word_distance": word_distance,
            "phones_distances": phones_output,
        }
        if (
            word_mismatch["word_label_mismatch"]
            or word_mismatch["phone_alignment"]["count_match"] is False
            or word_mismatch["phone_label_mismatches"]
            or word_mismatch["phone_alignment"]["reference_only"]
            or word_mismatch["phone_alignment"]["learner_only"]
        ):
            words_output[word_key]["mismatch"] = word_mismatch

    output = {
        "sentence_text": sentence_text.strip(),
        "sentence_distance": sentence_distance,
        "words": words_output,
    }
    if (
        sentence_mismatch["word_alignment"]["count_match"] is False
        or sentence_mismatch["word_label_mismatches"]
        or any(
            item["count_match"] is False or item["reference_only"] or item["learner_only"]
            for item in sentence_mismatch["phone_alignment"]
        )
    ):
        output["mismatch"] = sentence_mismatch
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print sentence/word/phone DTW distance JSON from text, TextGrids, and wav paths."
    )
    parser.add_argument("--text", required=True, help="Sentence text")
    parser.add_argument("--reference-textgrid", required=True, help="Reference TextGrid path in WSL or local format")
    parser.add_argument("--learner-textgrid", required=True, help="Learner TextGrid path in WSL or local format")
    parser.add_argument("--reference-wav", required=True, help="Reference wav path in WSL or local format")
    parser.add_argument("--learner-wav", required=True, help="Learner wav path in WSL or local format")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--nfilts", type=int, default=40, help="GFCC filterbank count")
    parser.add_argument("--nceps", type=int, default=13, help="GFCC cepstral coefficient count")
    return parser


def main() -> None:
    """该脚本的主入口点。

    解析命令行参数，调用核心逻辑来处理音频和文本网格文件，
    计算动态时间规整（DTW）距离，并将最终结果以 JSON 格式输出到标准输出。
    """
    args = build_arg_parser().parse_args()

    output = build_output(
        sentence_text=args.text,
        reference_tg=resolve_input_path(args.reference_textgrid),
        learner_tg=resolve_input_path(args.learner_textgrid),
        reference_wav=resolve_input_path(args.reference_wav),
        learner_wav=resolve_input_path(args.learner_wav),
        sample_rate=args.sample_rate,
        nfilts=args.nfilts,
        nceps=args.nceps,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    
"""
python scripts/mfa_pipeline/print_dtw_distance_json.py \
  --text "e cici tiko na yalewa" \
  --reference-textgrid /mnt/d/workdir/data/fijian/aligned/reference/u1_l1_010.TextGrid \
  --learner-textgrid /mnt/d/workdir/data/fijian/aligned/reference/u1_l1_010.TextGrid \
  --reference-wav /mnt/d/workdir/data/fijian/raw_corpus/u1_l1_010.wav \
  --learner-wav /mnt/d/workdir/data/fijian/raw_corpus/u1_l1_010.wav

python scripts/mfa_pipeline/print_dtw_distance_json.py \
  --text "e cici tiko na yalewa" \
  --reference-textgrid /mnt/d/workdir/data/fijian/aligned/reference/u1_l1_010.TextGrid \
  --learner-textgrid /mnt/d/workdir/data/fijian/aligned/learner/guoziyu/guoziyu_question_007.TextGrid \
  --reference-wav /mnt/d/workdir/data/fijian/raw_corpus/u1_l1_010.wav \
  --learner-wav /mnt/d/workdir/data/fijian/learner_pcm/guoziyu_recordings/guoziyu_question_007.wav
"""