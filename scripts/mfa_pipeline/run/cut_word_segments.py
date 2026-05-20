# 按词边界切音频

from pathlib import Path
import soundfile as sf
import pandas as pd
from praatio import textgrid


def load_audio(wav_path: Path):
    """
    读取音频
    返回:
        audio: np.ndarray, shape = (n_samples,) or (n_samples, n_channels)
        sr: int
    """
    audio, sr = sf.read(str(wav_path))
    return audio, sr


def time_to_sample(t: float, sr: int) -> int:
    return max(0, int(round(t * sr)))


def sanitize_label(label: str) -> str:
    """
    清理标签，便于作为文件名
    """
    label = label.strip()
    if not label:
        return "EMPTY"
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        label = label.replace(ch, "_")
    return label


def cut_intervals_from_tier(
    tg_path: Path,
    wav_path: Path,
    tier_name: str,
    out_dir: Path,
    keep_empty: bool = False,
):
    """
    按指定 tier 切分音频并保存
    """
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    if tier_name not in tg.tierNames:
        raise ValueError(f"Tier '{tier_name}' not found in {tg_path.name}. Available: {tg.tierNames}")

    tier = tg.getTier(tier_name)
    audio, sr = load_audio(wav_path)

    utt_id = wav_path.stem
    utt_out_dir = out_dir / utt_id
    utt_out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seg_idx = 0

    for start, end, label in tier.entries:
        label = label.strip()
        if (not keep_empty) and (label == ""):
            continue

        start_samp = time_to_sample(float(start), sr)
        end_samp = time_to_sample(float(end), sr)

        if end_samp <= start_samp:
            continue

        segment = audio[start_samp:end_samp]
        clean_label = sanitize_label(label)
        out_name = f"{seg_idx:03d}_{clean_label}.wav"
        out_path = utt_out_dir / out_name

        sf.write(str(out_path), segment, sr)

        rows.append(
            {
                "utt_id": utt_id,
                "tier": tier_name,
                "seg_idx": seg_idx,
                "label": label,
                "start": float(start),
                "end": float(end),
                "duration": float(end) - float(start),
                "wav_path": str(out_path),
            }
        )
        seg_idx += 1

    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(utt_out_dir / f"{utt_id}_{tier_name}_segments.csv", index=False, encoding="utf-8-sig")
    return meta_df


def main():
    wav_path = Path("data/fijian/raw_corpus/u1_l1_008.wav")
    tg_path = Path("data/fijian/aligned/u1_l1_008.TextGrid")
    out_dir = Path("data/fijian/parsed/words")  
    # Path: 路径对象，用于表示文件系统中的路径
    # 为什么要用Path：跨平台兼容性，Path对象可以在不同操作系统（如Windows、macOS、Linux）上使用，而无需担心路径分隔符的问题。

    df = cut_intervals_from_tier(
        tg_path=tg_path,
        wav_path=wav_path,
        tier_name="words",
        out_dir=out_dir,
        keep_empty=False,
    )

    print(df.to_string(index=False))
    print(f"\nSaved word segments to: {out_dir / wav_path.stem}")


if __name__ == "__main__":
    main()