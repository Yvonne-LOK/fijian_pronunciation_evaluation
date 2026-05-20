# 批量读取整个 aligned 文件夹

from pathlib import Path
import pandas as pd
from praatio import textgrid


def extract_tier_intervals(tg: textgrid.Textgrid, tier_name: str, utt_id: str):
    if tier_name not in tg.tierNames:
        raise ValueError(f"Tier '{tier_name}' not found. Available tiers: {tg.tierNames}")

    tier = tg.getTier(tier_name)
    results = []

    for start, end, label in tier.entries:
        label = label.strip() if isinstance(label, str) else str(label)
        results.append(
            {
                "utt_id": utt_id,
                "tier": tier_name,
                "label": label,
                "start": float(start),
                "end": float(end),
                "duration": float(end) - float(start),
            }
        )
    return results


def parse_one_textgrid(tg_path: Path):
    tg = textgrid.openTextgrid(
        str(tg_path),
        includeEmptyIntervals=True
    )
    utt_id = tg_path.stem

    words = extract_tier_intervals(tg, "words", utt_id)
    phones = extract_tier_intervals(tg, "phones", utt_id)
    return words, phones


def main():
    aligned_dir = Path("data/fijian/aligned")
    out_dir = Path("data/fijian/parsed")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_words = []
    all_phones = []

    tg_files = sorted(aligned_dir.glob("*.TextGrid"))
    if not tg_files:
        raise FileNotFoundError(f"No TextGrid files found in {aligned_dir}")

    for tg_path in tg_files:
        try:
            words, phones = parse_one_textgrid(tg_path)
            all_words.extend(words)
            all_phones.extend(phones)
            print(f"[OK] {tg_path.name}")
        except Exception as e:
            print(f"[ERROR] {tg_path.name}: {e}")

    words_df = pd.DataFrame(all_words)
    phones_df = pd.DataFrame(all_phones)

    words_df.to_csv(out_dir / "all_words.csv", index=False, encoding="utf-8-sig")
    phones_df.to_csv(out_dir / "all_phones.csv", index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Words saved to:  {out_dir / 'all_words.csv'}")
    print(f"Phones saved to: {out_dir / 'all_phones.csv'}")


if __name__ == "__main__":
    main()