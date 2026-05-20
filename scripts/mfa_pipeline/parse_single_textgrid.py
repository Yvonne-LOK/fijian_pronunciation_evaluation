# 读取单个 TextGrid

from pathlib import Path
import pandas as pd
from praatio import textgrid


def extract_tier_intervals(tg: textgrid.Textgrid, tier_name: str):
    """
    从指定 tier 中提取区间信息
    返回 list[dict]
    """
    if tier_name not in tg.tierNames:
        raise ValueError(f"Tier '{tier_name}' not found. Available tiers: {tg.tierNames}")

    tier = tg.getTier(tier_name)
    results = []

    # IntervalTier 的 entries 形式通常为: (start, end, label)
    for start, end, label in tier.entries:
        label = label.strip() if isinstance(label, str) else str(label)
        results.append(
            {
                "tier": tier_name,
                "label": label,
                "start": float(start),
                "end": float(end),
                "duration": float(end) - float(start),
            }
        )
    return results


def read_one_textgrid(textgrid_path: str):
    """
    读取单个 TextGrid，返回 words_df, phones_df
    """
    tg = textgrid.openTextgrid(
        textgrid_path,
        includeEmptyIntervals=True
    )

    words = extract_tier_intervals(tg, "words")
    phones = extract_tier_intervals(tg, "phones")

    words_df = pd.DataFrame(words)
    phones_df = pd.DataFrame(phones)

    return words_df, phones_df


def main():
    tg_path = "data/fijian/aligned/u1_l1_008.TextGrid"

    words_df, phones_df = read_one_textgrid(tg_path)

    print("\n=== WORDS ===")
    print(words_df.to_string(index=False))

    print("\n=== PHONES ===")
    print(phones_df.to_string(index=False))

    # 可选：保存 csv
    out_dir = Path("data/fijian/parsed")
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(tg_path).stem
    # 保存到同一个csv的两个sheet：words 和 phones
    with pd.ExcelWriter(out_dir / f"{stem}_parsed.xlsx") as writer:
        words_df.to_excel(writer, sheet_name="words", index=False)
        phones_df.to_excel(writer, sheet_name="phones", index=False)    
    # words_df.to_csv(out_dir / f"{stem}_words.csv", index=False, encoding="utf-8-sig")
    # phones_df.to_csv(out_dir / f"{stem}_phones.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved to: {out_dir}/{stem}_parsed.xlsx")


if __name__ == "__main__":
    main()