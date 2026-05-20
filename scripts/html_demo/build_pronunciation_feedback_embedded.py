#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
from pathlib import Path


RUN_DIR_PATTERN = re.compile(r"^learner_(?P<learner>.+)_(?P<stamp>\d{8}_\d{6})$")

TRANSLATIONS_ZH = {
    "u1_l1_001": "你好",
    "u1_l1_003": "一个女人",
    "u1_l1_004": "一个男人",
    "u1_l1_007": "男人在跑步",
    "u1_l1_008": "男人在走路",
    "u1_l1_009": "女人在走路",
    "u1_l1_010": "女人在跑步",
    "u1_l1_011": "男人在唱歌",
    "u1_l1_012": "女人在跳舞",
    "u1_l1_013": "女人在唱歌",
}


def default_path(windows_path: str, wsl_path: str) -> Path:
    return Path(windows_path if os.name == "nt" else wsl_path)


def resolve_path(path_str: str | Path) -> Path:
    raw = str(path_str).strip()
    if os.name != "nt" and re.match(r"^[A-Za-z]:\\", raw):
        drive = raw[0].lower()
        suffix = raw[2:].replace("\\", "/")
        return Path(f"/mnt/{drive}{suffix}")
    if os.name == "nt" and raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5].upper()
        suffix = raw[6:].replace("/", "\\")
        return Path(f"{drive}:{suffix}")
    return Path(raw)


DEFAULT_REFERENCE_DIR = default_path(r"d:\workdir\data\fijian\raw_corpus", "/mnt/d/workdir/data/fijian/raw_corpus")
DEFAULT_LEARNER_PCM_ROOT = default_path(r"d:\workdir\data\fijian\learner_pcm", "/mnt/d/workdir/data/fijian/learner_pcm")
DEFAULT_EVAL_RUNS_DIR = default_path(r"d:\workdir\outputs\eval_runs", "/mnt/d/workdir/outputs/eval_runs")
DEFAULT_OUTPUT_DIR = default_path(r"d:\workdir\scripts\html_demo", "/mnt/d/workdir/scripts/html_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-file pronunciation feedback HTML with embedded audio."
    )
    parser.add_argument("learner_name", help="Learner name, for example zengjianbin")
    parser.add_argument("--eval-runs-dir", type=Path, default=DEFAULT_EVAL_RUNS_DIR, help="Root eval_runs directory.")
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR, help="Reference wav/txt directory.")
    parser.add_argument("--learner-pcm-root", type=Path, default=DEFAULT_LEARNER_PCM_ROOT, help="Root learner pcm directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for embedded HTML.")
    return parser.parse_args()


def find_latest_run(eval_runs_dir: Path, learner_name: str) -> tuple[str, Path]:
    latest_stamp: str | None = None
    latest_dir: Path | None = None
    for path in eval_runs_dir.iterdir():
        if not path.is_dir():
            continue
        match = RUN_DIR_PATTERN.match(path.name)
        if not match or match.group("learner") != learner_name:
            continue
        stamp = match.group("stamp")
        if latest_stamp is None or stamp > latest_stamp:
            latest_stamp = stamp
            latest_dir = path
    if latest_stamp is None or latest_dir is None:
        raise FileNotFoundError(f"No eval run found for learner={learner_name!r} under {eval_runs_dir}")
    return latest_stamp, latest_dir


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_score(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def tone_class(score: float | None) -> str:
    if score is None:
        return "tone-neutral"
    if score >= 85.0:
        return "tone-good"
    if score >= 60.0:
        return "tone-mid"
    return "tone-bad"


def sentence_status(sentence_score: float | None) -> tuple[str, str]:
    if sentence_score is None:
        return "status-pending", "待评估"
    if sentence_score >= 60.0:
        return "status-pass", "通过 √"
    return "status-fail", "不通过 x"


def sentence_translation(reference_id: str) -> str:
    return TRANSLATIONS_ZH.get(reference_id, "")


def encode_audio_data_uri(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing audio file: {path}")
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    suffix = path.suffix.lower()
    mime = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
    }.get(suffix, "application/octet-stream")
    return f"data:{mime};base64,{encoded}"


def build_audio_uri_map(records: list[dict], reference_dir: Path, learner_dir: Path) -> dict[str, str]:
    uri_map: dict[str, str] = {}
    for record in records:
        pair_id = str(record.get("pair_id", ""))
        reference_id, learner_id = pair_id.split("__", 1) if "__" in pair_id else (pair_id, pair_id)
        ref_key = f"reference::{reference_id}"
        learner_key = f"learner::{learner_id}"
        if ref_key not in uri_map:
            uri_map[ref_key] = encode_audio_data_uri(reference_dir / f"{reference_id}.wav")
        if learner_key not in uri_map:
            uri_map[learner_key] = encode_audio_data_uri(learner_dir / f"{learner_id}.wav")
    return uri_map


def render_word(word_label: str, word_entry: dict, sentence_score: float | None) -> str:
    word_score = safe_score(word_entry.get("word_score"))

    if sentence_score is not None and sentence_score >= 85.0:
        return f'<span class="word tone-good">{html.escape(word_label)}</span>'

    if sentence_score is not None and sentence_score >= 75.0:
        cls = "tone-good" if word_score is not None and word_score >= 85.0 else "tone-mid"
        return f'<span class="word {cls}">{html.escape(word_label)}</span>'

    if word_score is None or word_score >= 85.0:
        return f'<span class="word tone-good">{html.escape(word_label)}</span>'
    if word_score >= 60.0:
        return f'<span class="word tone-mid">{html.escape(word_label)}</span>'

    phones = word_entry.get("phones_distances", {})
    if not isinstance(phones, dict) or not phones:
        return f'<span class="word tone-bad">{html.escape(word_label)}</span>'

    phone_items: list[tuple[str, dict, float | None]] = []
    for phone_key, phone_entry in phones.items():
        phone_items.append((phone_key, phone_entry, safe_score(phone_entry.get("phone_score"))))

    red_candidates = [
        (idx, score)
        for idx, (_, _, score) in enumerate(phone_items)
        if score is not None and score < 60.0
    ]
    red_indices = {
        idx
        for idx, _ in sorted(red_candidates, key=lambda item: item[1])[:2]
    }

    pieces: list[str] = []
    for idx, (_, phone_entry, _) in enumerate(phone_items):
        mapping = str(phone_entry.get("mapping", "")).strip() or "?"
        cls = "tone-bad" if idx in red_indices else "tone-mid"
        pieces.append(
            f'<span class="phone-piece {cls}">{html.escape(mapping)}</span>'
        )
    return f'<span class="word word-fragmented">{"".join(pieces)}</span>'


def render_feedback_placeholder() -> str:
    return """
      <section class="feedback-box">
        <div class="feedback-title">评测反馈</div>
        <p class="feedback-text">
          整体读得不错！建议重点关注个别发音片段。后续可以在这里补充更自然的句级反馈，例如指出哪个单词或字母片段需要重点练习，并配合标准音频再次跟读。
        </p>
      </section>
    """


def render_card(record: dict, audio_uri_map: dict[str, str]) -> str:
    pair_id = str(record.get("pair_id", ""))
    reference_id, learner_id = pair_id.split("__", 1) if "__" in pair_id else (pair_id, pair_id)
    question_no = learner_id.split("_question_", 1)[1] if "_question_" in learner_id else learner_id
    sentence_text = str(record.get("sentence_text", "")).strip()
    sentence_score = safe_score(record.get("sentence_score"))
    status_class, status_text = sentence_status(sentence_score)

    translation_zh = sentence_translation(reference_id)
    reference_audio = audio_uri_map[f"reference::{reference_id}"]
    learner_audio = audio_uri_map[f"learner::{learner_id}"]

    words_html: list[str] = []
    for word_label, word_entry in record.get("words", {}).items():
        words_html.append(render_word(str(word_label), word_entry, sentence_score))

    return f"""
    <article class="card">
      <div class="card-head">
        <div>
          <div class="eyebrow">Question {html.escape(question_no)}</div>
          <h2>{html.escape(sentence_text)}</h2>
          <p class="translation">{html.escape(translation_zh)}</p>
        </div>
        <div class="status-chip {status_class}">{html.escape(status_text)}</div>
      </div>

      <div class="audio-grid">
        <div class="audio-box">
          <div class="audio-label">标准音频</div>
          <audio controls preload="none" src="{reference_audio}"></audio>
        </div>
        <div class="audio-box">
          <div class="audio-label">学生录音</div>
          <audio controls preload="none" src="{learner_audio}"></audio>
        </div>
      </div>

      <div class="rendered-line">
        {" ".join(words_html)}
      </div>

      {render_feedback_placeholder()}
    </article>
    """


def build_html(records: list[dict], audio_uri_map: dict[str, str]) -> str:
    cards = "\n".join(render_card(record, audio_uri_map) for record in records)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Fijian Pronunciation FeedBack</title>
  <style>
    :root {{
      --bg: #f3eee4;
      --card: rgba(255,255,255,0.9);
      --ink: #1f2a24;
      --muted: #66756c;
      --border: rgba(31,42,36,0.12);
      --good: #1f8a4c;
      --mid: #c28d12;
      --bad: #c43a2f;
      --neutral: #39423d;
      --pass-bg: rgba(31, 138, 76, 0.12);
      --fail-bg: rgba(196, 58, 47, 0.12);
      --pending-bg: rgba(57, 66, 61, 0.1);
      --shadow: 0 18px 45px rgba(43, 37, 26, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(210, 175, 132, 0.28), transparent 28%),
        radial-gradient(circle at top right, rgba(91, 146, 115, 0.18), transparent 22%),
        linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 72px;
    }}
    .hero {{
      margin-bottom: 26px;
      padding: 28px 30px;
      border: 1px solid var(--border);
      border-radius: 24px;
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(246,240,231,0.88));
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.15;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    .legend-pill {{
      display: inline-flex;
      align-items: center;
      padding: 7px 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
      border: 1px solid rgba(0,0,0,0.06);
      background: rgba(255,255,255,0.75);
    }}
    .cards {{
      display: grid;
      gap: 20px;
    }}
    .card {{
      padding: 24px;
      border-radius: 24px;
      border: 1px solid var(--border);
      background: var(--card);
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
    }}
    .eyebrow {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .card h2 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.25;
    }}
    .translation {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .status-chip {{
      flex-shrink: 0;
      padding: 10px 14px;
      border-radius: 999px;
      font-size: 14px;
      font-weight: 800;
      border: 1px solid rgba(0,0,0,0.06);
    }}
    .status-pass {{
      color: var(--good);
      background: var(--pass-bg);
    }}
    .status-fail {{
      color: var(--bad);
      background: var(--fail-bg);
    }}
    .status-pending {{
      color: var(--neutral);
      background: var(--pending-bg);
    }}
    .audio-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin: 18px 0 20px;
    }}
    .audio-box {{
      padding: 14px;
      border-radius: 18px;
      background: rgba(245, 240, 232, 0.92);
      border: 1px solid rgba(31,42,36,0.08);
    }}
    .audio-label {{
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--muted);
      font-weight: 700;
    }}
    audio {{
      width: 100%;
    }}
    .rendered-line {{
      font-size: 31px;
      line-height: 1.9;
      letter-spacing: 0.02em;
      padding: 18px 20px;
      border-radius: 20px;
      background: rgba(255,255,255,0.65);
      border: 1px dashed rgba(31,42,36,0.12);
    }}
    .word {{
      display: inline-block;
      margin-right: 0.15em;
      font-weight: 700;
    }}
    .word-fragmented {{
      padding: 0 0.03em;
    }}
    .phone-piece {{
      display: inline-block;
      border-bottom: 2px solid currentColor;
      margin: 0 0.01em;
    }}
    .tone-good {{ color: var(--good); }}
    .tone-mid {{ color: var(--mid); }}
    .tone-bad {{ color: var(--bad); }}
    .tone-neutral {{ color: var(--neutral); }}
    .feedback-box {{
      margin-top: 18px;
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(244, 239, 230, 0.95);
      border: 1px solid rgba(31,42,36,0.08);
    }}
    .feedback-title {{
      margin-bottom: 8px;
      font-size: 14px;
      font-weight: 800;
      color: var(--ink);
    }}
    .feedback-text {{
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      font-size: 15px;
    }}
    @media (max-width: 720px) {{
      .page {{ padding: 18px 14px 42px; }}
      .hero, .card {{ padding: 18px; }}
      .card-head {{ flex-direction: column; }}
      .card h2 {{ font-size: 24px; }}
      .rendered-line {{ font-size: 26px; line-height: 1.7; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>Fijian Pronunciation FeedBack</h1>
      <div class="legend">
        <span class="legend-pill tone-good">绿色：表现很好，继续保持</span>
        <span class="legend-pill tone-mid">黄色：表现达标，继续进步</span>
        <span class="legend-pill tone-bad">红色：发音有误，需要纠正</span>
      </div>
    </section>
    <section class="cards">
      {cards}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    learner_name = args.learner_name.strip()
    eval_runs_dir = resolve_path(args.eval_runs_dir)
    reference_dir = resolve_path(args.reference_dir)
    learner_pcm_root = resolve_path(args.learner_pcm_root)
    output_dir = resolve_path(args.output_dir)

    stamp, run_dir = find_latest_run(eval_runs_dir, learner_name)
    dtw_jsonl = run_dir / "dtw_distances" / "dtw_distances.jsonl"
    learner_dir = learner_pcm_root / f"{learner_name}_recordings"
    output_html = output_dir / f"{learner_name}_pronunciation_feedback_embedded.html"

    if not dtw_jsonl.exists():
        raise FileNotFoundError(f"Missing dtw_distances.jsonl: {dtw_jsonl}")
    if not learner_dir.exists():
        raise FileNotFoundError(f"Missing learner recordings dir: {learner_dir}")

    print(f"[INFO] learner={learner_name}")
    print(f"[INFO] latest run stamp={stamp}")
    print(f"[INFO] selected run dir={run_dir}")
    print(f"[INFO] dtw jsonl={dtw_jsonl}")

    records = load_records(dtw_jsonl)
    audio_uri_map = build_audio_uri_map(records, reference_dir, learner_dir)
    html_text = build_html(records, audio_uri_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    file_size_mb = output_html.stat().st_size / (1024 * 1024)

    print(f"[INFO] embedded audio files={len(audio_uri_map)}")
    print(f"[INFO] html size={file_size_mb:.2f} MB")
    print(f"[DONE] html={output_html}")


if __name__ == "__main__":
    main()
