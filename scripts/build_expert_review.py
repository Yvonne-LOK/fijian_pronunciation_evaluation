#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_expert_review.py
======================
Extracts 5 cards from each of 6 student HTML files and assembles
a self-contained expert_review.html with Likert questionnaire,
localStorage auto-save, and UTF-8-BOM CSV export.

Usage:
    python build_expert_review.py

Output:
    D:/workdir/scripts/html_demo/expert_review.html
"""

import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent / "html_demo"
OUTPUT   = BASE_DIR / "expert_review.html"

# S1-S6 mapping (pinyin alphabetical order)
STUDENTS = {
    "S1": BASE_DIR / "guoziyu_pronunciation_feedback_embedded.html",
    "S2": BASE_DIR / "jiangshuman_pronunciation_feedback_embedded.html",
    "S3": BASE_DIR / "luoyumeng_pronunciation_feedback_embedded.html",
    "S4": BASE_DIR / "xuhaiyue_pronunciation_feedback_embedded.html",
    "S5": BASE_DIR / "zengjianbin_pronunciation_feedback_embedded.html",
    "S6": BASE_DIR / "zhangle_pronunciation_feedback_embedded.html",
}

# Balanced Incomplete Block Design: 6 students × 5 texts = 30 units
# Each text covered by exactly 3 students; each student covers 5 texts
SAMPLING = {
    "S1": [1, 2, 3, 4, 5],
    "S2": [1, 2, 6, 7, 8],
    "S3": [1, 3, 6, 9, 10],
    "S4": [2, 4, 7, 9, 10],
    "S5": [3, 5, 7, 8, 9],
    "S6": [4, 5, 6, 8, 10],
}

# For each text number, the ordered students that cover it
TEXT_TO_STUDENTS = {}
for sid, texts in SAMPLING.items():
    for t in texts:
        TEXT_TO_STUDENTS.setdefault(t, []).append(sid)

# ── Card extraction ──────────────────────────────────────────────────────────

def extract_cards(html_path: Path) -> dict:
    """
    Return {question_number: card_data_dict} from one student HTML file.
    card_data_dict keys: html, text, translation, verdict, verdict_class
    """
    print(f"  Parsing {html_path.name} …", end=" ", flush=True)
    content = html_path.read_text(encoding="utf-8")

    # Split on article boundaries; each article is one card
    parts = content.split('<article class="card">')
    cards = {}
    for part in parts[1:]:
        end_idx = part.find("</article>")
        if end_idx == -1:
            continue
        block = '<article class="card">' + part[: end_idx + 10]

        # Question number from eyebrow
        m = re.search(r"Question\s+0*(\d+)", block)
        if not m:
            continue
        num = int(m.group(1))

        # Fijian text from <h2>
        h2 = re.search(r"<h2[^>]*>(.*?)</h2>", block, re.S)
        text_fijian = h2.group(1).strip() if h2 else ""

        # Chinese translation
        tr = re.search(r'class="translation"[^>]*>(.*?)</p>', block, re.S)
        translation = tr.group(1).strip() if tr else ""

        # Verdict
        if "status-pass" in block:
            verdict = "通过"
            verdict_class = "status-pass"
        elif "status-fail" in block:
            verdict = "不通过"
            verdict_class = "status-fail"
        else:
            verdict = "待定"
            verdict_class = "status-pending"

        cards[num] = {
            "html": block,
            "text": text_fijian,
            "translation": translation,
            "verdict": verdict,
            "verdict_class": verdict_class,
        }

    print(f"found {len(cards)} cards")
    return cards


def extract_inner_components(card_html: str) -> dict:
    """
    Pull out the reusable sub-blocks from a card's HTML:
      - standard_audio_src  (base64 data URI string)
      - student_audio_src
      - rendered_line_html  (inner HTML of .rendered-line)
      - feedback_html       (inner HTML of .feedback-text)
    """
    # Two <audio> elements in order: first = standard, second = student
    audio_srcs = re.findall(r'<audio[^>]+src="(data:audio[^"]+)"', card_html)
    std_audio  = audio_srcs[0] if len(audio_srcs) > 0 else ""
    stu_audio  = audio_srcs[1] if len(audio_srcs) > 1 else ""

    # Rendered line (everything inside .rendered-line div)
    rl = re.search(r'<div class="rendered-line">(.*?)</div>', card_html, re.S)
    rendered_line = rl.group(1).strip() if rl else ""

    # Feedback text (everything inside .feedback-text p)
    fb = re.search(r'<p class="feedback-text">(.*?)</p>', card_html, re.S)
    feedback = fb.group(1).strip() if fb else ""

    return {
        "std_audio": std_audio,
        "stu_audio": stu_audio,
        "rendered_line": rendered_line,
        "feedback": feedback,
    }


# ── HTML generation ──────────────────────────────────────────────────────────

LIKERT_TEMPLATE = """
<div class="likert-group" data-qid="{unit_id}_q1">
  <div class="likert-label">
    <span class="q-num">Q1</span>
    <span class="q-text">【判定一致性】您是否同意系统给出的"<span class="verdict-inline {vclass}">{verdict}</span>"判定?</span>
    <span class="required-mark">*</span>
  </div>
  <div class="radio-row">
    <label class="radio-opt"><input type="radio" name="{unit_id}_q1" value="1"><span>完全同意</span></label>
    <label class="radio-opt"><input type="radio" name="{unit_id}_q1" value="2"><span>基本同意</span></label>
    <label class="radio-opt"><input type="radio" name="{unit_id}_q1" value="3"><span>不太同意</span></label>
    <label class="radio-opt"><input type="radio" name="{unit_id}_q1" value="4"><span>完全不同意</span></label>
  </div>
</div>
<div class="likert-group" data-qid="{unit_id}_q2">
  <div class="likert-label">
    <span class="q-num">Q2</span>
    <span class="q-text">【音素标注】您认为系统标记的音素颜色(红/黄/绿)是否准确反映了学习者的发音?</span>
    <span class="required-mark">*</span>
  </div>
  <div class="radio-row">
    <label class="radio-opt"><input type="radio" name="{unit_id}_q2" value="1"><span>准确</span></label>
    <label class="radio-opt"><input type="radio" name="{unit_id}_q2" value="2"><span>大部分准确</span></label>
    <label class="radio-opt"><input type="radio" name="{unit_id}_q2" value="3"><span>有明显误标或漏标</span></label>
    <label class="radio-opt"><input type="radio" name="{unit_id}_q2" value="4"><span>完全不符</span></label>
  </div>
</div>
<div class="likert-group" data-qid="{unit_id}_q3">
  <div class="likert-label">
    <span class="q-num">Q3</span>
    <span class="q-text">【LLM反馈正确性】您认为下方的文字反馈在语音学知识上是否正确?</span>
    <span class="required-mark">*</span>
  </div>
  <div class="radio-row">
    <label class="radio-opt"><input type="radio" name="{unit_id}_q3" value="1"><span>完全正确</span></label>
    <label class="radio-opt"><input type="radio" name="{unit_id}_q3" value="2"><span>基本正确，有小瑕疵</span></label>
    <label class="radio-opt"><input type="radio" name="{unit_id}_q3" value="3"><span>有明显错误</span></label>
    <label class="radio-opt"><input type="radio" name="{unit_id}_q3" value="4"><span>无法判断或反馈过于笼统</span></label>
  </div>
</div>
<div class="optional-comment">
  <label class="comment-label" for="{unit_id}_comment">如有任何想补充的，请写在这里（可选）：</label>
  <textarea id="{unit_id}_comment" name="{unit_id}_comment" rows="2" maxlength="500"
    placeholder="选填，不超过 500 字"></textarea>
</div>
"""


def build_unit_card(unit_index: int, text_num: int, student_id: str,
                    card_data: dict) -> str:
    """Build one expert-review assessment card (1-based unit_index)."""
    unit_id   = f"T{text_num:02d}_{student_id}"
    tid_label = f"T{text_num:02d}"
    comps = extract_inner_components(card_data["html"])

    verdict        = card_data["verdict"]
    verdict_class  = card_data["verdict_class"]
    badge_emoji    = "√" if verdict_class == "status-pass" else ("×" if verdict_class == "status-fail" else "?")
    badge_label    = f"{verdict} {badge_emoji}"

    likert = LIKERT_TEMPLATE.format(
        unit_id=unit_id,
        verdict=verdict,
        vclass=verdict_class,
    )

    return f"""
<article class="eval-card" id="card-{unit_id}" data-unit="{unit_id}"
         data-text="{tid_label}" data-student="{student_id}"
         data-verdict="{verdict_class}">
  <div class="eval-card-header">
    <span class="eval-card-index">评估 {unit_index}/30</span>
    <span class="eval-card-tid">{tid_label}</span>
    <span class="eval-card-sid">{student_id}</span>
  </div>

  <div class="card-head">
    <div>
      <div class="eyebrow">文本 {tid_label}</div>
      <h2>{card_data["text"]}</h2>
      <p class="translation">{card_data["translation"]}</p>
    </div>
    <div class="status-chip {verdict_class}">{badge_label}</div>
  </div>

  <div class="audio-grid">
    <div class="audio-box">
      <div class="audio-label">标准音频</div>
      <audio controls preload="none" src="{comps["std_audio"]}"></audio>
    </div>
    <div class="audio-box">
      <div class="audio-label">学生录音（{student_id}）</div>
      <audio controls preload="none" src="{comps["stu_audio"]}"></audio>
    </div>
  </div>

  <div class="rendered-line">{comps["rendered_line"]}</div>

  <section class="feedback-box">
    <div class="feedback-title">LLM 评测反馈</div>
    <p class="feedback-text">{comps["feedback"]}</p>
  </section>

  <div class="questionnaire-block">
    {likert}
  </div>
</article>
"""


def build_overall_section() -> str:
    return """
<section class="overall-section" id="overall-section">
  <h2 class="overall-title">整体评价</h2>

  <div class="likert-group" data-qid="overall_qa">
    <div class="likert-label">
      <span class="q-num">Q-A</span>
      <span class="q-text">综合来看，LLM 反馈的详略程度如何？</span>
      <span class="required-mark">*</span>
    </div>
    <div class="radio-row">
      <label class="radio-opt"><input type="radio" name="overall_qa" value="1"><span>详略得当</span></label>
      <label class="radio-opt"><input type="radio" name="overall_qa" value="2"><span>普遍偏冗长</span></label>
      <label class="radio-opt"><input type="radio" name="overall_qa" value="3"><span>普遍偏简略</span></label>
      <label class="radio-opt"><input type="radio" name="overall_qa" value="4"><span>详略不一，不稳定</span></label>
    </div>
  </div>

  <div class="likert-group" data-qid="overall_qb">
    <div class="likert-label">
      <span class="q-num">Q-B</span>
      <span class="q-text">综合来看，LLM 反馈的语气与口吻是否适合教学场景？</span>
      <span class="required-mark">*</span>
    </div>
    <div class="radio-row">
      <label class="radio-opt"><input type="radio" name="overall_qb" value="1"><span>非常适合</span></label>
      <label class="radio-opt"><input type="radio" name="overall_qb" value="2"><span>基本适合</span></label>
      <label class="radio-opt"><input type="radio" name="overall_qb" value="3"><span>不太适合</span></label>
      <label class="radio-opt"><input type="radio" name="overall_qb" value="4"><span>不适合</span></label>
    </div>
  </div>

  <div class="likert-group" data-qid="overall_qc">
    <div class="likert-label">
      <span class="q-num">Q-C</span>
      <span class="q-text">您认为这套系统对斐济语初学者的发音学习是否有帮助？</span>
      <span class="required-mark">*</span>
    </div>
    <div class="radio-row">
      <label class="radio-opt"><input type="radio" name="overall_qc" value="1"><span>很有帮助</span></label>
      <label class="radio-opt"><input type="radio" name="overall_qc" value="2"><span>有一定帮助</span></label>
      <label class="radio-opt"><input type="radio" name="overall_qc" value="3"><span>帮助有限</span></label>
      <label class="radio-opt"><input type="radio" name="overall_qc" value="4"><span>几乎没有帮助</span></label>
    </div>
  </div>

  <div class="optional-comment">
    <label class="comment-label" for="overall_qd">
      <span class="q-num">Q-D</span>
      您认为系统最需要改进的地方是？（可选，≤200 字）
    </label>
    <textarea id="overall_qd" name="overall_qd" rows="4" maxlength="200"
      placeholder="选填，不超过 200 字"></textarea>
  </div>

  <div class="optional-comment">
    <label class="comment-label" for="overall_qe">
      <span class="q-num">Q-E</span>
      其他任何建议：（可选）
    </label>
    <textarea id="overall_qe" name="overall_qe" rows="4" maxlength="500"
      placeholder="选填"></textarea>
  </div>
</section>
"""


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
    :root {
      --bg: #f3eee4;
      --card: rgba(255,255,255,0.9);
      --ink: #1f2a24;
      --muted: #66756c;
      --border: rgba(31,42,36,0.12);
      --good: #1f8a4c;
      --mid: #c28d12;
      --bad: #c43a2f;
      --neutral: #39423d;
      --pass-bg: rgba(31,138,76,0.12);
      --fail-bg: rgba(196,58,47,0.12);
      --pending-bg: rgba(57,66,61,0.1);
      --shadow: 0 4px 20px rgba(43,37,26,0.10);
      --accent: #2c6e49;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(210,175,132,0.28), transparent 28%),
        radial-gradient(circle at top right, rgba(91,146,115,0.18), transparent 22%),
        linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
      min-height: 100vh;
    }
    .page {
      max-width: 900px;
      margin: 0 auto;
      padding: 32px 20px 80px;
    }

    /* ── Survey header ── */
    .survey-header {
      padding: 32px 36px;
      border-radius: 24px;
      background: linear-gradient(135deg,rgba(255,255,255,0.95),rgba(246,240,231,0.9));
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      margin-bottom: 32px;
    }
    .survey-header h1 {
      margin: 0 0 12px;
      font-size: 26px;
      line-height: 1.3;
      color: var(--accent);
    }
    .survey-desc {
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
      margin: 0 0 20px;
    }
    .name-row {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
    }
    .name-row label {
      font-weight: 700;
      font-size: 15px;
      white-space: nowrap;
    }
    .name-row input {
      flex: 1;
      max-width: 320px;
      padding: 10px 14px;
      border-radius: 10px;
      border: 1.5px solid var(--border);
      font-size: 15px;
      font-family: inherit;
      background: rgba(255,255,255,0.8);
      transition: border-color .2s;
    }
    .name-row input:focus {
      outline: none;
      border-color: var(--accent);
    }
    .name-row input.invalid {
      border-color: var(--bad);
    }
    .progress-wrap {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .progress-label {
      font-size: 14px;
      color: var(--muted);
      white-space: nowrap;
      min-width: 110px;
    }
    .progress-track {
      flex: 1;
      height: 8px;
      background: rgba(31,42,36,0.10);
      border-radius: 999px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      background: var(--accent);
      border-radius: 999px;
      transition: width .3s ease;
      width: 0%;
    }
    .autosave-note {
      font-size: 12px;
      color: var(--muted);
      margin-top: 10px;
    }

    /* ── Text group header ── */
    .text-group-header {
      margin: 36px 0 12px;
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .1em;
      color: var(--muted);
    }
    .text-group-header::before,
    .text-group-header::after {
      content: "";
      flex: 1;
      height: 1px;
      background: var(--border);
    }

    /* ── Eval card ── */
    .eval-card {
      padding: 28px;
      border-radius: 20px;
      border: 1px solid var(--border);
      background: var(--card);
      box-shadow: var(--shadow);
      margin-bottom: 20px;
      backdrop-filter: blur(8px);
      scroll-margin-top: 20px;
    }
    .eval-card.unanswered {
      border-color: rgba(196,58,47,0.35);
      box-shadow: 0 0 0 2px rgba(196,58,47,0.15);
    }
    .eval-card-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 16px;
    }
    .eval-card-index {
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
    }
    .eval-card-tid {
      padding: 3px 10px;
      border-radius: 999px;
      background: rgba(44,110,73,0.10);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    .eval-card-sid {
      padding: 3px 10px;
      border-radius: 999px;
      background: rgba(57,66,61,0.08);
      color: var(--neutral);
      font-size: 12px;
      font-weight: 700;
    }

    /* ── Original card styles (preserved) ── */
    .card-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 16px;
    }
    .eyebrow {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .12em;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .eval-card h2 {
      margin: 0;
      font-size: 26px;
      line-height: 1.25;
    }
    .translation {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 15px;
    }
    .status-chip {
      flex-shrink: 0;
      padding: 8px 14px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 800;
      border: 1px solid rgba(0,0,0,0.06);
    }
    .status-pass { color: var(--good); background: var(--pass-bg); }
    .status-fail { color: var(--bad); background: var(--fail-bg); }
    .status-pending { color: var(--neutral); background: var(--pending-bg); }

    .audio-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px,1fr));
      gap: 10px;
      margin: 0 0 18px;
    }
    .audio-box {
      padding: 12px;
      border-radius: 14px;
      background: rgba(245,240,232,0.92);
      border: 1px solid rgba(31,42,36,0.08);
    }
    .audio-label {
      margin-bottom: 6px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
    }
    audio { width: 100%; }

    .rendered-line {
      font-size: 26px;
      line-height: 1.9;
      letter-spacing: .02em;
      padding: 14px 18px;
      border-radius: 16px;
      background: rgba(255,255,255,0.65);
      border: 1px dashed rgba(31,42,36,0.12);
      margin-bottom: 16px;
      flex-wrap: wrap;
    }
    .word { display: inline-block; padding: 0 4px; border-radius: 6px; }
    .phone-piece { display: inline-block; padding: 0 1px; border-radius: 3px; }
    .word-fragmented { display: inline-block; }
    .tone-good { color: var(--good); background: rgba(31,138,76,0.10); }
    .tone-mid  { color: var(--mid);  background: rgba(194,141,18,0.10); }
    .tone-bad  { color: var(--bad);  background: rgba(196,58,47,0.10); }

    .feedback-box {
      padding: 14px 18px;
      border-radius: 14px;
      background: rgba(57,66,61,0.05);
      border: 1px solid rgba(57,66,61,0.10);
      margin-bottom: 22px;
    }
    .feedback-title {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .feedback-text {
      margin: 0;
      font-size: 15px;
      line-height: 1.7;
      color: var(--ink);
    }

    /* ── Questionnaire block ── */
    .questionnaire-block {
      border-top: 1px solid var(--border);
      padding-top: 20px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .likert-group { display: flex; flex-direction: column; gap: 10px; }
    .likert-label {
      display: flex;
      align-items: baseline;
      flex-wrap: wrap;
      gap: 6px;
      font-size: 15px;
      line-height: 1.5;
    }
    .q-num {
      flex-shrink: 0;
      font-weight: 800;
      color: var(--accent);
      font-size: 13px;
    }
    .q-text { flex: 1; }
    .required-mark { color: var(--bad); font-weight: 700; }
    .verdict-inline {
      display: inline-block;
      padding: 1px 8px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 13px;
    }
    .radio-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .radio-opt {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 8px 14px;
      border-radius: 10px;
      border: 1.5px solid var(--border);
      cursor: pointer;
      font-size: 14px;
      background: rgba(255,255,255,0.7);
      transition: border-color .15s, background .15s;
      user-select: none;
    }
    .radio-opt:hover {
      border-color: var(--accent);
      background: rgba(44,110,73,0.06);
    }
    .radio-opt input[type="radio"] {
      width: 16px; height: 16px;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .radio-opt:has(input:checked) {
      border-color: var(--accent);
      background: rgba(44,110,73,0.10);
      font-weight: 600;
    }

    .optional-comment { display: flex; flex-direction: column; gap: 6px; }
    .comment-label {
      font-size: 14px;
      color: var(--muted);
      display: flex;
      align-items: baseline;
      flex-wrap: wrap;
      gap: 6px;
    }
    textarea {
      padding: 10px 14px;
      border-radius: 10px;
      border: 1.5px solid var(--border);
      font-size: 14px;
      font-family: inherit;
      resize: vertical;
      background: rgba(255,255,255,0.7);
      line-height: 1.6;
      transition: border-color .2s;
    }
    textarea:focus {
      outline: none;
      border-color: var(--accent);
    }

    /* ── Overall section ── */
    .overall-section {
      margin-top: 48px;
      padding: 32px 36px;
      border-radius: 24px;
      background: linear-gradient(135deg,rgba(255,255,255,0.95),rgba(246,240,231,0.9));
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      gap: 22px;
    }
    .overall-title {
      margin: 0 0 4px;
      font-size: 20px;
      color: var(--accent);
    }

    /* ── Footer ── */
    .survey-footer {
      margin-top: 40px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 14px;
    }
    .export-btn {
      padding: 16px 48px;
      border-radius: 14px;
      background: var(--accent);
      color: #fff;
      font-size: 17px;
      font-weight: 700;
      border: none;
      cursor: pointer;
      transition: opacity .2s, transform .1s;
      font-family: inherit;
    }
    .export-btn:hover:not(:disabled) { opacity: 0.88; transform: translateY(-1px); }
    .export-btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }
    .export-hint {
      font-size: 13px;
      color: var(--muted);
    }
    .autosave-footer {
      font-size: 13px;
      color: var(--good);
      font-weight: 600;
    }

    @media (max-width: 600px) {
      .survey-header, .overall-section { padding: 20px 18px; }
      .eval-card { padding: 18px 16px; }
      .rendered-line { font-size: 22px; }
      .radio-row { flex-direction: column; }
    }
"""


# ── JavaScript ───────────────────────────────────────────────────────────────

JS = r"""
  const STORAGE_KEY = 'fijian_expert_review_v1';
  const REQUIRED_RADIO_COUNT = 33; // 30 cards × 3 + 3 overall (qa/qb/qc)

  // All required radio names
  const UNIT_IDS = """ + "UNIT_IDS_PLACEHOLDER" + r""";
  const REQUIRED_NAMES = [
    ...UNIT_IDS.flatMap(uid => [`${uid}_q1`,`${uid}_q2`,`${uid}_q3`]),
    'overall_qa','overall_qb','overall_qc'
  ];

  // ── Persistence ────────────────────────────────────────────────────────────
  function saveState() {
    const data = { reviewer: document.getElementById('reviewer-name').value };
    document.querySelectorAll('input[type="radio"]:checked').forEach(r => {
      data[r.name] = r.value;
    });
    document.querySelectorAll('textarea').forEach(ta => {
      if (ta.id) data[ta.id] = ta.value;
    });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    updateProgress();
  }

  function loadState() {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    let data;
    try { data = JSON.parse(raw); } catch(e) { return; }
    if (data.reviewer) document.getElementById('reviewer-name').value = data.reviewer;
    Object.entries(data).forEach(([name, val]) => {
      const radio = document.querySelector(`input[type="radio"][name="${name}"][value="${val}"]`);
      if (radio) radio.checked = true;
      const ta = document.getElementById(name);
      if (ta && ta.tagName === 'TEXTAREA') ta.value = val;
    });
    updateProgress();
  }

  // ── Progress ───────────────────────────────────────────────────────────────
  function countAnswered() {
    return REQUIRED_NAMES.filter(name =>
      document.querySelector(`input[type="radio"][name="${name}"]:checked`)
    ).length;
  }

  function updateProgress() {
    const done  = countAnswered();
    const total = REQUIRED_RADIO_COUNT;
    document.getElementById('progress-count').textContent = done;
    document.getElementById('progress-fill').style.width = (done / total * 100) + '%';

    const btn = document.getElementById('export-btn');
    if (done >= total && document.getElementById('reviewer-name').value.trim()) {
      btn.disabled = false;
      btn.title = '';
    } else {
      btn.disabled = true;
      const missing = total - done;
      const nameMissing = !document.getElementById('reviewer-name').value.trim();
      btn.title = nameMissing ? '请先填写您的姓名' : `还有 ${missing} 题未答`;
    }
  }

  // ── CSV export ─────────────────────────────────────────────────────────────
  function getRadioVal(name) {
    const el = document.querySelector(`input[type="radio"][name="${name}"]:checked`);
    return el ? el.value : '';
  }
  function getTextVal(id) {
    const el = document.getElementById(id);
    return el ? el.value.replace(/"/g,'""') : '';
  }

  function buildCSV() {
    const reviewer = document.getElementById('reviewer-name').value.trim();
    const now = new Date().toISOString();

    const rows = [];

    // Comment row: legend
    rows.push('# 答案编码说明: Q1(判定一致性) 1=完全同意 2=基本同意 3=不太同意 4=完全不同意; ' +
              'Q2(音素标注) 1=准确 2=大部分准确 3=有明显误标或漏标 4=完全不符; ' +
              'Q3(LLM反馈正确性) 1=完全正确 2=基本正确有小瑕疵 3=有明显错误 4=无法判断或过于笼统; ' +
              'Q-A 1=详略得当 2=普遍偏冗长 3=普遍偏简略 4=详略不一不稳定; ' +
              'Q-B 1=非常适合 2=基本适合 3=不太适合 4=不适合; ' +
              'Q-C 1=很有帮助 2=有一定帮助 3=帮助有限 4=几乎没有帮助');

    // Header row
    rows.push([
      'reviewer_name','submit_time','record_type','unit_id','text_id','student_id',
      'text_content','system_verdict',
      'q1_judgment_agreement','q2_phoneme_annotation','q3_llm_correctness','optional_comment'
    ].join(','));

    // Unit rows
    UNIT_IDS.forEach(uid => {
      const card = document.getElementById(`card-${uid}`);
      const textId   = card.dataset.text;
      const student  = card.dataset.student;
      const h2       = card.querySelector('h2');
      const textContent = h2 ? h2.textContent.trim().replace(/"/g,'""') : '';
      const verdictClass = card.dataset.verdict;
      const verdict = verdictClass === 'status-pass' ? '通过' : (verdictClass === 'status-fail' ? '不通过' : '待定');

      const row = [
        `"${reviewer}"`,
        `"${now}"`,
        '"unit"',
        `"${uid}"`,
        `"${textId}"`,
        `"${student}"`,
        `"${textContent}"`,
        `"${verdict}"`,
        getRadioVal(`${uid}_q1`),
        getRadioVal(`${uid}_q2`),
        getRadioVal(`${uid}_q3`),
        `"${getTextVal(`${uid}_comment`)}"`
      ].join(',');
      rows.push(row);
    });

    // Overall row
    rows.push([
      `"${reviewer}"`,
      `"${now}"`,
      '"overall"',
      '"overall"','"—"','"—"','"—"','"—"',
      '""','""','"—"',
      `"${getTextVal('overall_qd')}"`,
    ].join(',') + `,"${getTextVal('overall_qe')}",${getRadioVal('overall_qa')},${getRadioVal('overall_qb')},${getRadioVal('overall_qc')}`);

    // The overall row needs extra columns — fix schema uniformity with a proper header
    // Actually re-do with a unified wide schema:
    return buildCSVUnified(reviewer, now);
  }

  function buildCSVUnified(reviewer, now) {
    const rows = [];

    rows.push('# 答案编码: Q1 1=完全同意 2=基本同意 3=不太同意 4=完全不同意 | ' +
              'Q2 1=准确 2=大部分准确 3=有明显误标/漏标 4=完全不符 | ' +
              'Q3 1=完全正确 2=基本正确有小瑕疵 3=有明显错误 4=无法判断/过于笼统 | ' +
              'QA 1=详略得当 2=偏冗长 3=偏简略 4=详略不一 | ' +
              'QB 1=非常适合 2=基本适合 3=不太适合 4=不适合 | ' +
              'QC 1=很有帮助 2=有一定帮助 3=帮助有限 4=几乎没有帮助');

    rows.push([
      'reviewer_name','submit_time','record_type',
      'unit_id','text_id','student_id','text_content','system_verdict',
      'q1_judgment_agreement','q2_phoneme_annotation','q3_llm_correctness','optional_comment',
      'qa_feedback_length','qb_feedback_tone','qc_system_helpfulness',
      'qd_improvement','qe_other_suggestions'
    ].join(','));

    UNIT_IDS.forEach(uid => {
      const card = document.getElementById(`card-${uid}`);
      const textId  = card.dataset.text;
      const student = card.dataset.student;
      const h2      = card.querySelector('h2');
      const tc      = h2 ? h2.textContent.trim().replace(/"/g,'""') : '';
      const vc      = card.dataset.verdict;
      const verdict = vc === 'status-pass' ? '通过' : (vc === 'status-fail' ? '不通过' : '待定');

      rows.push([
        `"${reviewer}"`,`"${now}"`,'unit',
        `"${uid}"`,`"${textId}"`,`"${student}"`,`"${tc}"`,`"${verdict}"`,
        getRadioVal(`${uid}_q1`),
        getRadioVal(`${uid}_q2`),
        getRadioVal(`${uid}_q3`),
        `"${getTextVal(`${uid}_comment`)}"`,
        '','','','',''
      ].join(','));
    });

    // Overall row
    rows.push([
      `"${reviewer}"`,`"${now}"`,'overall',
      '"overall"','"—"','"—"','"—"','"—"',
      '','','','',
      getRadioVal('overall_qa'),
      getRadioVal('overall_qb'),
      getRadioVal('overall_qc'),
      `"${getTextVal('overall_qd')}"`,
      `"${getTextVal('overall_qe')}"`
    ].join(','));

    return rows.join('\r\n');
  }

  function downloadCSV() {
    // Validate required fields
    const name = document.getElementById('reviewer-name').value.trim();
    if (!name) {
      alert('请先填写您的姓名！');
      document.getElementById('reviewer-name').focus();
      document.getElementById('reviewer-name').classList.add('invalid');
      return;
    }
    const missing = REQUIRED_NAMES.filter(n =>
      !document.querySelector(`input[type="radio"][name="${n}"]:checked`)
    );
    if (missing.length > 0) {
      // Find first unanswered card
      const firstMissingName = missing[0];
      const firstMissingEl = document.querySelector(`[data-qid="${firstMissingName}"]`);
      if (firstMissingEl) {
        firstMissingEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Highlight containing card
        const card = firstMissingEl.closest('.eval-card, .overall-section');
        if (card) {
          card.classList.add('unanswered');
          setTimeout(() => card.classList.remove('unanswered'), 3000);
        }
      }
      alert(`还有 ${missing.length} 个必答题未填写，已为您定位到第一个。`);
      return;
    }
    const csv = buildCSV();
    // UTF-8 BOM for Excel compatibility
    const bom = '﻿';
    const blob = new Blob([bom + csv], { type: 'text/csv;charset=utf-8' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `expert_review_${name}_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ── Init ───────────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    loadState();

    document.getElementById('reviewer-name').addEventListener('input', () => {
      document.getElementById('reviewer-name').classList.remove('invalid');
      saveState();
    });

    document.querySelectorAll('input[type="radio"]').forEach(r => {
      r.addEventListener('change', saveState);
    });
    document.querySelectorAll('textarea').forEach(ta => {
      ta.addEventListener('input', saveState);
    });

    document.getElementById('export-btn').addEventListener('click', downloadCSV);

    updateProgress();
  });
"""


# ── Main assembly ─────────────────────────────────────────────────────────────

def main():
    print("=== build_expert_review.py ===")
    print(f"Output: {OUTPUT}\n")

    # 1. Parse all student files
    all_cards = {}
    for sid, path in STUDENTS.items():
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        all_cards[sid] = extract_cards(path)

    # 2. Build list of 30 evaluation units (ordered by text number)
    units = []
    for t_num in range(1, 11):
        for sid in TEXT_TO_STUDENTS[t_num]:
            units.append((t_num, sid))

    # Verify 30 units
    assert len(units) == 30, f"Expected 30 units, got {len(units)}"

    # 3. Build all card HTML blocks
    print("\nBuilding assessment cards …")
    card_blocks = []
    unit_ids    = []
    current_text = None

    for idx, (t_num, sid) in enumerate(units, start=1):
        card_data = all_cards[sid].get(t_num)
        if card_data is None:
            print(f"  WARNING: S{sid} has no card for T{t_num:02d}", file=sys.stderr)
            continue

        # Insert text-group separator
        if t_num != current_text:
            current_text = t_num
            card_blocks.append(
                f'<div class="text-group-header">文本 T{t_num:02d}</div>\n'
            )

        unit_id = f"T{t_num:02d}_{sid}"
        unit_ids.append(unit_id)
        card_blocks.append(build_unit_card(idx, t_num, sid, card_data))
        print(f"  [{idx:02d}/30] T{t_num:02d}_{sid}")

    # 4. Inject unit IDs into JS
    unit_ids_js = "[" + ",".join(f'"{u}"' for u in unit_ids) + "]"
    js_final = JS.replace("UNIT_IDS_PLACEHOLDER", unit_ids_js)

    # 5. Assemble final HTML
    print("\nAssembling expert_review.html …")
    cards_html = "\n".join(card_blocks)
    overall    = build_overall_section()

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>斐济语发音评测系统 - 专家评估问卷</title>
  <style>
{CSS}
  </style>
</head>
<body>
<div class="page">

  <!-- ── Survey header ── -->
  <div class="survey-header">
    <h1>斐济语发音评测系统 — 专家评估问卷</h1>
    <p class="survey-desc">
      感谢您参与本次评估。问卷共包含 30 个学生录音评估单元及 5 道整体评价题，预计需要 25–30 分钟。
      您的答案会自动保存到本地浏览器，关闭窗口后重新打开仍可继续填写。
      填写完毕后，请点击底部"导出 CSV"按钮，将文件发送给我。
    </p>
    <div class="name-row">
      <label for="reviewer-name">您的姓名 <span style="color:var(--bad)">*</span></label>
      <input type="text" id="reviewer-name" placeholder="请输入您的姓名（将写入 CSV）" autocomplete="name" />
    </div>
    <div class="progress-wrap">
      <span class="progress-label">已完成 <strong id="progress-count">0</strong> / 33 题</span>
      <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
    </div>
    <p class="autosave-note">✓ 答案已自动保存到本地，关闭窗口不会丢失进度</p>
  </div>

  <!-- ── Legend ── -->
  <div style="margin-bottom:24px; display:flex; flex-wrap:wrap; gap:10px; font-size:13px;">
    <span style="padding:5px 12px; border-radius:999px; font-weight:700;
                 color:var(--good); background:var(--pass-bg); border:1px solid rgba(0,0,0,0.06);">
      绿色：发音很好</span>
    <span style="padding:5px 12px; border-radius:999px; font-weight:700;
                 color:var(--mid); background:rgba(194,141,18,0.10); border:1px solid rgba(0,0,0,0.06);">
      黄色：发音达标</span>
    <span style="padding:5px 12px; border-radius:999px; font-weight:700;
                 color:var(--bad); background:var(--fail-bg); border:1px solid rgba(0,0,0,0.06);">
      红色：发音有误</span>
  </div>

  <!-- ── Assessment cards ── -->
  {cards_html}

  <!-- ── Overall evaluation ── -->
  {overall}

  <!-- ── Footer ── -->
  <div class="survey-footer">
    <button id="export-btn" class="export-btn" disabled title="还有题目未答">
      导出 CSV
    </button>
    <span class="export-hint">导出前将校验所有必答题；CSV 使用 UTF-8 with BOM（Excel 兼容）</span>
    <span class="autosave-footer">✓ 答案已自动保存，随时可以关闭后继续</span>
  </div>

</div>

<script>
{js_final}
</script>
</body>
</html>
"""

    OUTPUT.write_text(html, encoding="utf-8")
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"\nDone! {OUTPUT}  ({size_mb:.1f} MB)")
    print(f"\nSampling matrix (answer to user):")
    print("  {")
    for sid, texts in SAMPLING.items():
        num = sid[1]
        name = list(STUDENTS.values())[int(num)-1].stem.replace('_pronunciation_feedback_embedded','')
        print(f'    {num}: {texts},  # {name}')
    print("  }")


if __name__ == "__main__":
    main()
