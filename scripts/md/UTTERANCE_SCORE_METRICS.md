# Utterance Score Metrics

This document explains the fields written to `utterance_scores.csv` by the current MFA + GFCC + DTW pipeline.

## Scope

`utterance_scores.csv` is one row per reference-learner utterance pair.

The row is produced in:

- [batch_gfcc_dtw_eval.py](/d:/workdir/scripts/mfa_pipeline/run/batch_gfcc_dtw_eval.py)

The scoring pipeline is:

1. MFA alignment produces `TextGrid` files.
2. The script cuts phone-level and word-level segments from reference and learner audio.
3. GFCC features are extracted.
4. DTW is computed on:
   - the whole utterance
   - trimmed whole utterance
   - active speech interval from aligned phones
   - each matched word segment
   - each matched phone segment

## Identifier Fields

- `pair_id`
  The pair identifier used for this row. If `pair_csv` contains `pair_id`, that value is used. Otherwise it is generated from `reference_id` and `learner_id`.

- `reference_id`
  Basename of the reference utterance, without extension.

- `learner_id`
  Basename of the learner utterance, without extension.

- `status`
  Overall processing result for the utterance pair.
  Usually:
  - `ok`: scoring completed
  - `missing_input`: one or more required files were not found

- `missing_inputs`
  Only appears for failed rows such as `missing_input`.
  Semicolon-separated missing file paths.

## Sentence-Level Metrics

These are computed from full-utterance audio, not from averaged word or phone scores.

Important:

- the core score columns in this pipeline are already mapped to a 0-100 scale
- new `*_percent` columns are string-formatted versions for easier reading
- new `*_grade` columns map scores into qualitative bands

- `sentence_score_raw`
  Sentence-level GFCC+DTW score using the entire reference waveform and entire learner waveform as-is.

  Meaning:
  - includes leading silence, trailing silence, and pauses
  - most sensitive to recording start/end blank regions

- `sentence_score_trimmed`
  Sentence-level GFCC+DTW score after trimming leading and trailing silence with `librosa.effects.trim`.

  Meaning:
  - reduces the effect of obvious head/tail silence
  - still keeps internal pauses

- `sentence_score_active`
  Sentence-level GFCC+DTW score after cropping both utterances to the active speech interval defined by aligned phone boundaries.

  Calculation:
  - reference crop: from `min(phone.start)` to `max(phone.end)` in the reference `TextGrid`
  - learner crop: from `min(phone.start)` to `max(phone.end)` in the learner `TextGrid`

  Meaning:
  - focuses on the speech region only
  - less affected by recording blank space
  - this is the recommended sentence-level score in the current pipeline

- `sentence_score`
  Current canonical sentence-level score.

  At present:
  - `sentence_score = sentence_score_active`

- `sentence_score_percent`
  Percentage-formatted display of `sentence_score`, for example `82.37%`.

- `sentence_score_grade`
  Qualitative band of `sentence_score`.

  Current mapping:
  - below 60: `不合格`
  - 60 to below 70: `合格`
  - 70 to below 85: `良好`
  - 85 and above: `优秀`

- `sentence_dtw_raw_distance`
  Raw DTW cumulative distance for the current canonical sentence score.

  Meaning:
  - lower is better
  - not directly comparable across tasks of very different duration unless normalized

- `sentence_dtw_norm_distance`
  Sentence-level DTW distance normalized by DTW path length.

  Calculation:
  - `norm_distance = raw_distance / path_len`

  Meaning:
  - lower is better
  - more comparable than raw distance across different utterance lengths

- `sentence_dtw_path_len`
  Length of the optimal DTW alignment path for the canonical sentence score.

- `sentence_ref_num_frames`
  Number of GFCC frames extracted from the reference utterance used in the canonical sentence score.

- `sentence_learner_num_frames`
  Number of GFCC frames extracted from the learner utterance used in the canonical sentence score.

## Aggregated Segment Scores

These summarize the scores from phone-level and word-level matched segments.

- `phone_mean_score`
  Mean score across all phone segments whose reference and learner segment pair was successfully scored.

  Calculation:
  - each phone segment is scored with GFCC + DTW
  - mean is taken across rows where `status == ok` in `phone_scores.csv`

- `word_mean_score`
  Mean score across all word segments whose reference and learner segment pair was successfully scored.

  Calculation:
  - each word segment is scored with GFCC + DTW
  - mean is taken across rows where `status == ok` in `word_scores.csv`

- `phone_mean_score_percent`
  Percentage-formatted display of `phone_mean_score`.

- `phone_mean_score_grade`
  Qualitative band of `phone_mean_score` using the same 60/70/85 thresholds.

- `word_mean_score_percent`
  Percentage-formatted display of `word_mean_score`.

- `word_mean_score_grade`
  Qualitative band of `word_mean_score` using the same 60/70/85 thresholds.

- `overall_score`
  Overall utterance score computed as the mean of:
  - `sentence_score`
  - `phone_mean_score`
  - `word_mean_score`

  Meaning:
  - provides a single combined percentage score for the utterance
  - useful for reporting when you want one headline result

- `overall_score_percent`
  Percentage-formatted display of `overall_score`.

- `overall_score_grade`
  Qualitative band of `overall_score` using the same 60/70/85 thresholds.

## Segment Count and Quality Diagnostics

These fields help explain whether a low score may come from alignment mismatch rather than pure pronunciation quality.

- `matched_phone_segments`
  Number of phone rows successfully scored.

  Roughly:
  - count of phone rows with `status == ok`

- `matched_word_segments`
  Number of word rows successfully scored.

- `mismatched_phone_labels`
  Number of aligned phone pairs where the reference label and learner label are not the same.

  Meaning:
  - often indicates alignment inconsistency
  - can also reflect recognition/alignment errors rather than learner pronunciation alone

- `mismatched_word_labels`
  Number of aligned word pairs where the reference word label and learner word label are not the same.

- `missing_phone_segments`
  Number of phone rows that were not successfully scored.

  Includes cases such as:
  - segment exists only on one side
  - feature extraction failed
  - segment audio invalid

- `missing_word_segments`
  Number of word rows that were not successfully scored.

## Core DTW Score Formula

All segment-level and sentence-level scores in this pipeline are derived from the same mapping:

1. Extract GFCC features.
2. Z-normalize the feature matrix by feature dimension.
3. Compute DTW cumulative distance with Euclidean local distance.
4. Normalize by DTW path length:

```text
norm_distance = raw_distance / path_len
```

5. Convert normalized distance to a 0-100 score:

```text
score = 100 * exp(-alpha * norm_distance / beta)
```

Where the current defaults are:

- `alpha = 8.0`
- `beta = 0.6`

Interpretation:

- higher score is better
- lower DTW distance gives higher score
- score is clipped to `[0, 100]`

## How To Read A Row

Recommended reading order:

1. Check `status`
   If not `ok`, the row is not usable for scoring comparison.

2. Check `sentence_score`
   This is the main utterance-level score to use in the current pipeline.

3. Compare `sentence_score_raw`, `sentence_score_trimmed`, and `sentence_score_active`
   This helps judge whether silence or head/tail blank space is affecting the utterance-level result.

4. Check `phone_mean_score` and `word_mean_score`
   These show average local pronunciation similarity.

5. Check mismatch and missing counters
   If these are high, low scores may partly come from alignment or segmentation issues.

## Practical Notes

- `sentence_score` and `phone_mean_score`/`word_mean_score` answer different questions.
  - `sentence_score` reflects whole-utterance acoustic similarity
  - `phone_mean_score` and `word_mean_score` reflect average local segment similarity

- A learner can have:
  - decent segment scores but lower sentence score
    if rhythm, pacing, or pauses differ a lot
  - low segment means with moderate sentence score
    if global contour is similar but local articulation is weak

- High `mismatched_*` or `missing_*` counts should be treated as a warning flag before interpreting the score pedagogically.
