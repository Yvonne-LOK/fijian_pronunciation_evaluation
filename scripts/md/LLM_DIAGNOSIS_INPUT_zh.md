# LLM 诊断输入说明

## 文件位置

运行 `scripts/mfa_pipeline/run/run_eval_pipeline.sh` 后，会在当前评测输出目录下自动生成：

`llm_diagnosis_input.jsonl`

例如：

`outputs/eval_runs/learner_guoziyu_20260312_101530/llm_diagnosis_input.jsonl`

## 设计目标

这个文件不是原始评分表，而是专门给 LLM 使用的结构化摘要。

核心思路是分层诊断：

1. 先看句子级分数
2. 只有句子级未达阈值时，才下钻到词级
3. 只有定位到低分词后，才继续看词内低分音素

这样可以避免 LLM 直接面对大量原始数字，也避免音素级噪声反过来误导整句判断。

## 诊断规则

默认规则来自 `run_eval_pipeline.sh` 顶部参数：

- `MANUAL_LLM_SENTENCE_THRESHOLD="85"`
- `MANUAL_LLM_WORD_THRESHOLD="75"`
- `MANUAL_LLM_PHONE_THRESHOLD="70"`
- `MANUAL_LLM_MAX_WEAK_WORDS="2"`
- `MANUAL_LLM_MAX_WEAK_PHONES_PER_WORD="2"`

含义如下：

- 当 `sentence_score_new_relative < 85` 时，认为该句需要进一步诊断
- 在该句的词级结果中，挑出低于 75 分的词作为 `weak_words`
- 在每个低分词内部，再挑出低于 70 分的音素作为 `weak_phones`
- 如果低分项过多，只保留最需要关注的前几项

## 每行 JSON 的结构

每一行对应一个句子样本，基本结构如下：

```json
{
  "pair_id": "u1_l1_003__guoziyu_question_002",
  "reference_id": "u1_l1_003",
  "learner_id": "guoziyu_question_002",
  "reference_text": "e dua na siga",
  "status": "ok",
  "diagnosis_policy": {
    "sentence_threshold": 85.0,
    "word_threshold": 75.0,
    "phone_threshold": 70.0,
    "sentence_metric": "sentence_score_new_relative",
    "word_metric": "relative_score_in_utterance",
    "phone_metric": "relative_score_in_word"
  },
  "sentence": {
    "score_relative": 62.4,
    "dtw_norm_distance_dimnorm": 0.27,
    "needs_diagnosis": true,
    "matched_word_segments": 4,
    "matched_phone_segments": 12
  },
  "drill_down_level": "word_phone",
  "strong_words": [],
  "weak_words": [
    {
      "word": "dua",
      "word_label_learner": "dua",
      "seg_idx": 1,
      "word_score_relative": 58.1,
      "word_dtw_norm_distance_dimnorm": 0.29,
      "ref_start": 0.64,
      "ref_end": 0.98,
      "learner_start": 1.02,
      "learner_end": 1.52,
      "weak_phones": [
        {
          "phone": "D",
          "phone_label_learner": "D",
          "seg_idx": 3,
          "phone_score_relative": 41.7,
          "phone_dtw_norm_distance_dimnorm": 0.31,
          "ref_start": 0.64,
          "ref_end": 0.73
        }
      ]
    }
  ],
  "notes": [
    "句子级分数未达到诊断阈值，需要进一步查看词级和音素级证据。"
  ]
}
```

## 关键字段说明

- `sentence.score_relative`
  句子级相对分数，当前是句内可比、批内可比的 0-100 打分。

- `sentence.needs_diagnosis`
  是否需要继续下钻分析。

- `drill_down_level`
  当前记录使用的诊断层级。
  可取值：
  - `sentence_only`：句子整体达标，不继续做负面细分
  - `word_phone`：句子未达标，需要下钻到词和音素

- `strong_words`
  该句中相对表现较好的词，可供 LLM 生成正向反馈。

- `weak_words`
  该句中需要重点关注的词。

- `weak_words[].weak_phones`
  低分词内部进一步定位到的低分音素，用于解释“这个词为什么可能读得不够好”。

## 给 LLM 的推荐用法

建议让 LLM 只基于这个 JSONL 生成反馈，不直接读取整张 CSV。推荐提示词任务：

1. 先总结句子整体表现
2. 如果 `drill_down_level = sentence_only`，以正向反馈为主，最多补充轻微提醒
3. 如果 `drill_down_level = word_phone`，优先指出低分词，再解释对应低分音素
4. 建议输出“哪里读得较好 + 哪里可改进 + 具体练习建议”

## 当前限制

- 词级和音素级分数仍然依赖 MFA 边界与切分质量
- 音素级分数更适合做定位证据，不适合作为整句总判断
- 目前还没有接入音素知识库，所以 LLM 生成具体发音建议时，最好额外提供音素说明表
