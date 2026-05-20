# Utterance Score 指标说明

本文档说明当前流程生成的 `utterance_scores.csv` 中各字段的含义、计算方式和解读建议。

英文版对应文件：

- [UTTERANCE_SCORE_METRICS.md](/d:/workdir/scripts/UTTERANCE_SCORE_METRICS.md)

## 1. 适用范围

`utterance_scores.csv` 以“每对参考音频-学习者音频”为一行。

当前行级评分主要由以下脚本生成：

- [batch_gfcc_dtw_eval.py](/d:/workdir/scripts/mfa_pipeline/run/batch_gfcc_dtw_eval.py)

当前整体流程是：

1. MFA 对齐，生成 `TextGrid`
2. 根据 `TextGrid` 切出 phone 和 word 片段
3. 提取 GFCC 特征
4. 对整句、词、音素分别做 DTW
5. 输出句级、词级、音素级及诊断字段

## 2. 总体说明

需要先说明一点：

- 当前主要分数字段本身已经是 **0 到 100 的百分制分数**
- 它们不是原始 DTW 距离
- 新增的 `*_percent` 字段只是把数值格式化成了更直观的百分号字符串
- 新增的 `*_grade` 字段是把分数映射到等级

当前等级映射规则为：

- 小于 60：`不合格`
- 60 到小于 70：`合格`
- 70 到小于 85：`良好`
- 大于等于 85：`优秀`

## 3. 标识类字段

- `pair_id`
  该行样本对的标识。
  如果 `pair_csv` 中提供了 `pair_id`，则使用该值；否则由 `reference_id` 和 `learner_id` 自动拼接生成。

- `reference_id`
  参考音频 basename，不含扩展名。

- `learner_id`
  学习者音频 basename，不含扩展名。

- `status`
  当前这对样本的整体处理状态。

  常见取值：
  - `ok`：处理和评分成功
  - `missing_input`：缺少必要输入文件

- `missing_inputs`
  仅在失败行中出现。
  表示缺失文件路径，多个路径用分号分隔。

## 4. 句级指标

句级指标是直接基于整句音频计算的，不是由词级或音素级平均得到。

### 4.1 `sentence_score_raw`

整句原始句级分数。

计算方式：

- 直接用整条参考音频和整条学习者音频提取 GFCC
- 然后做 DTW
- 再把 DTW 距离映射为 0 到 100 分

特点：

- 会包含开头静音、结尾静音、内部停顿
- 对录音头尾空白更敏感

### 4.2 `sentence_score_trimmed`

去掉首尾静音后的整句句级分数。

计算方式：

- 先对参考音频和学习者音频分别做首尾静音裁剪
- 再做 GFCC + DTW + 百分制映射

特点：

- 能减弱明显录音头尾空白的影响
- 但仍保留内部停顿

### 4.3 `sentence_score_active`

基于 `TextGrid` 中有效发音区间裁剪后的句级分数。

计算方式：

- 对参考音频：
  使用参考 `TextGrid` 中非静音 phone 的最早开始时间到最晚结束时间
- 对学习者音频：
  使用学习者 `TextGrid` 中非静音 phone 的最早开始时间到最晚结束时间
- 然后在这两个裁剪后的有效发音区间上做 GFCC + DTW + 百分制映射

特点：

- 更聚焦于真正的发音内容
- 受录音前后空白的影响最小
- 是当前流程里最推荐的句级分数

### 4.4 `sentence_score`

当前默认使用的句级总分。

目前定义为：

- `sentence_score = sentence_score_active`

### 4.5 `sentence_score_percent`

`sentence_score` 的百分号展示形式，例如：

- `82.37%`

### 4.6 `sentence_score_grade`

`sentence_score` 对应的等级结果。

### 4.7 `sentence_score_raw_percent`

`sentence_score_raw` 的百分号展示形式。

### 4.8 `sentence_score_trimmed_percent`

`sentence_score_trimmed` 的百分号展示形式。

### 4.9 `sentence_score_active_percent`

`sentence_score_active` 的百分号展示形式。

### 4.10 `sentence_dtw_raw_distance`

当前默认句级分数对应的原始 DTW 累积距离。

说明：

- 越小越好
- 不同长度句子之间不宜直接比较原始距离

### 4.11 `sentence_dtw_norm_distance`

当前默认句级分数对应的归一化 DTW 距离。

计算方式：

```text
norm_distance = raw_distance / path_len
```

说明：

- 越小越好
- 相比原始距离，更适合跨不同句长比较

### 4.12 `sentence_dtw_path_len`

当前默认句级分数对应的 DTW 最优路径长度。

### 4.13 `sentence_ref_num_frames`

当前默认句级分数中，参考音频提取出的 GFCC 帧数。

### 4.14 `sentence_learner_num_frames`

当前默认句级分数中，学习者音频提取出的 GFCC 帧数。

## 5. 词级和音素级平均分

这些字段是对词级或音素级片段得分取平均后的结果。

### 5.1 `phone_mean_score`

音素级平均分。

计算方式：

- 对每个成功匹配的 phone 片段做 GFCC + DTW 评分
- 对所有 `status == ok` 的 phone 片段分数求平均

### 5.2 `phone_mean_score_percent`

`phone_mean_score` 的百分号展示形式。

### 5.3 `phone_mean_score_grade`

`phone_mean_score` 对应的等级。

### 5.4 `word_mean_score`

词级平均分。

计算方式：

- 对每个成功匹配的 word 片段做 GFCC + DTW 评分
- 对所有 `status == ok` 的 word 片段分数求平均

### 5.5 `word_mean_score_percent`

`word_mean_score` 的百分号展示形式。

### 5.6 `word_mean_score_grade`

`word_mean_score` 对应的等级。

## 6. 综合总评字段

### 6.1 `overall_score`

整句综合总评。

计算方式：

- 取以下三项的平均值：
  - `sentence_score`
  - `phone_mean_score`
  - `word_mean_score`

作用：

- 适合在需要“单一总分”时作为汇总分数使用
- 兼顾整句整体相似度和局部发音质量

### 6.2 `overall_score_percent`

`overall_score` 的百分号展示形式。

### 6.3 `overall_score_grade`

`overall_score` 对应的等级。

## 7. 片段匹配与诊断字段

这些字段主要用于辅助判断“低分到底是发音差，还是对齐/分段出了问题”。

### 7.1 `matched_phone_segments`

成功完成评分的音素片段数量。

通常等价于：

- `phone_scores.csv` 中 `status == ok` 的行数

### 7.2 `matched_word_segments`

成功完成评分的词片段数量。

### 7.3 `mismatched_phone_labels`

参考音素标签和学习者音素标签不一致的片段数量。

说明：

- 如果这个值较高，往往意味着对齐不稳定
- 也可能说明分段错位，而不一定只是学习者发音差

### 7.4 `mismatched_word_labels`

参考词标签和学习者词标签不一致的片段数量。

### 7.5 `missing_phone_segments`

未成功完成评分的音素片段数量。

可能原因包括：

- 仅一侧存在该片段
- 特征提取失败
- 音频片段无效

### 7.6 `missing_word_segments`

未成功完成评分的词片段数量。

## 8. 当前分数的核心计算公式

当前整句、词、音素评分本质上都遵循同一套计算逻辑：

1. 提取 GFCC 特征
2. 对特征按维度做 z-normalization
3. 用欧氏距离作为局部距离，计算 DTW
4. 用路径长度做归一化：

```text
norm_distance = raw_distance / path_len
```

5. 再把归一化距离映射到 0 到 100：

```text
score = 100 * exp(-alpha * norm_distance / beta)
```

当前默认参数：

- `alpha = 8.0`
- `beta = 0.6`

解释：

- 分数越高越好
- DTW 距离越小，分数越高
- 最终结果会被裁剪到 `[0, 100]`

## 9. 如何解读一行结果

建议按这个顺序看：

1. 先看 `status`
   如果不是 `ok`，这行不能作为正常评分解释。

2. 看 `sentence_score`
   这是当前流程默认推荐的句级分数。

3. 比较这三项：
   - `sentence_score_raw`
   - `sentence_score_trimmed`
   - `sentence_score_active`

   如果三者差异很大，通常说明静音或头尾空白对句级结果影响明显。

4. 看：
   - `phone_mean_score`
   - `word_mean_score`

   它们反映局部发音质量。

5. 看诊断字段：
   - `mismatched_phone_labels`
   - `mismatched_word_labels`
   - `missing_phone_segments`
   - `missing_word_segments`

   如果这些值偏高，则低分可能部分来自对齐或分段问题，而不完全是学习者发音本身。

6. 如果需要一个汇总结论，再看：
   - `overall_score`
   - `overall_score_grade`

## 10. 实际使用建议

- 如果你只想保留一个句级分数，建议优先使用：
  - `sentence_score`

- 如果你需要一个最终总评，建议使用：
  - `overall_score`
  - `overall_score_grade`

- 如果你做教学反馈：
  - `sentence_score` 更适合看整句整体表现
  - `word_mean_score` 和 `phone_mean_score` 更适合看局部发音细节

- 如果 `mismatched_*` 或 `missing_*` 偏高，建议先检查对齐质量，再解释分数。
