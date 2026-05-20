# MFA-GFCC-DTW 一键评测脚本说明

## 1. 目标

`run_eval_pipeline.sh` 用来把下面这条链路串起来：

1. 对参考音频和学习者音频跑 MFA 对齐，生成 `TextGrid`
2. 按 `words` / `phones` tier 做细粒度切分
3. 对每个切分片段提取 GFCC 特征
4. 用 DTW 计算参考发音和学习者发音的距离
5. 输出句级、词级、音素级评测分数

核心入口：

```bash
bash scripts/mfa_pipeline/run/run_eval_pipeline.sh
```

## 2. 脚本顶部手动配置

`run_eval_pipeline.sh` 最开头有一组 `MANUAL_*` 变量，专门给你手动填写参数。

优先级是：

1. 脚本顶部手动填写的 `MANUAL_*`
2. 当前 shell 里的环境变量
3. 脚本内置默认值

也就是说，只要某个 `MANUAL_*` 不是空字符串，它就会覆盖环境变量里的同名设置。

例如你可以直接在脚本顶部这样写：

```bash
MANUAL_PAIR_CSV="/mnt/d/workdir/scripts/mfa_pipeline/run/pairs.csv"
MANUAL_LEARNER_CORPUS_DIR="/mnt/d/workdir/data/fijian/new_learner"
MANUAL_RUN_LEARNER_ALIGN="1"
MANUAL_RUN_TAG="learner_001_20260311"
```

这样以后直接执行脚本即可，不需要每次临时 `export`。

## 3. 预先准备的数据

脚本默认依赖下面这些输入已经准备好：

1. `REFERENCE_CORPUS_DIR`
   默认：`data/fijian/raw_corpus`
   这里放参考/标准录音，每条语音至少有：
   - `xxx.wav`
   - `xxx.txt`

2. `LEARNER_CORPUS_DIR`
   默认：`data/fijian/learner_pcm`
   这里放某个学习者的一批跟读录音，每条语音至少有：
   - `xxx.wav`
   - `xxx.txt`

3. `DICTIONARY_PATH`
   默认：`data/fijian/fijian.dict`
   这是 MFA 使用的发音词典。

4. `ACOUSTIC_MODEL_PATH`
   默认：`models/fijian_model.zip`
   这是可直接给 `mfa align` 使用的声学模型。

## 4. 文件格式和命名要求

### 4.1 音频和文本格式

- 音频建议为单声道 WAV，推荐 16 kHz
- 文本文件扩展名为 `.txt`
- `wav` 和 `txt` 必须同名，例如：

```text
u1_l1_001.wav
u1_l1_001.txt
```

### 4.2 参考音频和学习者音频如何配对

默认规则：按同名文件配对。

例如：

- `REFERENCE_CORPUS_DIR/u1_l1_001.wav`
- `LEARNER_CORPUS_DIR/u1_l1_001.wav`

会被当作一对进行评测。

如果两边文件名不一致，需要提供 `PAIR_CSV`。CSV 至少要有两列：

```csv
reference_id,learner_id
u1_l1_001,001_question_005
u1_l1_002,001_question_006
```

也可以额外加一列 `pair_id` 作为输出里的样本标识：

```csv
pair_id,reference_id,learner_id
lesson1_q5,u1_l1_001,001_question_005
lesson1_q6,u1_l1_002,001_question_006
```

仓库里现成示例可以直接用：

```text
scripts/mfa_pipeline/run/example_pairs.csv
```

## 5. 目录要求

脚本默认会使用这些目录：

- 参考语料：`data/fijian/raw_corpus`
- 学习者语料：`data/fijian/learner_pcm`
- 参考对齐输出：`data/fijian/aligned`
- 学习者对齐输出：`data/fijian/aligned/learner`
- 评测结果输出：`outputs/eval_runs/<RUN_TAG>`

如果你不想用这些默认路径，可以在脚本顶部修改对应的 `MANUAL_*` 变量。

## 6. 运行方式

### 6.1 最简单的方式

如果你已经准备好：

- 参考语料目录
- 学习者语料目录
- 发音词典
- 已训练好的 MFA 模型

直接运行：

```bash
bash scripts/mfa_pipeline/run/run_eval_pipeline.sh
```

### 6.2 指定新学习者目录

推荐方式：直接修改脚本顶部的

```bash
MANUAL_LEARNER_CORPUS_DIR="/path/to/new_learner"
```

也可以继续用环境变量：

```bash
LEARNER_CORPUS_DIR=/path/to/new_learner \
bash scripts/mfa_pipeline/run/run_eval_pipeline.sh
```

### 6.3 文件名不一致时指定映射表

推荐方式：直接修改脚本顶部的

```bash
MANUAL_PAIR_CSV="/mnt/d/workdir/scripts/mfa_pipeline/run/pairs.csv"
```

也可以继续用环境变量：

```bash
PAIR_CSV=/path/to/pairs.csv \
bash scripts/mfa_pipeline/run/run_eval_pipeline.sh
```

### 6.4 直接跑当前仓库里的示例

当前仓库里自带一条学习者样例，可用：

```bash
PAIR_CSV=scripts/mfa_pipeline/run/example_pairs.csv \
RUN_REFERENCE_ALIGN=0 \
RUN_LEARNER_ALIGN=0 \
bash scripts/mfa_pipeline/run/run_eval_pipeline.sh
```

如果你已经在脚本顶部写好了 `MANUAL_PAIR_CSV` 和对齐开关，直接执行脚本也可以。

### 6.5 模型不存在时自动训练

```bash
TRAIN_MODEL_IF_MISSING=1 \
RUN_REFERENCE_ALIGN=1 \
bash scripts/mfa_pipeline/run/run_eval_pipeline.sh
```

这会在 `ACOUSTIC_MODEL_PATH` 不存在时先执行：

```bash
mfa train REFERENCE_CORPUS_DIR DICTIONARY_PATH ACOUSTIC_MODEL_PATH
```

## 7. 评测新数据时通常要改哪些参数

最常改的是下面这些变量。现在更推荐直接修改脚本顶部对应的 `MANUAL_*`：

1. `MANUAL_LEARNER_CORPUS_DIR`
   指向这次要评测的学习者录音目录

2. `MANUAL_PAIR_CSV`
   只有当学习者文件名和参考文件名不一致时才需要改

3. `MANUAL_ACOUSTIC_MODEL_PATH`
   如果换了新的 MFA 模型，需要改成新的模型路径

4. `MANUAL_DICTIONARY_PATH`
   如果换了词典，需要改成新的词典路径

5. `MANUAL_RUN_REFERENCE_ALIGN`
   如果参考语料已经对齐过，通常保持 `0`
   如果你更换了参考音频、参考文本、词典或模型，需要改为 `1`

6. `MANUAL_RUN_LEARNER_ALIGN`
   新学习者数据一般都保持 `1`

7. `MANUAL_RUN_TAG`
   如果想让输出目录更容易识别，可以手工指定，例如：

```bash
MANUAL_RUN_TAG="learner_001_20260311"
```

8. `MANUAL_RUN_OUTPUT_DIR`
   如果你想把结果固定输出到某个目录，而不是按 `RUN_TAG` 自动生成，可直接指定完整目录

9. `MANUAL_SAMPLE_RATE`、`MANUAL_GFCC_NFILTS`、`MANUAL_GFCC_NCEPS`、`MANUAL_DTW_ALPHA`、`MANUAL_DTW_BETA`
   只有当你想调整特征提取或打分参数时才需要改

## 8. 输出结果说明

每次运行会在 `RUN_OUTPUT_DIR` 下生成：

1. `utterance_scores.csv`
   每条录音一行，主要看：
   - `phone_mean_score`
   - `word_mean_score`
   - `mismatched_phone_labels`
   - `missing_phone_segments`

2. `word_scores.csv`
   每个词一行，包含：
   - 参考词标签
   - 学习者词标签
   - DTW 距离
   - 词级分数

3. `phone_scores.csv`
   每个音素一行，包含：
   - 参考音素标签
   - 学习者音素标签
   - DTW 距离
   - 音素级分数

4. `run_summary.json`
   本次批量运行的总体统计

5. `segments/`
   切分后的词片段和音素片段，便于回查

## 9. 重要假设

当前实现基于以下假设：

1. 参考音频和学习者音频读的是同一条文本
2. MFA 产生的 `words` 和 `phones` tier 名称分别为 `words`、`phones`
3. 参考和学习者的分段是按顺序一一对应的

如果第 3 条不成立，输出里的这些字段会提示你：

- `mismatched_phone_labels`
- `mismatched_word_labels`
- `missing_phone_segments`
- `missing_word_segments`

## 10. 依赖

除了 Python 依赖外，还必须单独安装 MFA。

Python 侧至少需要这些库：

- `numpy`
- `pandas`
- `librosa`
- `praatio`
- `spafe`
- `fastdtw`

如果还没安装，可在项目环境中补齐。
