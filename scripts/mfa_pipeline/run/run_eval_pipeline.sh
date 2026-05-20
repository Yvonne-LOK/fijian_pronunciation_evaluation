#!/usr/bin/env bash
set -euo pipefail

# Manual overrides.
# Leave empty to use the existing environment variable or the built-in default.
# If a value is set here, it takes precedence over any environment variable.
MANUAL_LEARNER_NAME=${1:-"guoziyu"}

MANUAL_PYTHON_BIN=""
MANUAL_MFA_BIN=""

# 最重要！在这里指定学习者的录音文件地址
MANUAL_REFERENCE_CORPUS_DIR="/mnt/d/workdir/data/fijian/raw_corpus"
MANUAL_LEARNER_CORPUS_DIR=""
# 重要！在这里指定z-score的calibration json文件
MANUAL_DTW_CALIBRATION_JSON="/mnt/d/workdir/outputs/calibration/dtw_score_calibration_test.json"


MANUAL_DICTIONARY_PATH="/mnt/d/workdir/data/fijian/fijian_bootstrap_arpa.dict"
MANUAL_ACOUSTIC_MODEL_PATH="english_us_arpa"

MANUAL_REFERENCE_ALIGNMENT_DIR="/mnt/d/workdir/data/fijian/aligned/reference"
MANUAL_LEARNER_ALIGNMENT_ROOT="/mnt/d/workdir/data/fijian/aligned/learner"
MANUAL_OUTPUT_ROOT="/mnt/d/workdir/outputs/eval_runs"
MANUAL_RUN_TAG=""
MANUAL_RUN_OUTPUT_DIR=""

MANUAL_PAIR_CSV="/mnt/d/workdir/scripts/mfa_pipeline/run/pairs.csv"

MANUAL_RUN_REFERENCE_ALIGN="0"
# MANUAL_RUN_REFERENCE_ALIGN="1"
MANUAL_RUN_LEARNER_ALIGN="1"
MANUAL_TRAIN_MODEL_IF_MISSING="0"
MANUAL_PREPARE_LEARNER_MFA_CORPUS_FROM_PAIRS="1"

MANUAL_SAMPLE_RATE=""
MANUAL_GFCC_NFILTS=""
MANUAL_GFCC_NCEPS=""
MANUAL_DTW_ALPHA=""
MANUAL_DTW_BETA=""
MANUAL_LLM_SENTENCE_THRESHOLD="85"
MANUAL_LLM_WORD_THRESHOLD="75"
MANUAL_LLM_PHONE_THRESHOLD="70"
MANUAL_LLM_MAX_WEAK_WORDS="2"
MANUAL_LLM_MAX_WEAK_PHONES_PER_WORD="2"

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/mnt/d/workdir"

pick_value() {
  local manual_value="$1"
  local env_value="${2:-}"
  local default_value="${3:-}"
  if [[ -n "${manual_value}" ]]; then
    printf '%s\n' "${manual_value}"
  elif [[ -n "${env_value}" ]]; then
    printf '%s\n' "${env_value}"
  else
    printf '%s\n' "${default_value}"
  fi
}

pick_manual_or_default() {
  local manual_value="$1"
  local default_value="${2:-}"
  if [[ -n "${manual_value}" ]]; then
    printf '%s\n' "${manual_value}"
  else
    printf '%s\n' "${default_value}"
  fi
}

PYTHON_BIN="$(pick_value "${MANUAL_PYTHON_BIN}" "${PYTHON_BIN:-}" "python")"
MFA_BIN="$(pick_value "${MANUAL_MFA_BIN}" "${MFA_BIN:-}" "mfa")"

LEARNER_NAME="$(pick_value "${MANUAL_LEARNER_NAME}" "${LEARNER_NAME:-}" "")"
if [[ -z "${LEARNER_NAME}" ]]; then
  echo "LEARNER_NAME is empty. Set MANUAL_LEARNER_NAME at the top of this script." >&2
  exit 1
fi

REFERENCE_CORPUS_DIR="$(pick_value "${MANUAL_REFERENCE_CORPUS_DIR}" "${REFERENCE_CORPUS_DIR:-}" "/mnt/d/workdir/data/fijian/raw_corpus")"
LEARNER_CORPUS_DIR="$(pick_value "${MANUAL_LEARNER_CORPUS_DIR}" "${LEARNER_CORPUS_DIR:-}" "/mnt/d/workdir/data/fijian/learner_pcm/${LEARNER_NAME}_recordings")"
DICTIONARY_PATH="$(pick_value "${MANUAL_DICTIONARY_PATH}" "${DICTIONARY_PATH:-}" "/mnt/d/workdir/data/fijian/fijian.dict")"
ACOUSTIC_MODEL_PATH="$(pick_value "${MANUAL_ACOUSTIC_MODEL_PATH}" "${ACOUSTIC_MODEL_PATH:-}" "english_us_arpa")"

REFERENCE_ALIGNMENT_DIR="$(pick_value "${MANUAL_REFERENCE_ALIGNMENT_DIR}" "${REFERENCE_ALIGNMENT_DIR:-}" "/mnt/d/workdir/data/fijian/aligned/reference")"
LEARNER_ALIGNMENT_ROOT="$(pick_value "${MANUAL_LEARNER_ALIGNMENT_ROOT}" "${LEARNER_ALIGNMENT_ROOT:-}" "/mnt/d/workdir/data/fijian/aligned/learner")"
LEARNER_ALIGNMENT_DIR="${LEARNER_ALIGNMENT_ROOT}/${LEARNER_NAME}"
OUTPUT_ROOT="$(pick_manual_or_default "${MANUAL_OUTPUT_ROOT}" "${PROJECT_ROOT}/outputs/eval_runs")"
RUN_TAG="$(pick_manual_or_default "${MANUAL_RUN_TAG}" "learner_${LEARNER_NAME}_$(date +%Y%m%d_%H%M%S)")"
RUN_OUTPUT_DIR="$(pick_manual_or_default "${MANUAL_RUN_OUTPUT_DIR}" "${OUTPUT_ROOT}/${RUN_TAG}")"

PAIR_CSV="$(pick_value "${MANUAL_PAIR_CSV}" "${PAIR_CSV:-}" "${PROJECT_ROOT}/scripts/mfa_pipeline/run/pairs.csv")"

RUN_REFERENCE_ALIGN="$(pick_value "${MANUAL_RUN_REFERENCE_ALIGN}" "${RUN_REFERENCE_ALIGN:-}" "0")"
RUN_LEARNER_ALIGN="$(pick_value "${MANUAL_RUN_LEARNER_ALIGN}" "${RUN_LEARNER_ALIGN:-}" "1")"
TRAIN_MODEL_IF_MISSING="$(pick_value "${MANUAL_TRAIN_MODEL_IF_MISSING}" "${TRAIN_MODEL_IF_MISSING:-}" "0")"
PREPARE_LEARNER_MFA_CORPUS_FROM_PAIRS="$(pick_value "${MANUAL_PREPARE_LEARNER_MFA_CORPUS_FROM_PAIRS}" "${PREPARE_LEARNER_MFA_CORPUS_FROM_PAIRS:-}" "1")"

SAMPLE_RATE="$(pick_value "${MANUAL_SAMPLE_RATE}" "${SAMPLE_RATE:-}" "16000")"
GFCC_NFILTS="$(pick_value "${MANUAL_GFCC_NFILTS}" "${GFCC_NFILTS:-}" "40")"
GFCC_NCEPS="$(pick_value "${MANUAL_GFCC_NCEPS}" "${GFCC_NCEPS:-}" "13")"
DTW_ALPHA="$(pick_value "${MANUAL_DTW_ALPHA}" "${DTW_ALPHA:-}" "8.0")"
DTW_BETA="$(pick_value "${MANUAL_DTW_BETA}" "${DTW_BETA:-}" "0.6")"
DTW_CALIBRATION_JSON="$(pick_value "${MANUAL_DTW_CALIBRATION_JSON}" "${DTW_CALIBRATION_JSON:-}" "")"
LLM_SENTENCE_THRESHOLD="$(pick_value "${MANUAL_LLM_SENTENCE_THRESHOLD}" "${LLM_SENTENCE_THRESHOLD:-}" "85")"
LLM_WORD_THRESHOLD="$(pick_value "${MANUAL_LLM_WORD_THRESHOLD}" "${LLM_WORD_THRESHOLD:-}" "75")"
LLM_PHONE_THRESHOLD="$(pick_value "${MANUAL_LLM_PHONE_THRESHOLD}" "${LLM_PHONE_THRESHOLD:-}" "70")"
LLM_MAX_WEAK_WORDS="$(pick_value "${MANUAL_LLM_MAX_WEAK_WORDS}" "${LLM_MAX_WEAK_WORDS:-}" "2")"
LLM_MAX_WEAK_PHONES_PER_WORD="$(pick_value "${MANUAL_LLM_MAX_WEAK_PHONES_PER_WORD}" "${LLM_MAX_WEAK_PHONES_PER_WORD:-}" "2")"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/mfa_pipeline/run_eval_pipeline.sh [learner_name]

Environment variables you may override:
  Manual overrides at the top of this file take precedence over these values.
  REFERENCE_CORPUS_DIR      Native/reference wav+txt corpus directory
  LEARNER_CORPUS_DIR        Learner wav+txt corpus directory
  DICTIONARY_PATH           MFA dictionary path
  ACOUSTIC_MODEL_PATH       MFA acoustic model path
  REFERENCE_ALIGNMENT_DIR   Output/reference TextGrid directory
  LEARNER_ALIGNMENT_ROOT    Root directory for learner TextGrid outputs
  LEARNER_NAME              Learner subdirectory name under LEARNER_ALIGNMENT_ROOT
  OUTPUT_ROOT               Root directory for evaluation results
  RUN_TAG                   Run name, used under OUTPUT_ROOT
  PAIR_CSV                  Optional CSV with reference_id, learner_id, pair_id
  RUN_REFERENCE_ALIGN       1 to align the reference corpus in this run
  RUN_LEARNER_ALIGN         1 to align the learner corpus in this run
  TRAIN_MODEL_IF_MISSING    1 to train MFA model if ACOUSTIC_MODEL_PATH is missing
  PREPARE_LEARNER_MFA_CORPUS_FROM_PAIRS 1 to build an MFA-ready learner corpus from PAIR_CSV
  SAMPLE_RATE               Audio sample rate for scoring
  GFCC_NFILTS               GFCC filterbank count
  GFCC_NCEPS                GFCC cepstral coefficient count
  DTW_ALPHA                 DTW score mapping alpha
  DTW_BETA                  DTW score mapping beta
  DTW_CALIBRATION_JSON      Optional calibration JSON for sentence/word/phone DTW score mapping
  LLM_SENTENCE_THRESHOLD    Sentence score threshold for drill-down diagnosis
  LLM_WORD_THRESHOLD        Word score threshold for weak word selection
  LLM_PHONE_THRESHOLD       Phone score threshold for weak phone selection
  LLM_MAX_WEAK_WORDS        Maximum weak words per utterance for LLM input
  LLM_MAX_WEAK_PHONES_PER_WORD Maximum weak phones per weak word for LLM input
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing command: ${cmd}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local name="$2"
  if [[ ! -d "${path}" ]]; then
    echo "${name} does not exist: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local name="$2"
  if [[ ! -f "${path}" ]]; then
    echo "${name} does not exist: ${path}" >&2
    exit 1
  fi
}

require_textgrids() {
  local dir_path="$1"
  local name="$2"
  if [[ ! -d "${dir_path}" ]]; then
    echo "${name} does not exist: ${dir_path}" >&2
    exit 1
  fi
  if ! find "${dir_path}" -maxdepth 1 -name "*.TextGrid" | grep -q .; then
    echo "${name} has no TextGrid files: ${dir_path}" >&2
    exit 1
  fi
}

align_corpus() {
  local corpus_dir="$1"
  local output_dir="$2"
  mkdir -p "${output_dir}"
  "${MFA_BIN}" align "${corpus_dir}" "${DICTIONARY_PATH}" "${ACOUSTIC_MODEL_PATH}" "${output_dir}" --clean
}
# align_corpus() {
#   local corpus_dir="$1"
#   local output_dir="$2"
#   mkdir -p "${output_dir}"

#   echo "[INFO] validating corpus: ${corpus_dir}"
#   "${MFA_BIN}" validate "${corpus_dir}" "${DICTIONARY_PATH}" "${ACOUSTIC_MODEL_PATH}" --clean

#   echo "[INFO] aligning corpus: ${corpus_dir}"
#   "${MFA_BIN}" align "${corpus_dir}" "${DICTIONARY_PATH}" "${ACOUSTIC_MODEL_PATH}" "${output_dir}" --clean
# }

train_model() {
  local model_dir
  model_dir="$(dirname "${ACOUSTIC_MODEL_PATH}")"
  mkdir -p "${model_dir}"
  "${MFA_BIN}" train "${REFERENCE_CORPUS_DIR}" "${DICTIONARY_PATH}" "${ACOUSTIC_MODEL_PATH}"
}

prepare_learner_mfa_corpus_from_pairs() {
  local pair_csv="$1"
  local learner_audio_dir="$2"
  local reference_corpus_dir="$3"
  local prepared_dir="$4"

  mkdir -p "${prepared_dir}"

  "${PYTHON_BIN}" -c "import csv, shutil, sys; from pathlib import Path
pair_csv = Path(sys.argv[1])
learner_audio_dir = Path(sys.argv[2])
reference_corpus_dir = Path(sys.argv[3])
prepared_dir = Path(sys.argv[4])
required = {'reference_id', 'learner_id'}
with pair_csv.open('r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise SystemExit(f'{pair_csv} must contain columns: {sorted(required)}')
    for row in reader:
        reference_id = row['reference_id'].strip()
        learner_id = row['learner_id'].strip()
        if not reference_id or not learner_id:
            raise SystemExit(f'Invalid row in {pair_csv}: {row}')
        src_wav = learner_audio_dir / f'{learner_id}.wav'
        src_txt = reference_corpus_dir / f'{reference_id}.txt'
        dst_wav = prepared_dir / f'{learner_id}.wav'
        dst_txt = prepared_dir / f'{learner_id}.txt'
        if not src_wav.exists():
            raise SystemExit(f'Missing learner wav for MFA prep: {src_wav}')
        if not src_txt.exists():
            raise SystemExit(f'Missing reference txt for MFA prep: {src_txt}')
        dst_wav.parent.mkdir(parents=True, exist_ok=True)
        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_wav, dst_wav)
        shutil.copy2(src_txt, dst_txt)
" "${pair_csv}" "${learner_audio_dir}" "${reference_corpus_dir}" "${prepared_dir}"
}

filter_pair_csv_for_learner() {
  local source_csv="$1"
  local learner_name="$2"
  local output_csv="$3"

  mkdir -p "$(dirname "${output_csv}")"

  "${PYTHON_BIN}" -c "import csv, sys; from pathlib import Path
source_csv = Path(sys.argv[1])
learner_name = sys.argv[2].strip()
output_csv = Path(sys.argv[3])
with source_csv.open('r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
        raise SystemExit(f'Empty CSV: {source_csv}')
    fieldnames = reader.fieldnames
    required = {'reference_id', 'learner_id'}
    if not required.issubset(fieldnames):
        raise SystemExit(f'{source_csv} must contain columns: {sorted(required)}')

    rows = []
    for row in reader:
        out_row = dict(row)
        learner_id = out_row.get('learner_id', '').strip()
        if not learner_id:
            continue

        # Template mode: replace the placeholder 'name' with the actual learner name.
        if 'name' in learner_id:
            out_row['learner_id'] = learner_id.replace('name', learner_name)
            if 'learner_name' in fieldnames:
                out_row['learner_name'] = learner_name
            rows.append({key: out_row.get(key, '') for key in fieldnames})
            continue

        # Compatibility mode for explicit per-learner rows.
        row_learner_name = out_row.get('learner_name', '').strip()
        matched = False
        if 'learner_name' in fieldnames and row_learner_name:
            matched = row_learner_name == learner_name
        else:
            matched = learner_id.startswith(f'{learner_name}_')

        if matched:
            rows.append({key: out_row.get(key, '') for key in fieldnames})

if not rows:
    raise SystemExit(
        f'No pair rows matched learner={learner_name!r} in {source_csv}. '
        'Expected either learner_id template rows like name_question_001 '
        'or explicit rows like xuhaiyue_question_001.'
    )

with output_csv.open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
" "${source_csv}" "${learner_name}" "${output_csv}"
}

main() {
  require_command "${PYTHON_BIN}"
  require_command "${MFA_BIN}"

  require_dir "${REFERENCE_CORPUS_DIR}" "REFERENCE_CORPUS_DIR"
  require_dir "${LEARNER_CORPUS_DIR}" "LEARNER_CORPUS_DIR"
  require_file "${DICTIONARY_PATH}" "DICTIONARY_PATH"
  if [[ -n "${DTW_CALIBRATION_JSON}" ]]; then
    require_file "${DTW_CALIBRATION_JSON}" "DTW_CALIBRATION_JSON"
  fi

  echo "[INFO] learner name: ${LEARNER_NAME}"
  echo "[INFO] learner corpus dir: ${LEARNER_CORPUS_DIR}"
  echo "[INFO] learner alignment output dir: ${LEARNER_ALIGNMENT_DIR}"
  echo "[INFO] run tag: ${RUN_TAG}"
  if [[ -n "${DTW_CALIBRATION_JSON}" ]]; then
    echo "[INFO] dtw calibration json: ${DTW_CALIBRATION_JSON}"
  else
    echo "[INFO] dtw calibration json: <disabled>"
  fi

  # if [[ ! -f "${ACOUSTIC_MODEL_PATH}" ]]; then
  #   if [[ "${TRAIN_MODEL_IF_MISSING}" == "1" ]]; then
  #     echo "[INFO] acoustic model missing, training MFA model: ${ACOUSTIC_MODEL_PATH}"
  #     train_model
  #   else
  #     echo "ACOUSTIC_MODEL_PATH does not exist: ${ACOUSTIC_MODEL_PATH}" >&2
  #     echo "Set TRAIN_MODEL_IF_MISSING=1 if you want this script to run mfa train first." >&2
  #     exit 1
  #   fi
  # fi

  mkdir -p "${REFERENCE_ALIGNMENT_DIR}" "${LEARNER_ALIGNMENT_DIR}" "${RUN_OUTPUT_DIR}"

  if [[ "${RUN_REFERENCE_ALIGN}" == "1" ]]; then
    echo "[INFO] aligning reference corpus with MFA"
    align_corpus "${REFERENCE_CORPUS_DIR}" "${REFERENCE_ALIGNMENT_DIR}"
  else
    require_textgrids "${REFERENCE_ALIGNMENT_DIR}" "REFERENCE_ALIGNMENT_DIR"
    echo "[INFO] skip reference alignment, using existing TextGrids in ${REFERENCE_ALIGNMENT_DIR}"
  fi

  if [[ -n "${PAIR_CSV}" ]]; then
    require_file "${PAIR_CSV}" "PAIR_CSV"
    FILTERED_PAIR_CSV="${RUN_OUTPUT_DIR}/pairing/filtered_pairs.csv"
    echo "[INFO] filtering pair csv for learner ${LEARNER_NAME}"
    filter_pair_csv_for_learner "${PAIR_CSV}" "${LEARNER_NAME}" "${FILTERED_PAIR_CSV}"
    PAIR_CSV_ARG=(--pair-csv "${FILTERED_PAIR_CSV}")
  else
    PAIR_CSV_ARG=()
  fi

  LEARNER_ALIGN_INPUT_DIR="${LEARNER_CORPUS_DIR}"
  if [[ ${#PAIR_CSV_ARG[@]} -gt 0 && "${PREPARE_LEARNER_MFA_CORPUS_FROM_PAIRS}" == "1" ]]; then
    LEARNER_ALIGN_INPUT_DIR="${RUN_OUTPUT_DIR}/mfa_input/learner"
    echo "[INFO] preparing MFA-ready learner corpus from ${FILTERED_PAIR_CSV}"
    prepare_learner_mfa_corpus_from_pairs \
      "${FILTERED_PAIR_CSV}" \
      "${LEARNER_CORPUS_DIR}" \
      "${REFERENCE_CORPUS_DIR}" \
      "${LEARNER_ALIGN_INPUT_DIR}"
  fi

  if [[ "${RUN_LEARNER_ALIGN}" == "1" ]]; then
    echo "[INFO] aligning learner corpus with MFA"
    align_corpus "${LEARNER_ALIGN_INPUT_DIR}" "${LEARNER_ALIGNMENT_DIR}"
  else
    require_textgrids "${LEARNER_ALIGNMENT_DIR}" "LEARNER_ALIGNMENT_DIR"
    echo "[INFO] skip learner alignment, using existing TextGrids in ${LEARNER_ALIGNMENT_DIR}"
  fi

  echo "[INFO] running batch GFCC + DTW evaluation"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/mfa_pipeline/run/batch_gfcc_dtw_eval.py" \
    --reference-dir "${REFERENCE_CORPUS_DIR}" \
    --learner-dir "${LEARNER_CORPUS_DIR}" \
    --reference-aligned-dir "${REFERENCE_ALIGNMENT_DIR}" \
    --learner-aligned-dir "${LEARNER_ALIGNMENT_DIR}" \
    --output-dir "${RUN_OUTPUT_DIR}" \
    --sample-rate "${SAMPLE_RATE}" \
    --nfilts "${GFCC_NFILTS}" \
    --nceps "${GFCC_NCEPS}" \
    --alpha "${DTW_ALPHA}" \
    --beta "${DTW_BETA}" \
    "${PAIR_CSV_ARG[@]}"

  echo "[INFO] building structured LLM diagnosis input"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_llm_diagnosis_input.py" \
    --utterance-scores "${RUN_OUTPUT_DIR}/utterance_scores.csv" \
    --word-scores "${RUN_OUTPUT_DIR}/word_scores.csv" \
    --phone-scores "${RUN_OUTPUT_DIR}/phone_scores.csv" \
    --reference-dir "${REFERENCE_CORPUS_DIR}" \
    --output-jsonl "${RUN_OUTPUT_DIR}/llm_diagnosis_input/llm_diagnosis_input.jsonl" \
    --sentence-threshold "${LLM_SENTENCE_THRESHOLD}" \
    --word-threshold "${LLM_WORD_THRESHOLD}" \
    --phone-threshold "${LLM_PHONE_THRESHOLD}" \
    --max-words "${LLM_MAX_WEAK_WORDS}" \
    --max-phones-per-word "${LLM_MAX_WEAK_PHONES_PER_WORD}"

  echo "[INFO] building DTW distance-only jsonl"
  DTW_CALIBRATION_ARG=()
  if [[ -n "${DTW_CALIBRATION_JSON}" ]]; then
    DTW_CALIBRATION_ARG=(--calibration-json "${DTW_CALIBRATION_JSON}")
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_dtw_distance_jsonl.py" \
    --reference-dir "${REFERENCE_CORPUS_DIR}" \
    --utterance-scores "${RUN_OUTPUT_DIR}/utterance_scores.csv" \
    --word-scores "${RUN_OUTPUT_DIR}/word_scores.csv" \
    --phone-scores "${RUN_OUTPUT_DIR}/phone_scores.csv" \
    --output-jsonl "${RUN_OUTPUT_DIR}/dtw_distances/dtw_distances.jsonl" \
    "${DTW_CALIBRATION_ARG[@]}"

  echo "[DONE] results written to ${RUN_OUTPUT_DIR}"
  echo "[DONE] utterance scores: ${RUN_OUTPUT_DIR}/utterance_scores.csv"
  echo "[DONE] word scores: ${RUN_OUTPUT_DIR}/word_scores.csv"
  echo "[DONE] phone scores: ${RUN_OUTPUT_DIR}/phone_scores.csv"
  echo "[DONE] LLM diagnosis input dir: ${RUN_OUTPUT_DIR}/llm_diagnosis_input"
  echo "[DONE] LLM diagnosis jsonl: ${RUN_OUTPUT_DIR}/llm_diagnosis_input/llm_diagnosis_input.jsonl"
  echo "[DONE] DTW distance dir: ${RUN_OUTPUT_DIR}/dtw_distances"
  echo "[DONE] DTW distance jsonl: ${RUN_OUTPUT_DIR}/dtw_distances/dtw_distances.jsonl"
}

main "$@"
