#!/bin/bash

# ===================== 配置区 =====================
# 默认学习者姓名（如果命令行不传参，会使用这个值）
DEFAULT_LEARNER_NAME="guoziyu"

# 原始音频目录（固定路径，无需频繁修改）
SOURCE_DIR="/mnt/d/workdir/data/音频采集结果"
# 目标音频目录（固定前缀，人名会自动拼接）
TARGET_BASE_DIR="/mnt/d/workdir/data/fijian/learner_pcm"
# =================================================

# 处理参数：如果命令行传了参数，就用传的；否则用默认值
LEARNER_NAME=${1:-$DEFAULT_LEARNER_NAME}

# 拼接完整的原始目录和目标目录
SOURCE_FULL_DIR="$SOURCE_DIR/${LEARNER_NAME}_recordings"
TARGET_FULL_DIR="$TARGET_BASE_DIR/${LEARNER_NAME}_recordings"

# 第一步：创建目标目录（-p 确保父目录不存在时也能创建）
echo "=== 开始创建目录：$TARGET_FULL_DIR ==="
mkdir -p "$TARGET_FULL_DIR"

# 第二步：检查原始音频目录是否存在
if [ ! -d "$SOURCE_FULL_DIR" ]; then
  echo "错误：原始音频目录 $SOURCE_FULL_DIR 不存在！"
  exit 1
fi

# 第三步：检查目录下是否有 .wav 文件
WAV_COUNT=$(ls "$SOURCE_FULL_DIR"/*.wav 2>/dev/null | wc -l)
if [ $WAV_COUNT -eq 0 ]; then
  echo "警告：$SOURCE_FULL_DIR 目录下没有找到 .wav 文件，跳过转换！"
  exit 0
fi

# 第四步：批量转换音频（单声道、16000采样率、16位PCM）
echo "=== 开始转换 $LEARNER_NAME 的音频文件 ==="
for f in "$SOURCE_FULL_DIR"/*.wav; do
  base=$(basename "$f")
  target_path="$TARGET_FULL_DIR/$base"
  # 使用ffmpeg转换，-y 覆盖已存在的文件
  ffmpeg -y -i "$f" -ac 1 -ar 16000 -c:a pcm_s16le "$target_path"
  # 转换完成提示
  echo "已转换：$base -> $target_path"
done

echo "=== 所有音频转换完成！目标目录：$TARGET_FULL_DIR ==="