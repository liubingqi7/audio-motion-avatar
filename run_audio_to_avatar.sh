#!/bin/bash

# Audio to Avatar Pipeline
# 使用方法: ./run_audio_to_avatar.sh <音频文件> <图像文件> [输出视频路径]

if [ $# -lt 2 ]; then
    echo "使用方法: $0 <音频文件> <图像文件> [输出视频路径]"
    echo "示例: $0 ./audio.wav ./person.jpg ./output.mp4"
    exit 1
fi

AUDIO_FILE="$1"
IMAGE_FILE="$2"
VIDEO_OUTPUT="${3:-./output_video.mp4}"

# 检查输入文件是否存在
if [ ! -f "$AUDIO_FILE" ]; then
    echo "错误: 音频文件不存在: $AUDIO_FILE"
    exit 1
fi

if [ ! -f "$IMAGE_FILE" ]; then
    echo "错误: 图像文件不存在: $IMAGE_FILE"
    exit 1
fi

echo "=== Audio to Avatar Pipeline ==="
echo "输入音频: $AUDIO_FILE"
echo "输入图像: $IMAGE_FILE"
echo "输出视频: $VIDEO_OUTPUT"
echo ""

# 运行pipeline
python scripts/audio_to_avatar.py \
    --audio "$AUDIO_FILE" \
    --image "$IMAGE_FILE" \
    --video_output "$VIDEO_OUTPUT" \
    --cleanup

echo ""
echo "Pipeline完成！"
echo "输出视频: $VIDEO_OUTPUT" 