#!/bin/bash

# 下载 mp4 视频脚本

# 检查并创建 tmp_data 目录
if [ ! -d "./tmp_data" ]; then
    mkdir -p ./tmp_data
fi

# 下载和转换视频
process_video() {
    local url=$1
    local filename=$2
    wget -O "./tmp_data/${filename}.mp4" "${url}"
    ffmpeg -i "./tmp_data/${filename}.mp4" -vn -acodec pcm_s16le -ar 44100 -ac 2 "./tmp_data/${filename}.wav"
    rm -rf "./tmp_data/${filename}.mp4"
}

# 视频链接和文件名数组
urls=(
    "download_url1"
)

filenames=(
    "1"
)

# 获取数组长度
length=${#urls[@]}

# 并行处理所有视频
for (( i=0; i<${length}; i++ )); do
    process_video "${urls[$i]}" "${filenames[$i]}" &
done

# 等待所有后台任务完成
wait

echo "所有视频已处理完成"
