"""
非 wav 音频转为 wav 音频策略
"""

import os
from loguru import logger
import ffmpeg
import shutil


class ConvertToWavStrategy:
    def __init__(self, root_path, timestamp):
        """
        初始化 ConvertToWavStrategy 类。

        参数:
        - root_path: 根路径
        - timestamp: 时间戳
        """
        self.root_path = root_path
        self.timestamp = timestamp

    def process(self, input_audio_path):
        """
        将非 WAV 格式的常见音频文件转换为 WAV 格式，并将所有文件放置到 output_dir。

        参数:
        - input_audio_path: 输入音频文件路径

        返回:
        - output_dir: 输出目录路径
        """
        # 创建输出目录
        output_dir = os.path.join(
            self.root_path, "process", "converted_to_wav", self.timestamp
        )
        os.makedirs(output_dir, exist_ok=True)

        # 获取输入目录中的所有文件
        all_files = [
            os.path.join(input_audio_path, name)
            for name in os.listdir(input_audio_path)
        ]

        # 常见的音频文件扩展名
        common_audio_extensions = {".mp3", ".flac", ".aac", ".m4a", ".ogg", ".wav"}

        for file_path in all_files:
            if not os.path.isfile(file_path):
                continue

            # 获取文件名和扩展名
            file_name, file_ext = os.path.splitext(os.path.basename(file_path))

            # 跳过非音频文件
            if file_ext.lower() not in common_audio_extensions:
                logger.warning(f"Skipped non-audio file: {file_path}")
                continue

            output_file_path = os.path.join(output_dir, f"{file_name}.wav")

            if file_ext.lower() != ".wav":
                # 使用 ffmpeg 将非 WAV 文件转换为 WAV 文件
                try:
                    ffmpeg.input(file_path).output(
                        output_file_path, acodec="pcm_s16le", ac=2, ar="44100"
                    ).run(overwrite_output=True)
                    logger.info(f"Converted {file_path} to {output_file_path}")
                except ffmpeg.Error as e:
                    logger.error(f"Failed to convert {file_path}: {e}")
            else:
                # 直接复制 WAV 文件到输出目录
                shutil.copy(file_path, output_file_path)
                logger.info(f"Copied {file_path} to {output_file_path}")

        # 返回输出目录路径
        return output_dir
