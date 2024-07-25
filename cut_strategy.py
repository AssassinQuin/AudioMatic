"""
切割策略
"""

import torchaudio
import os
from loguru import logger

from .strategy import AudioProcessingStrategy


class CutAudioStrategy(AudioProcessingStrategy):
    def __init__(self, root_path, timestamp, device):
        """
        初始化切割策略类。

        参数:
        root_path: 根目录路径
        timestamp: 时间戳
        device: 设备类型（cuda 或 cpu）
        """
        self.root_path = root_path
        self.timestamp = timestamp
        self.device = device

    def process(self, input_audio_path):
        """
        处理音频文件并创建输出目录。

        参数:
        input_audio_path: 输入音频文件的路径。

        返回:
        输出目录路径。
        """
        logger.info(f"【cutAudioStrategy】开始处理，输入目录: {input_audio_path}")

        output_path = os.path.join(self.root_path, "process", "cut", self.timestamp)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 处理目录中的所有 WAV 文件
        self.split_all_wav_files_in_directory(input_audio_path, output_path)

        logger.info(f"【cutAudioStrategy】处理结束，输出目录: {output_path}")

        return output_path

    def find_silence_intervals(
        self, waveform, sample_rate, min_silence_len=1000, silence_thresh=-40
    ):
        """
        查找音频样本中的静音间隔。

        参数:
        waveform: 音频样本。
        sample_rate: 音频帧率。
        min_silence_len: 最小静音长度，以毫秒为单位。
        silence_thresh: 静音阈值，以分贝为单位。

        返回:
        静音间隔列表。
        """
        silence_thresh = 10 ** (silence_thresh / 20)  # 将分贝转换为幅度值
        hop_length = int(sample_rate * (min_silence_len / 1000))
        silences = []

        for i in range(0, waveform.shape[1] - hop_length, hop_length):
            chunk = waveform[:, i : i + hop_length]
            if chunk.abs().mean() < silence_thresh:
                silences.append((i, i + hop_length))

        return silences

    def split_wav_on_pauses(
        self, filename, output_path, max_segment_duration=10 * 60 * 1000
    ):
        """
        根据暂停和最大段长度将WAV文件拆分为多个段。

        参数:
        filename: WAV文件的路径。
        output_path: 输出路径。
        max_segment_duration: 段的最大持续时间（以毫秒为单位）。
        """
        waveform, sample_rate = torchaudio.load(filename, normalize=True)
        waveform = waveform.to(self.device)
        segments = []
        start = 0

        logger.info(f"开始拆分文件: {filename}")

        while start < waveform.shape[1]:
            end = start + int(max_segment_duration * sample_rate / 1000)
            if end >= waveform.shape[1]:
                logger.info(f"到达音频末尾。从 {start} 到结尾添加最后一段。")
                segments.append(waveform[:, start:])
                break

            segment = waveform[:, start:end]
            logger.info(f"处理从 {start} 到 {end} 的段")

            pauses = self.find_silence_intervals(
                segment, sample_rate, min_silence_len=1000, silence_thresh=-40
            )
            if pauses:
                nearest_pause = pauses[-1][1]
                logger.info(f"在 {nearest_pause} 处找到暂停。将在此点拆分。")
                segments.append(waveform[:, start : start + nearest_pause])
                start += nearest_pause
            else:
                logger.info(f"在段内未找到暂停。将在 {end} 处拆分。")
                segments.append(segment)
                start += int(max_segment_duration * sample_rate / 1000)

        base_filename = os.path.splitext(os.path.basename(filename))[0]
        for i, segment in enumerate(segments):
            output_filename = os.path.join(output_path, f"{base_filename}-{i}.wav")
            torchaudio.save(output_filename, segment.cpu(), sample_rate)
            logger.info(f"段 {i} 导出到 {output_filename}")

        logger.info(f"完成拆分文件: {filename}")

    def split_all_wav_files_in_directory(self, directory, output_path):
        """
        处理目录中的所有 WAV 文件。

        参数:
        directory: 包含 WAV 文件的目录。
        output_path: 输出路径。
        """
        # 获取目录中所有以 .wav 结尾的文件名，并按升序排序
        wav_files = sorted([f for f in os.listdir(directory) if f.endswith(".wav")])

        for filename in wav_files:
            full_path = os.path.join(directory, filename)
            self.split_wav_on_pauses(full_path, output_path)
