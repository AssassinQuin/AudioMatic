import os
import shutil
from uuid import uuid4
from loguru import logger


from strategy import AudioProcessingStrategy


class MergeAudioStrategy(AudioProcessingStrategy):
    def __init__(self, root_path, timestamp, device, is_delete_last_input=True):
        """
        初始化切割策略类。

        参数:
        root_path: 根目录路径
        timestamp: 时间戳
        """
        self.root_path = root_path
        self.timestamp = timestamp
        self.device = device
        self.is_delete_last_input = is_delete_last_input

    def process(self, input_audio_path):
        """
        合并目录中的所有 wav 文件为一个文件。

        参数:
        input_audio_path: 输入音频文件目录

        返回:
        合并后的 wav 文件路径
        """
        import torch
        import torchaudio

        # 创建输出目录
        output_path = os.path.join(self.root_path, "process", "merge", self.timestamp)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 获取目录中所有以 .wav 结尾的文件名，并按升序排序
        wav_files = sorted(
            [f for f in os.listdir(input_audio_path) if f.endswith(".wav")]
        )

        # 合并所有 wav 文件
        merged_waveform = None
        sample_rate = None
        for filename in wav_files:
            full_path = os.path.join(input_audio_path, filename)
            waveform, sr = torchaudio.load(full_path)
            waveform = waveform.to(self.device)  # 将音频数据加载到GPU
            if merged_waveform is None:
                merged_waveform = waveform
                sample_rate = sr
            else:
                merged_waveform = torch.cat((merged_waveform, waveform), dim=1)

        # 生成 UUID
        uuid = str(uuid4())[:4]
        output_filename = os.path.join(output_path, f"{uuid}.wav")

        # 保存合并后的 wav 文件
        torchaudio.save(
            output_filename, merged_waveform.cpu(), sample_rate
        )  # 将数据移回CPU并保存
        if self.is_delete_last_input:
            logger.info(f"【mergeStrategy】删除输入目录: {input_audio_path}")
            shutil.rmtree(input_audio_path)

        return output_filename
