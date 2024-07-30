import os
import shutil
from loguru import logger
from strategy import AudioProcessingStrategy


class MergeAudioStrategy(AudioProcessingStrategy):
    def __init__(self, root_path, timestamp, device, is_delete_last_input=True):
        """
        初始化合并策略类。

        参数:
        root_path: 根目录路径
        timestamp: 时间戳
        device: 设备（如 'cpu' 或 'cuda'）
        is_delete_last_input: 是否删除输入目录
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

        # 根据前缀合并文件
        prefix_dict = {}
        for filename in wav_files:
            prefix = filename.split("-")[0]
            if prefix not in prefix_dict:
                prefix_dict[prefix] = []
            prefix_dict[prefix].append(filename)

        output_files = []
        for prefix, files in prefix_dict.items():
            merged_waveform = None
            sample_rate = None

            for filename in files:
                full_path = os.path.join(input_audio_path, filename)
                waveform, sr = torchaudio.load(full_path)
                waveform = waveform.to(self.device)  # 将音频数据加载到GPU

                # 检查合并时的显存使用情况
                if merged_waveform is None:
                    merged_waveform = waveform
                    sample_rate = sr
                else:
                    try:
                        merged_waveform = torch.cat((merged_waveform, waveform), dim=1)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            # 如果显存不足，则将部分数据移回CPU
                            logger.warning("显存不足，部分数据移回CPU进行处理")
                            merged_waveform = merged_waveform.cpu()
                            waveform = waveform.cpu()
                            merged_waveform = torch.cat(
                                (merged_waveform, waveform), dim=1
                            )
                            merged_waveform = merged_waveform.to(self.device)
                        else:
                            raise e

            # 将最终的合并结果移回CPU以进行保存
            merged_waveform = merged_waveform.cpu()

            # 保存合并后的 wav 文件
            output_filename = os.path.join(output_path, f"{prefix}.wav")
            torchaudio.save(output_filename, merged_waveform, sample_rate)
            output_files.append(output_filename)

        if self.is_delete_last_input:
            logger.info(f"【mergeStrategy】删除输入目录: {input_audio_path}")
            shutil.rmtree(input_audio_path)

        return output_path


if __name__ == "__main__":
    import torch
    from tool import get_project_root

    root_path = get_project_root()  # 获取项目根目录
    timestamp = "20240730_085330"
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备（GPU或CPU）
    merge_strategy = MergeAudioStrategy(root_path, timestamp, device, False)
    merged_audio_paths = merge_strategy.process(
        "/root/autodl-tmp/AudioMatic/process/VR-DeEchoAggressive_vocal/20240730_085330"
    )
    for path in merged_audio_paths:
        logger.info(f"合并后的音频文件路径: {path}")
