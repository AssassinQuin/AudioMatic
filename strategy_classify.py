"""
提取分类说话人策略
使用 FunASR:https://github.com/modelscope/FunASR/blob/main/README_zh.md

参数测试环境：4060 8G
"""

import os
import shutil
from loguru import logger

from strategy import AudioProcessingStrategy


class ClassifyAudioStrategy(AudioProcessingStrategy):
    def __init__(self, root_path, timestamp, device, is_delete_last_input=True):
        """
        初始化提取分类说话人策略类。

        参数:
        root_path: 根目录路径
        timestamp: 时间戳
        device: 设备类型（cuda 或 cpu）
        """
        self.root_path = root_path
        self.timestamp = timestamp
        self.device = device
        self.model = None
        self.is_delete_last_input = is_delete_last_input

    def process(self, input_audio_path):
        """
        处理音频文件，提取说话人并分类保存。

        参数:
        input_audio_path: 输入音频文件的路径
        """
        from funasr import AutoModel
        from pydub import AudioSegment

        logger.info(f"开始处理音频文件: {input_audio_path}")

        # 使用时在加载
        self.model = AutoModel(
            model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 60000},
            punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
            device=self.device,
        )
        wav_file_path = self.merge_wav_files(input_audio_path)
        res = self.model.generate(
            input=wav_file_path,
            batch_size_s=60,
            cache={},
            use_itn=True,
            merge_vad=True,
            merge_length_s=60,
        )

        audio = AudioSegment.from_wav(wav_file_path)

        # 创建输出目录
        output_dir = os.path.join(self.root_path, "process", "classify", self.timestamp)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("开始处理音频裁剪和文本保存。")
        current_speaker = None
        current_segment = None
        current_text = ""
        current_start_time = 0

        # 处理每个识别结果
        for item in res[0]["sentence_info"]:
            text = item["text"]
            timestamps = item["timestamp"]
            speaker = item["spk"]

            # 使用 timestamp 第一个元素的第一个时间戳作为片段开始时间
            start_time = timestamps[0][0]
            # 使用 timestamp 最后一个元素的最后一个时间戳作为片段结束时间
            end_time = timestamps[-1][-1]
            # 生成音频片段
            segment = audio[start_time:end_time]

            # 检查时间戳值
            logger.info(f"""
处理片段：start={start_time} ms, end={end_time} ms
音频片段长度：{len(segment)} ms - {len(audio)} ms
识别文本：{text}
说话人：{speaker}
""")

            # 若 speaker 是同个人，时长小于 15s，则合并多个音频片段到一个音频片段
            if current_speaker == speaker and (end_time - current_start_time) < 22000:
                if current_segment is None:
                    current_segment = segment
                else:
                    current_segment += segment
                current_text += text
            else:
                # 保存当前合并的音频片段和文本
                if current_segment is not None and len(current_segment) > 3000:
                    self.save_segment(
                        current_segment,
                        current_text,
                        current_speaker,
                        current_start_time,
                        output_dir,
                    )

                # 开始新的音频片段
                current_speaker = speaker
                current_segment = segment
                current_text = text
                current_start_time = start_time

        # 保存最后一个合并的音频片段和文本
        if current_segment is not None and len(current_segment) > 3000:
            self.save_segment(
                current_segment,
                current_text,
                current_speaker,
                current_start_time,
                output_dir,
            )

        logger.info(f"提取分类说话人完成，输出目录：{output_dir}")
        if self.is_delete_last_input:
            logger.info(f"【classifyStrategy】删除输入目录: {input_audio_path}")
            shutil.rmtree(wav_file_path)
        return output_dir

    def save_segment(self, segment, text, speaker, start_time, output_dir):
        """
        保存音频片段和对应的文本信息。

        参数:
        segment: 音频片段
        text: 文本信息
        speaker: 说话人标识
        start_time: 片段开始时间
        output_dir: 输出目录
        """
        # 创建说话人目录
        speaker_dir = os.path.join(output_dir, str(speaker))
        os.makedirs(speaker_dir, exist_ok=True)

        # 生成唯一的文件名
        name = f"{speaker}_{start_time}"
        audio_file = os.path.join(speaker_dir, f"{name}.wav")
        text_file = os.path.join(speaker_dir, f"{name}.normalized.txt")

        # 导出音频片段
        segment.export(audio_file, format="wav")

        # 保存文本信息
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    import torch
    from tool import get_project_root

    root_path = get_project_root()  # 获取项目根目录
    model_weights_root = f"{root_path}/uvr5/uvr5_weights"  # 模型权重根路径
    timestamp = "20240726_105018"
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备（GPU或CPU）
    classify_strategy = ClassifyAudioStrategy(root_path, timestamp, device, False)
    classify_strategy.process(
        "/root/code/AudioMatic/process/VR-DeEchoAggressive_vocal/20240726_105018"
    )
