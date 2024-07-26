import torchaudio
import os
from loguru import logger
from strategy import AudioProcessingStrategy


class CutAudioStrategy(AudioProcessingStrategy):
    def __init__(self, root_path, timestamp, device, is_delete_last_input=True):
        self.root_path = root_path
        self.timestamp = timestamp
        self.device = device
        self.is_delete_last_input = is_delete_last_input

    def process(self, input_audio_path):
        logger.info(f"【cutAudioStrategy】开始处理，输入目录: {input_audio_path}")

        output_path = os.path.join(self.root_path, "process", "cut", self.timestamp)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.split_all_wav_files_in_directory(input_audio_path, output_path)

        logger.info(f"【cutAudioStrategy】处理结束，输出目录: {output_path}")
        if self.is_delete_last_input:
            logger.info(f"【cutAudioStrategy】删除输入目录: {input_audio_path}")
            os.remove(input_audio_path)

        return output_path

    def find_silence_intervals(
        self, waveform, sample_rate, min_silence_len=1000, silence_thresh=-40
    ):
        silence_thresh = 10 ** (silence_thresh / 20)
        hop_length = int(sample_rate * (min_silence_len / 1000))
        silences = []

        for i in range(0, waveform.shape[1] - hop_length, hop_length):
            chunk = waveform[:, i : i + hop_length]
            if chunk.abs().mean() < silence_thresh:
                silences.append((i, i + hop_length))

        silence_midpoints = [(start + end) // 2 for start, end in silences]

        return silence_midpoints

    def split_wav_on_pauses(
        self, filename, output_path, max_segment_duration=10 * 60 * 1000
    ):
        waveform, sample_rate = torchaudio.load(filename, normalize=True)
        waveform = waveform.to(self.device)
        segments = []
        start = 0

        logger.info(f"开始拆分文件: {filename}")

        while start < waveform.shape[1]:
            end = start + int(max_segment_duration * sample_rate / 1000)
            end = min(end, waveform.shape[1])  # 确保 end 不超过音频长度

            segment = waveform[:, start:end]
            logger.info(f"处理从 {start} 到 {end} 的段")

            pauses = self.find_silence_intervals(
                segment, sample_rate, min_silence_len=1000, silence_thresh=-40
            )
            valid_pauses = [
                pause for pause in pauses if pause > 0 and pause < segment.shape[1]
            ]  # 过滤有效的暂停点

            if valid_pauses:
                nearest_pause = valid_pauses[0]
                logger.info(f"在 {start + nearest_pause} 处找到暂停。将在此点拆分。")
                segments.append(waveform[:, start : start + nearest_pause])
                start += nearest_pause
            else:
                logger.info(f"在段内未找到暂停。将在 {end} 处拆分。")
                segments.append(segment)
                start = end  # 继续处理下一段

        base_filename = os.path.splitext(os.path.basename(filename))[0]
        for i, segment in enumerate(segments):
            output_filename = os.path.join(output_path, f"{base_filename}-{i}.wav")
            torchaudio.save(output_filename, segment.cpu(), sample_rate)
            logger.info(f"段 {i} 导出到 {output_filename}")

        logger.info(f"完成拆分文件: {filename}")

    def split_all_wav_files_in_directory(self, directory, output_path):
        wav_files = sorted([f for f in os.listdir(directory) if f.endswith(".wav")])

        for filename in wav_files:
            full_path = os.path.join(directory, filename)
            self.split_wav_on_pauses(full_path, output_path)


if __name__ == "__main__":
    from tool import get_project_root
    import torch

    root_path = get_project_root()
    model_weights_root = f"{root_path}/uvr5/uvr5_weights"
    timestamp = "20240726_105018"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classify_strategy = CutAudioStrategy(root_path, timestamp, device, False)
    classify_strategy.process(
        "/root/code/AudioMatic/process/VR-DeEchoAggressive_vocal/20240726_105018"
    )
