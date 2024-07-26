"""
提取人声策略
1. 使用 HP2_all_vocals 模型
# 2. 使用 onnx_dereverb_By_FoxJoy 模型(太慢，放弃)
3. 使用 VR-DeEchoAggressive 模型

使用 uvr5: https://github.com/AIFSH/ComfyUI-UVR5
"""

import os
from uvr5.uvr_audio_process import uvr
from strategy import AudioProcessingStrategy
from loguru import logger


class ExtractVocalStrategy(AudioProcessingStrategy):
    def __init__(
        self,
        root_path,
        timestamp,
        device,
        model_weights_root="./uvr5/uvr5_weights",
        is_delete_last_input=True,
    ):
        """
        初始化提取人声策略类。

        参数:
        root_path: 根目录路径
        model_weights_root: 模型权重文件根目录
        timestamp: 时间戳
        device: 设备类型（cuda 或 cpu）
        """
        self.model_weights_root = model_weights_root
        self.device = device
        self.root_path = root_path
        self.timestamp = timestamp
        self.tmp_path = os.path.join(root_path, "process", "temp_gen")
        self.is_delete_last_input = is_delete_last_input
        self.model = None

        os.makedirs(self.tmp_path, exist_ok=True)

    def process(self, input_audio_path):
        """
        处理音频文件，依次使用 HP2_all_vocals、onnx_dereverb_By_FoxJoy 和 VR-DeEchoAggressive 模型提取人声。

        参数:
        input_audio_path: 输入音频文件路径

        返回:
        提取人声后的最终输出目录路径
        """
        logger.info(f"开始提取人声，输入目录:{input_audio_path}")
        self.model = uvr(self.model_weights_root, self.tmp_path, self.device)
        output_path = self.handle_HP2_all_vocals(input_audio_path)
        # output_path = self.handle_onnx_dereverb_By_FoxJoy(output_path)
        output_path = self.handle_VR_DeEchoAggressive(output_path)
        logger.info(f"提取人声完成，输出目录:{output_path}")
        return output_path

    def handle_HP2_all_vocals(self, input_audio_path):
        """
        使用 HP2_all_vocals 模型提取人声。

        参数:
        input_audio_path: 输入音频文件路径

        返回:
        提取人声后的输出目录路径
        """
        logger.info(f"【HP2_all_vocals】开始提取人声，输入目录:{input_audio_path}")
        vocal_output_path = os.path.join(
            self.root_path, "process", "hp2_all_vocals_vocal", self.timestamp
        )
        instrumental_output_path = os.path.join(
            self.root_path, "process", "hp2_all_vocals_instrumental", self.timestamp
        )
        self.model.process(
            "HP2_all_vocals",
            input_audio_path,
            vocal_output_path,
            instrumental_output_path,
        )
        logger.info(f"【HP2_all_vocals】提取人声完成，输出目录:{vocal_output_path}")
        if self.is_delete_last_input:
            logger.info(f"【HP2_all_vocals】删除输入目录:{input_audio_path}")
            os.remove(input_audio_path)
        return vocal_output_path

    def handle_onnx_dereverb_By_FoxJoy(self, input_audio_path):
        """
        使用 onnx_dereverb_By_FoxJoy 模型提取人声。

        参数:
        input_audio_path: 输入音频文件路径

        返回:
        提取人声后的输出目录路径
        """
        logger.info(
            f"【onnx_dereverb_By_FoxJoy】开始提取人声，输入目录:{input_audio_path}"
        )
        vocal_output_path = os.path.join(
            self.root_path, "process", "onnx_dereverb_By_FoxJoy_vocal", self.timestamp
        )
        instrumental_output_path = os.path.join(
            self.root_path,
            "process",
            "onnx_dereverb_By_FoxJoy_instrumental",
            self.timestamp,
        )
        self.model.process(
            "onnx_dereverb_By_FoxJoy",
            input_audio_path,
            vocal_output_path,
            instrumental_output_path,
        )
        logger.info(
            f"【onnx_dereverb_By_FoxJoy】提取人声完成，输出目录:{vocal_output_path}"
        )
        if self.is_delete_last_input:
            logger.info(f"【onnx_dereverb_By_FoxJoy】删除输入目录:{input_audio_path}")
            os.remove(input_audio_path)
        return vocal_output_path

    def handle_VR_DeEchoAggressive(self, input_audio_path):
        """
        使用 VR-DeEchoAggressive 模型提取人声。

        参数:
        input_audio_path: 输入音频文件路径

        返回:
        提取人声后的输出目录路径
        """
        logger.info(f"【VR-DeEchoAggressive】开始提取人声，输入目录:{input_audio_path}")
        vocal_output_path = os.path.join(
            self.root_path, "process", "VR-DeEchoAggressive_vocal", self.timestamp
        )
        instrumental_output_path = os.path.join(
            self.root_path,
            "process",
            "VR-DeEchoAggressive_instrumental",
            self.timestamp,
        )
        self.model.process(
            "VR-DeEchoAggressive",
            input_audio_path,
            vocal_output_path,
            instrumental_output_path,
        )
        logger.info(
            f"【VR-DeEchoAggressive】提取人声完成，输出目录:{vocal_output_path}"
        )
        if self.is_delete_last_input:
            logger.info(f"【VR-DeEchoAggressive】删除输入目录:{input_audio_path}")
            os.remove(input_audio_path)
        return vocal_output_path


if __name__ == "__main__":
    from tool import get_project_root
    import torch

    root_path = get_project_root()  # 获取项目根目录
    model_weights_root = f"{root_path}/uvr5/uvr5_weights"  # 模型权重根路径
    timestamp = "20240726_105018"
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备（GPU或CPU）
    classify_strategy = ExtractVocalStrategy(
        root_path, timestamp, device, model_weights_root, False
    )
    classify_strategy.process(
        "/root/code/AudioMatic/process/VR-DeEchoAggressive_vocal/20240726_105018"
    )
