from datetime import datetime
from cut_strategy import CutAudioStrategy
from extracte_strategy import ExtractVocalStrategy
from conver2wav_strategy import ConvertToWavStrategy
from processors import AudioProcessor
from classify_strategy import ClassifyAudioStrategy
from tool import get_project_root

import torch
from loguru import logger
import argparse


"""
流程：
1. 将音频文件转换为wav格式
2. 切割音频文件，把长文件切割为较短文件，若显存不足/较多 可以酌情切割
3. 提取人声
4. 分类说话人
"""


def initialize_strategies(root_path, model_weights_root, timestamp, device):
    """初始化策略"""
    convert_to_wav_strategy = ConvertToWavStrategy(root_path, timestamp)
    cut_strategy = CutAudioStrategy(root_path, timestamp, device)
    extract_vocal_strategy = ExtractVocalStrategy(
        root_path, model_weights_root, timestamp, device
    )
    classify_strategy = ClassifyAudioStrategy(root_path, timestamp, device)
    # 返回顺序是执行顺序
    return cut_strategy, extract_vocal_strategy, classify_strategy


def setup_processor_chain(strategies):
    """设置责任链"""
    processor = AudioProcessor(strategies[0])  # 创建第一个处理器
    current_processor = processor  # 当前处理器
    for strategy in strategies[1:]:  # 遍历剩余的策略
        next_processor = AudioProcessor(strategy)  # 创建下一个处理器
        current_processor.set_next(next_processor)  # 设置当前处理器的下一个处理器
        current_processor = next_processor  # 更新当前处理器
    return processor  # 返回第一个处理器


def process_audio(input_audio_path, processor):
    """处理音频文件"""
    processor.process(input_audio_path)  # 使用处理器链处理音频文件


def main(input_audio_path):
    root_path = get_project_root()  # 获取项目根目录
    model_weights_root = f"{root_path}/uvr5/uvr5_weights"  # 模型权重根路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备（GPU或CPU）

    logger.info("=============================")
    logger.info(f"使用根路径: {root_path}")
    logger.info(f"使用模型权重根路径: {model_weights_root}")
    logger.info(f"使用时间戳: {timestamp}")
    logger.info(f"使用设备: {device}")
    logger.info(f"使用输入音频路径: {input_audio_path}")
    logger.info("=============================")

    strategies = initialize_strategies(
        root_path, model_weights_root, timestamp, device
    )  # 初始化策略
    processor = setup_processor_chain(strategies)  # 设置处理器链
    process_audio(input_audio_path, processor)  # 处理音频文件


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理音频文件。")
    parser.add_argument(
        "input_audio_path",
        type=str,
        help="输入音频目录的路径。",
        nargs="?",
        default="/root/code/GPT-SoVITS/audio/test_1",
    )
    args = parser.parse_args()
    main(args.input_audio_path)
