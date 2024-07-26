from datetime import datetime
from strategy_cut import CutAudioStrategy
from strategy_extracte import ExtractVocalStrategy
from strategy_conver2wav import ConvertToWavStrategy
from processors import AudioProcessor
from strategy_classify import ClassifyAudioStrategy
from strategy_merge import MergeAudioStrategy
from tool import get_project_root

from loguru import logger
import argparse


def initialize_strategies(root_path, model_weights_root, timestamp, device):
    """
    流程：
    1. 将音频文件转换为wav格式
    2. 切割音频文件，把长文件切割为较短文件，若显存不足/较多 可以酌情切割
    3. 提取人声
    4. 分类说话人
    """
    # 将音频文件转换为wav格式策略
    convert_to_wav_strategy = ConvertToWavStrategy(root_path, timestamp)
    # 合并wav策略
    merge_audio_strategy = MergeAudioStrategy(root_path, timestamp, device)
    # 切割音频文件策略（参数自行调整）
    cut_strategy = CutAudioStrategy(root_path, timestamp, device)
    # 提取人声策略
    extract_vocal_strategy = ExtractVocalStrategy(
        root_path, model_weights_root, timestamp, device
    )
    # 分类说话人策略
    classify_strategy = ClassifyAudioStrategy(root_path, timestamp, device)
    # 返回顺序是执行顺序
    return [
        convert_to_wav_strategy,
        cut_strategy,
        extract_vocal_strategy,
        merge_audio_strategy,
        classify_strategy,
    ]


def setup_processor_chain(strategies):
    """设置责任链"""
    # 创建第一个处理器
    processor = AudioProcessor(strategies[0])
    # 当前处理器
    current_processor = processor
    # 遍历剩余的策略
    for strategy in strategies[1:]:
        next_processor = AudioProcessor(strategy)
        current_processor.set_next(next_processor)
        current_processor = next_processor
    return processor


def process_audio(input_audio_path, processor):
    """处理音频文件"""
    processor.process(input_audio_path)


def main(input_audio_path):
    import torch

    # 获取项目根目录
    root_path = get_project_root()
    # 模型权重根路径
    model_weights_root = f"{root_path}/uvr5/uvr5_weights"
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 选择设备（GPU或CPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=============================")
    logger.info(f"使用根路径: {root_path}")
    logger.info(f"使用模型权重根路径: {model_weights_root}")
    logger.info(f"使用时间戳: {timestamp}")
    logger.info(f"使用设备: {device}")
    logger.info(f"使用输入音频路径: {input_audio_path}")
    logger.info("=============================")

    # 初始化策略
    strategies = initialize_strategies(root_path, model_weights_root, timestamp, device)
    # 设置处理器链
    processor = setup_processor_chain(strategies)
    # 处理音频文件
    process_audio(input_audio_path, processor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理音频文件。")
    parser.add_argument(
        "input_audio_path",
        type=str,
        help="输入音频目录的路径。",
        nargs="?",
        default="/root/code/AudioMatic/tmp_data/converted_wav_22050",
    )
    args = parser.parse_args()
    main(args.input_audio_path)
