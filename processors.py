"""
责任链模式：创建音频处理流程
1. 先切分音频，处理长音频素材为短音频素材
2. 提取音频中的人声
3. 去除混响
4. 获取不同说话人声与文本素材
"""

from .strategy import AudioProcessingStrategy


class AudioProcessor:
    def __init__(self, strategy: AudioProcessingStrategy):
        self._next_processor = None
        self._strategy = strategy

    def set_next(self, processor):
        self._next_processor = processor
        return processor

    def process(self, audio):
        audio = self._strategy.process(audio)
        if self._next_processor:
            self._next_processor.process(audio)
