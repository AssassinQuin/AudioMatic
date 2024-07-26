"""
策略模式
"""

from abc import ABC, abstractmethod


class AudioProcessingStrategy(ABC):
    @abstractmethod
    def process(self, input_audio_path):
        pass
