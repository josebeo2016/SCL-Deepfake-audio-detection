from .base import BaseAugmentor
from .utils import librosa_to_pydub
from pydub import AudioSegment
import random

import logging
logger = logging.getLogger(__name__)
class SpeedAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Speed augmentor class requires these config:
        min_speed_factor: float, min speed factor
        max_speed_factor: float, max speed factor
        """
        super().__init__(config)
        self.speed_factor = random.uniform(config["min_speed_factor"], config["max_speed_factor"])
        
        self.audio_data = None
        
    def load(self, input_path: str):
        """
        :param input_path: path to the input audio file
        """
        # load with librosa
        super().load(input_path)
        # transform to pydub audio segment
        self.audio_data = librosa_to_pydub(self.data, sr=self.sr)
        
    def transform(self):
        """
        Speed up or down the audio using pydub `AudioSegment.speedup` method
        """
        self.augmented_audio = self.audio_data.speedup(self.speed_factor)