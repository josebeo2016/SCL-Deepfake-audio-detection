from .base import BaseAugmentor
from .utils import librosa_to_pydub
import random
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment

import logging
logger = logging.getLogger(__name__)
class PitchAugmentor(BaseAugmentor):
    """
        Pitch augmentation
        Config:
        min_pitch_shift: int, min pitch shift factor
        max_pitch_shift: int, max pitch shift factor
    """
    def __init__(self, config: dict):
        """
        This method initialize the `PitchAugmentor` object.
        
        :param config: dict, configuration dictionary
        """
        
        super().__init__(config)
        self.min_pitch_shift = config["min_pitch_shift"]
        self.max_pitch_shift = config["max_pitch_shift"]
        self.pitch_shift = random.randint(self.min_pitch_shift, self.max_pitch_shift)
        
    
    def transform(self):
        """
        Transform the audio by pitch shifting based on `librosa.effects.pitch_shift`
        The pitch shift factor is randomly selected between min_pitch_shift and max_pitch_shift
        """
        data = librosa.effects.pitch_shift(self.data, sr=self.sr, n_steps=self.pitch_shift)
        # transform to pydub audio segment
        self.augmented_audio = librosa_to_pydub(data, sr=self.sr)
    