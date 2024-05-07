from .base import BaseAugmentor
from .utils import librosa_to_pydub
import random
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import wave
import logging

logger = logging.getLogger(__name__)
class GaussianAugmentor(BaseAugmentor):
    """
        Gaussian augmentation
        Config:
        mean: int, gaussian mean factor
        std_dev: int, gaussian standard deviation factor
    """
    def __init__(self, config: dict):
        """
        This method initialize the `GaussianAugmentor` object.
        
        :param config: dict, configuration dictionary
        """
        
        super().__init__(config)
        self.min_amplitude = config['min_amplitude']
        self.max_amplitude = config['max_amplitude']
        assert self.min_amplitude > 0.0
        assert self.max_amplitude > 0.0
        assert self.max_amplitude >= self.min_amplitude
        self.amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
        
    
    def transform(self):
        """
        Transform the audio by adding gausian noise.
        """
        noise = np.random.randn(self.data.shape).astype(np.float32)
        data = self.data + self.amplitude * noise
        # transform to pydub audio segment
        self.augmented_audio = librosa_to_pydub(data, sr=self.sr)
