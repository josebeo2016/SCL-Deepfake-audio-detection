from .base import BaseAugmentor
from .utils import recursive_list_files
from pydub import AudioSegment
import random
import numpy as np
from .utils import librosa_to_pydub

import logging
logger = logging.getLogger(__name__)
class BackgroundNoiseAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Background noise augmentation method.
        
        Config:
        noise_path: str, path to the folder containing noise files
        min_SNR_dB: int, min SNR in dB
        max_SNR_dB: int, max SNR in dB
        """
        super().__init__(config)
        self.noise_path = config["noise_path"]
        self.noise_list = self.select_noise(self.noise_path)
        self.min_SNR_dB = config["min_SNR_dB"]
        self.max_SNR_dB = config["max_SNR_dB"]
        
    def select_noise(self, noise_path: str) -> list:
        noise_list = recursive_list_files(noise_path)
        return noise_list
    
    def load(self, input_path: str):
        """
        :param input_path: path to the input audio file
        """
        # load with librosa
        super().load(input_path)
        
        # Convert to pydub audio segment
        self.audio_data = librosa_to_pydub(self.data, sr=self.sr)
    
    def transform(self):
        # Load audio files
        noise_file = AudioSegment.from_file(random.choice(self.noise_list))
    
        # Set the desired SNR (signal-to-noise ratio) level in decibels
        SNR_dB = random.randint(self.min_SNR_dB, self.max_SNR_dB)
    
        # Calculate the power of the signal and noise
        signal_power = self.audio_data.dBFS
        noise_power = noise_file.dBFS
    
        # Calculate the scaling factor for the noise
        scaling_factor = SNR_dB * noise_power / signal_power
    
        # Apply the noise to the audio file
        scaled_audio = self.audio_data.apply_gain(scaling_factor)
        self.augmented_audio = scaled_audio.overlay(noise_file)
        
        
    
    