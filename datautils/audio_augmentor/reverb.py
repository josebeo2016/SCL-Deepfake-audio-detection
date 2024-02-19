from .base import BaseAugmentor
from .utils import recursive_list_files, librosa_to_pydub
import scipy.signal as ss
import librosa
import random
import numpy as np

import logging
logger = logging.getLogger(__name__)
class ReverbAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Reverb augmentation
        Config:
        rir_path: str, path to the folder containing RIR files 
        (RIR dataset example https://www.openslr.org/28/)
        """
        super().__init__(config)
        self.rir_path = config["rir_path"]
        self.rir_file = self.select_rir(self.rir_path)
        
    def select_rir(self, rir_path):
        """
        Randomly select the RIR file from the `rir_path`
        
        :param rir_path: path to the folder containing RIR files
        
        :return: path to the selected RIR file
        """
        rir_list = recursive_list_files(rir_path)
        return random.choice(rir_list)
        
    def transform(self):
        """
        Reverb the audio by convolving with the RIR file selected from `rir_path`
        """
        rir_data, _ = librosa.load(self.rir_file, sr=self.sr)
        # Compute convolution
        reverberate = np.convolve(self.data, rir_data)
        # Normalize output signal to avoid clipping
        reverberate /= (np.max(np.abs(reverberate)))
        
        # transform to pydub audio segment
        self.augmented_audio = librosa_to_pydub(reverberate, sr=self.sr)
    
        