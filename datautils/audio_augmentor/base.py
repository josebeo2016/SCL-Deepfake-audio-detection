import librosa
import os
import typing

import logging

logger = logging.getLogger(__name__)


class BaseAugmentor:
    """
    Basic augmentor class requires these config:
    aug_type: str, augmentation type
    output_path: str, output path
    out_format: str, output format
    """

    def __init__(self, config: dict):
        """
        This method initialize the `BaseAugmentor` object.
        """
        self.config = config
        self.aug_type = config["aug_type"]
        
        self.output_path = config["output_path"]
        self.out_format = config["out_format"]
        self.augmented_audio = None
        self.data = None
        self.sr = 16000

    def load(self, input_path: str):
        """
        Load audio file and normalize the data
        Librosa done this part
        self.data: audio data in numpy array (librosa load)
        :param input_path: path to the input audio file      
        """
        self.input_path = input_path
        self.file_name = self.input_path.split("/")[-1].split(".")[0]
        # load with librosa and auto resample to 16kHz
        self.data, self.sr = librosa.load(self.input_path, sr=self.sr)

        # Convert to mono channel
        self.data = librosa.to_mono(self.data)

    def transform(self):
        """
        Transform audio data (librosa load) to augmented audio data (pydub audio segment)
        Note that self.augmented_audio is pydub audio segment
        """
        raise NotImplementedError

    def save(self):
        """
        Save augmented audio data (pydub audio segment) to file
        self.out_format: output format
        This done the codec transform by pydub
        """
        self.augmented_audio.export(
            os.path.join(self.output_path, self.file_name + "." + self.out_format),
            format=self.out_format,
        )

