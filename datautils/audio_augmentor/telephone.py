from .base import BaseAugmentor
from .utils import recursive_list_files, librosa_to_pydub
import scipy.signal as ss
import librosa
import random
import numpy as np
import torchaudio.functional as F
import logging
import torch
from torchaudio.sox_effects import apply_effects_tensor as sox_fx

logger = logging.getLogger(__name__)
class TelephoneEncodingAugmentor(BaseAugmentor):
    """
    About
    -----

    This class makes audio sound like it's over telephone, by apply one or two things:
    * codec: choose between ALAW, ULAW, GSM
    * bandpass: Optional, can be None. This limits the frequency within common telephone range (300, 3400) be default.
                Note: lowpass value should be higher than that of highpass.

    Example
    -------

    CONFIG = {
        "aug_type": "telephone",
        "output_path": os.path.join(BASE_DIR,"data/augmented"),
        "out_format": "wav",
        "encoding": "ALAW",
        "bandpass": {
            "lowpass": "3400",
            "highpass": "400"
        }
    }
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.encoding = config.get("encoding", "ALAW")
        self.bandpass = config.get("bandpass", None)
        if self.bandpass:
            self.effects = [
                ["lowpass", self.bandpass.get("lowpass", "3400")],
                ["highpass", self.bandpass.get("highpass", "400")],
            ]
        
    def transform(self):
        """
        """
        if self.effects:
            aug_audio, _ = sox_fx(torch.tensor(self.data).reshape(1, -1), self.sr, self.effects)
        else:
            aug_audio = self.data
        aug_audio = F.apply_codec(torch.tensor(aug_audio).reshape(1, -1), self.sr, "wav", encoding=self.encoding)
        self.augmented_audio = librosa_to_pydub(aug_audio)