from .base import BaseAugmentor
from .utils import recursive_list_files, librosa_to_pydub
import scipy.signal as ss
import librosa
import random
import numpy as np
import torchaudio.functional as F
from torchaudio.io import AudioEffector
import logging
import torch
from torchaudio.sox_effects import apply_effects_tensor as sox_fx

logger = logging.getLogger(__name__)

def apply_codec(waveform, sample_rate, format, encoder=None):
    encoder = torchaudio.io.AudioEffector(format=format, encoder=encoder)
    return encoder.apply(waveform, sample_rate)

def apply_effect(waveform, sample_rate, effect):
    effector = torchaudio.io.AudioEffector(effect=effect)
    return effector.apply(waveform, sample_rate)
class TelephoneEncodingAugmentor(BaseAugmentor):
    """
    About
    -----

    This class makes audio sound like it's over telephone, by apply one or two things:
    * codec: choose between ALAW, ULAW, g722
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
        self.encoding = config.get("encoding", "g722")
        self.bandpass = config.get("bandpass", None)
        if self.bandpass:
            self.effects = ",".join(
                [
                    "lowpass=frequency=4000:poles=1",
                    "compand=attacks=0.02:decays=0.05:points=-60/-60|-30/-10|-20/-8|-5/-8|-2/-8:gain=-8:volume=-7:delay=0.05",
                ]
            )
        
    def transform(self):
        """
        """
        torch_audio = torch.tensor(self.data).reshape(1, -1)
        
        if self.effects:
            # aug_audio, _ = sox_fx(torch.tensor(self.data).reshape(1, -1), self.sr, self.effects)
            aug_audio = apply_effect(torch_audio.T, self.sr, self.effects)
        else:
            aug_audio = torch_audio
        codec_applied = apply_codec(aug_audio, self.sr, self.encoding).T
        # convert to numpy array
        aug_audio = codec_applied.numpy().flatten()
        
        self.augmented_audio = librosa_to_pydub(aug_audio)