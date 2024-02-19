import os
import numpy as np
import librosa  
from pydub import AudioSegment
import subprocess

import logging
logger = logging.getLogger(__name__)

def recursive_list_files(path: str, file_type: str =["wav", "mp3", "flac"]) -> list:
    """Recursively lists all files in a directory and its subdirectories"""
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            real_file_type = filename.split(".")[-1]
            if (real_file_type in file_type):
                files.append(os.path.join(dirpath, filename))
    return files

def pydub_to_librosa(audio_segment: AudioSegment) -> np.ndarray:
    """Convert pydub audio segment to librosa audio data"""
    return np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

def librosa_to_pydub(audio_data: np.ndarray, sr: int =16000) -> AudioSegment:
    """Convert librosa audio data to pydub audio segment"""
    audio_data = np.array(audio_data * (1<<15), dtype=np.int16)
    return AudioSegment(audio_data.tobytes(), 
                    frame_rate=sr,
                    sample_width=audio_data.dtype.itemsize, 
                    channels=1)

def run_cmd(cmd: str) -> str:
    try:
        output = subprocess.call(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        output = e.output
    # return output.decode()

def down_load_model(model_name: str, save_path: str):
    """
    This function help to download the pretrained model of famous ASV spoofing models
    
    :param model_name: the name of the model. Support ["rawnet2", "aasistssl", "xlsr2_300m", "lcnn", "btse"]
    :param save_path: the path to save the pretrained model
    """
    if model_name == "rawnet2":
        if os.path.exists(os.path.join(save_path,"pre_trained_DF_RawNet2.pth")):
            return
        logger.info("Downloading pretrained model for Rawnet2")
        run_cmd("wget https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip")
        run_cmd("unzip pre_trained_DF_RawNet2.zip")
        run_cmd("rm pre_trained_DF_RawNet2.zip")
        run_cmd("mv pre_trained_DF_RawNet2.pth {}".format(os.path.join(save_path,"pre_trained_DF_RawNet2.pth")))
    if model_name == "aasistssl":
        if os.path.exists(os.path.join(save_path,"LA_model.pth")):
            return
        logger.info("Downloading pretrained model for AASIST-SSL")
        run_cmd("gdown 11vFBNKzYUtWqdK358_JEygawFzmmaZjO")
        run_cmd("mv LA_model.pth {}".format(os.path.join(save_path,"LA_model.pth")))
    
    if model_name == "xlsr2_300m":
        if os.path.exists(os.path.join(save_path,"xlsr2_300m.pth")):
            return
        logger.info("Downloading pretrained model for Wav2Vec2.0 (XLSR-2.0-300M)")
        run_cmd("wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt")
        run_cmd("mv xlsr2_300m.pt {}".format(os.path.join(save_path,"xlsr2_300m.pth")))
        
    if model_name == "lcnn":
        if os.path.exists(os.path.join(save_path,"lcnn_full_230209.pth")):
            return
        logger.info("Downloading pretrained model for LCNN")
        run_cmd("gdown 1T2xWKuaFbtJjXdwUVPL7_hDPoKBE-MC8")
        run_cmd("mv lcnn_full_230209.pth {}".format(os.path.join(save_path,"lcnn_full_230209.pth")))
        
    if model_name == "btse":
        if os.path.exists(os.path.join(save_path,"tts_vc_trans_64_concat.pth")):
            return
        logger.info("Downloading pretrained model for BTSE")
        run_cmd("gdown 174L262CCMnvp4YQKG2t07Mrf_mDIf0ul")
        run_cmd("mv tts_vc_trans_64_concat.pth {}".format(os.path.join(save_path,"tts_vc_trans_64_concat.pth")))