import os
import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
import librosa
from core_scripts.data_io import wav_augmentation as nii_wav_aug
from core_scripts.data_io import wav_tools as nii_wav_tools
from datautils.RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
import logging

logging.basicConfig(filename='errors.log', level=logging.DEBUG)
def genList(dir_meta, is_train=False, is_eval=False, is_dev=False):
    # bonafide: 1, spoof: 0
    d_meta = {}
    file_list=[]
    # get dir of metafile only
    dir_meta = os.path.dirname(dir_meta)
    if is_train:
        metafile = os.path.join(dir_meta, 'scp/train_bonafide.lst')
    elif is_dev:
        metafile = os.path.join(dir_meta, 'scp/dev_bonafide.lst')
    elif is_eval:
        metafile = os.path.join(dir_meta, 'scp/test.lst')
        
    with open(metafile, 'r') as f:
        l_meta = f.readlines()
    
    if (is_train):
        for line in l_meta:
            key = line.strip().split()
            file_list.append(key[0])
        return [],file_list
    
    if (is_dev):
        for line in l_meta:
            key = line.strip().split()
            file_list.append(key[0])
        return [],file_list
    
    elif(is_eval):
        for line in l_meta:
            key = line.strip().split()
            file_list.append(key[0])

        return [], file_list
    
def pad(x, padding_type, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    if padding_type == "repeat":
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    elif padding_type == "zero":
        padded_x = np.zeros(max_len)
        padded_x[:x_len] = x
    return padded_x

class Dataset_for(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[], 
                 augmentation_methods=[], num_additional_real=2, num_additional_spoof=2, trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None, aug_dir=None, online_aug=False):
        """
        Args:
            list_IDs (string): Path to the .lst file with real audio filenames.
            vocoders (list): list of vocoder names.
            augmentation_methods (list): List of augmentation methods to apply.
        """
        self.args = args
        self.args.noise_path = noise_path
        self.args.rir_path = rir_path
        self.args.aug_dir = aug_dir
        self.args.online_aug = online_aug
        self.list_IDs = list_IDs
        self.bonafide_dir = os.path.join(base_dir, 'bonafide')
        self.vocoded_dir = os.path.join(base_dir, 'vocoded')
        # list available spoof samples (only .wav files)
        self.spoof_dir = os.path.join(base_dir, 'spoof')
        self.spoof_list = [f for f in os.listdir(self.spoof_dir) if os.path.isfile(os.path.join(self.spoof_dir, f)) and (f.endswith('.wav') or f.endswith('.flac'))]
        self.repeat_pad = True
        self.trim_length = trim_length
        self.sample_rate = wav_samp_rate

        self.vocoders = vocoders
        print("vocoders:", vocoders)
        self.num_additional_spoof = num_additional_spoof
        self.num_additional_real = num_additional_real
        self.augmentation_methods = augmentation_methods
        
        if len(augmentation_methods) < 1:
            # using default augmentation method RawBoostWrapper12
            self.augmentation_methods = ["RawBoost12"]

    def load_audio(self, file_path):
        waveform,_ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        # _, waveform = nii_wav_tools.waveReadAsFloat(file_path)
        return waveform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        # Anchor real audio sample
        real_audio_file = os.path.join(self.bonafide_dir, self.list_IDs[idx])
        real_audio = self.load_audio(real_audio_file)

        # Vocoded audio samples as negative data
        vocoded_audio_files = [os.path.join(self.vocoded_dir, vf + "_" + self.list_IDs[idx]) for vf in self.vocoders]
        vocoded_audios = []
        augmented_vocoded_audios = []
        for vf in vocoded_audio_files:
            vocoded_audio = self.load_audio(vf)
            vocoded_audios.append(np.expand_dims(vocoded_audio, axis=1))
            # Augmented vocoded samples as negative data with first augmentation method
            augmented_vocoded_audio = globals()[self.augmentation_methods[0]](vocoded_audio, self.args, self.sample_rate, audio_path = vf)
            augmented_vocoded_audios.append(np.expand_dims(augmented_vocoded_audio, axis=1))
        
        
        
        # Augmented real samples as positive data
        augmented_audios = []
        for augment in self.augmentation_methods:
            augmented_audio = globals()[augment](real_audio, self.args, self.sample_rate, audio_path = real_audio_file)
            # print("aug audio shape",augmented_audio.shape)
            augmented_audios.append(np.expand_dims(augmented_audio, axis=1))

        # Additional real audio samples as positive data
        idxs = list(range(len(self.list_IDs)))
        idxs.remove(idx)  # remove the current audio index
        additional_idxs = np.random.choice(idxs, self.num_additional_real, replace=False)
        additional_audios = [np.expand_dims(self.load_audio(os.path.join(self.bonafide_dir, self.list_IDs[i])),axis=1) for i in additional_idxs]
        
        # Additional spoof audio samples as negative data
        additional_spoof_idxs = np.random.choice(self.spoof_list, self.num_additional_spoof, replace=False)
        additional_spoofs = [np.expand_dims(self.load_audio(os.path.join(self.spoof_dir, i)),axis=1) for i in additional_spoof_idxs]
        
        # merge all the data
        batch_data = [np.expand_dims(real_audio, axis=1)] + augmented_audios + additional_audios + vocoded_audios + augmented_vocoded_audios + additional_spoofs
        batch_data = nii_wav_aug.batch_pad_for_multiview(
                batch_data, self.sample_rate, self.trim_length, random_trim_nosil=True, repeat_pad=self.repeat_pad)
        batch_data = np.concatenate(batch_data, axis=1)
        # print("batch_data.shape", batch_data.shape)
        
        # return will be anchor ID, batch data and label
        batch_data = Tensor(batch_data)
        # label is 1 for anchor and positive, 0 for vocoded
        label = [1] * (len(augmented_audios) +len(additional_audios) + 1) + [0] * (len(self.vocoders*2) +  len(additional_spoofs))
        # print("label", label)
        return self.list_IDs[idx], batch_data, Tensor(label)

class Dataset_for_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = os.path.join(base_dir, 'eval')
        self.cut=64600 # take ~4 sec audio (64600 samples)
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
            
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        X_pad = pad(X,"zero",self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


# ------------------audio augmentor wrappers---------------------------##
from .audio_augmentor import BackgroundNoiseAugmentor, PitchAugmentor, ReverbAugmentor, SpeedAugmentor, VolumeAugmentor
from .audio_augmentor.utils import pydub_to_librosa, librosa_to_pydub

def background_noise(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "background_noise",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "noise_path": args.noise_path,
        "min_SNR_dB": 5,
        "max_SNR_dB": 15
    }
    bga = BackgroundNoiseAugmentor(config)
    
    bga.load(in_file)
    bga.transform()
    if online:
        audio = bga.augmented_audio
        return pydub_to_librosa(audio)
    else: 
        bga.save()
    


def pitch(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "pitch",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_pitch_shift": -1,
        "max_pitch_shift": 1
    }
    pa = PitchAugmentor(config)
    pa.load(in_file)
    pa.transform()
    if online:
        audio = pa.augmented_audio
        return pydub_to_librosa(audio)
    else:
        pa.save()
    

    
def reverb(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "reverb",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "rir_path": args.rir_path,
    }
    ra = ReverbAugmentor(config)
    ra.load(in_file)
    ra.transform()
    if online:
        audio = ra.augmented_audio
        return pydub_to_librosa(audio)
    else:
        ra.save()

def speed(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "speed",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_speed_factor": 0.9,
        "max_speed_factor": 1.1
    }
    sa = SpeedAugmentor(config)
    sa.load(in_file)
    sa.transform()
    if online:
        audio = sa.augmented_audio
        return pydub_to_librosa(audio)
    else:
        sa.save()

    
def volume(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "volume",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_volume_dBFS": -10,
        "max_volume_dBFS": 10
    }
    va = VolumeAugmentor(config)
    va.load(in_file)
    va.transform()
    if online:
        audio = va.augmented_audio
        return pydub_to_librosa(audio)
    else:
        va.save()

def background_noise_wrapper(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)

    aug_audio_path = os.path.join(aug_dir, 'background_noise', utt_id)
    args.output_path = os.path.join(aug_dir, 'background_noise')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    if (args.online_aug):
        waveform = background_noise(args, utt_id, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            background_noise(args, utt_id)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def pitch_wrapper(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'pitch', utt_id)
    args.output_path = os.path.join(aug_dir, 'pitch')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    if (args.online_aug):
        waveform = pitch(args, utt_id, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            waveform = pitch(args, utt_id)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
    
def reverb_wrapper(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'reverb', utt_id)
    args.output_path = os.path.join(aug_dir, 'reverb')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    if (args.online_aug):
        waveform = reverb(args, utt_id, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            reverb(args, utt_id)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def speed_wrapper(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'speed', utt_id)
    args.output_path = os.path.join(aug_dir, 'speed')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    if (args.online_aug):
        waveform = speed(args, utt_id, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            speed(args, utt_id)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform


#--------------RawBoost data augmentation algorithms---------------------------##
import soundfile as sf
def RawBoost12(x, args, sr = 16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'RawBoost12', utt_id)
    if args.online_aug:
        return process_Rawboost_feature(x, sr,args, algo=5)
    else:
        # check if the augmented file exists
        if (os.path.exists(aug_audio_path)):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            waveform= process_Rawboost_feature(x, sr,args, algo=5)
            # save the augmented file,waveform in np array
            sf.write(aug_audio_path, waveform, sr, subtype='PCM_16')
            return waveform
            

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature
