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
            if ("LA_T" in key[0]):
                file_list.append(key[0])
        return [],file_list
    
    if (is_dev):
        for line in l_meta:
            key = line.strip().split()
            if ("LA_D" in key[0]):
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
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=['hifigan', 'hn-sinc-nsf', 'hn-sinc-nsf-hifi', 'waveglow'], augmentation_methods=['RawBoost12'], num_additional_real=2, trim_length=64000, wav_samp_rate=16000):
        """
        Args:
            list_IDs (string): Path to the .lst file with real audio filenames.
            vocoders (list): list of vocoder names.
            augmentation_methods (list): List of augmentation methods to apply.
            num_additional_real (int): Number of additional real audio samples to load as positive.
        """
        self.args = args
        self.list_IDs = list_IDs
        self.bonafide_dir = os.path.join(base_dir, 'bonafide')
        self.vocoded_dir = os.path.join(base_dir, 'vocoded')
        
        self.trim_length = trim_length
        self.sample_rate = wav_samp_rate

        self.vocoders = vocoders
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
        augmented_voco = []
        for vf in vocoded_audio_files:
            vocoded_audio = self.load_audio(vf)
            vocoded_audios.append(np.expand_dims(vocoded_audio, axis=1))
            # augment vocoded audio
            for augment in self.augmentation_methods:
                augmented_voco.append(np.expand_dims(globals()[augment](vocoded_audio, self.args, self.sample_rate), axis=1))

        # Augmented real samples as positive data
        augmented_audios = []
        for augment in self.augmentation_methods:
            augmented_audio = globals()[augment](real_audio, self.args, self.sample_rate)
            # print("aug audio shape",augmented_audio.shape)
            augmented_audios.append(np.expand_dims(augmented_audio, axis=1))
            
        # Additional real audio samples as positive data
        idxs = list(range(len(self.list_IDs)))
        idxs.remove(idx)  # remove the current audio index
        additional_idxs = np.random.choice(idxs, self.num_additional_real, replace=False)
        
        # augment additional real audio
        additional_audios = []
        augmented_additional_audios = []
        for augment in self.augmentation_methods:
            for i in additional_idxs:
                additional_audio = self.load_audio(os.path.join(self.bonafide_dir, self.list_IDs[i]))
                additional_audios.append(np.expand_dims(additional_audio,axis=1))
                augmented_additional_audios.append(np.expand_dims(globals()[augment](additional_audio, self.args, self.sample_rate),axis=1))
                

        # merge all the data
        batch_data = [np.expand_dims(real_audio, axis=1)] + augmented_audios + additional_audios + augmented_additional_audios + vocoded_audios + augmented_voco
        batch_data = nii_wav_aug.batch_pad_for_multiview(
                batch_data, self.sample_rate, self.trim_length, random_trim_nosil=True)
        batch_data = np.concatenate(batch_data, axis=1)

        
        # return will be anchor ID, batch data and label
        batch_data = Tensor(batch_data)
        # print("batch_data",batch_data.shape)
        # label is 1 for anchor and positive, 0 for vocoded
        label = [1] * ((1+self.num_additional_real)*(len(self.augmentation_methods)+1)) + [0] * len(self.vocoders*(1+len(self.augmentation_methods)))
        # print("label",label)
        return self.list_IDs[idx], batch_data, Tensor(label)

class Dataset_for_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = os.path.join(base_dir, 'eval')
        self.cut=64000 # take ~4 sec audio (64000 samples)
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
            
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        X_pad = pad(X,"zero",self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id
    


#--------------RawBoost data augmentation algorithms---------------------------##

def RawBoost12(x, args, sr = 16000):
    return process_Rawboost_feature(x, sr,args, algo=5)

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
