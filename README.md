# SCL-Deepfake-audio-detection

This is official implementation of our work "Balance, Multiple Augmentation, and Re-synthesis: A Triad Training Strategy
for Enhanced Audio Deepfake Detection"


## Preparing

* Setup conda environment and download wav2vec model by running:
```
bash 00_envsetup.sh
bash 01_download_pretrained.sh
```

* Noise data using in this repo is [MUSAN](https://www.openslr.org/17/) and [RIRS_NOISES](https://www.openslr.org/28/)

Please download those dataset by yourself.

### Dataset
We do not redistributed training data as:
* You should download [ASVspoof 2019](https://doi.org/10.7488/ds/2555) dataset and store bona fide samples online in `DATA/asvspoof_2019_supcon/bonafide`. We do not re-distribute that dataset.
* Vocoded sample can be found in Xin Wang et al. work. We use [voc.v4](https://zenodo.org/record/7314976/files/project09-voc.v4.tar) data. After downloading, you should store vocoded samples in `DATA/asvspoof_2019_supcon/vocoded`.

* Please note that training and dev file MUST be converted into `.wav` file.
* Eval set of ASVSpoof 2019 should be copied (or linked) to `DATA/asvspoof_2019_supcon/eval`
For example:
```
ln -s ./ASVspoof/LA/ASVspoof2019_LA_eval/flac/ DATA/asvspoof_2019_supcon/eval/
```

`DATA` folder should look like this:
```
DATA
├── asvspoof_2019_supcon
│   ├── bonafide
│   │   └── leave_bonafide_wav_here
│   ├── eval
│   │   └── leave_eval_wav_here
│   ├── protocol.txt
│   ├── scp
│   │   ├── dev_bonafide.lst
│   │   ├── test.lst
│   │   └── train_bonafide.lst
│   └── vocoded
│       └── leave_vocoded_wav_here
├── asvspoof_2021_DF
│   ├── flac -> /datab/Dataset/ASVspoof/LA/ASVspoof2021_DF_eval/flac
│   ├── protocol.txt
│   └── trial_metadata.txt
└── in_the_wild
    ├── in_the_wild.txt
    ├── protocol.txt
    └── wav -> /datab/Dataset/release_in_the_wild
```

## Configurations
Configuration should be checked and modified before further training or evaluating. Please read configuration files carefully.

- The MUSAN and RIR_NOISES location should be changed

By default, these configurations is set for training.
## Training
```
CUDA_VISIBLE_DEVICES=0 bash 02_train.sh <seed> <config> <data_path> <comment>
```
For example:
```
CUDA_VISIBLE_DEVICES=0 bash 02_train.sh 1234 configs/conf-3-linear.yaml DATA/asvspoof_2019_supcon "conf-3-linear-1234"
```

## Evaluating
```
CUDA_VISIBLE_DEVICES=0 bash 03_eval <config> <data_path> <batch_size> <model_path> <eval_output>
```
For example:
```
CUDA_VISIBLE_DEVICES=0 bash 03_eval.sh configs/conf-3-linear.yaml DATA/asvspoof_2019_supcon 128 out/model_80_1_1e-07_conf-3-linear/epoch_80.pth docs/la19.txt
```

* Download pre-trained of our conf-3-linear (best model) [here](https://drive.google.com/drive/folders/1F1Wbc_WCdXAOlnly-pgjq1seCtkXgOZP)
To calculate score with pretrained model:
```
CUDA_VISIBLE_DEVICES=0 bash 03_eval.sh configs/conf-3-linear.yaml DATA/asvspoof_2021_DF 128 pretrained/conf-3-linear.pth docs/df21.txt
```

## Calculate EER
Please refer to `Result.ipynb` for calculating EER and other performance metrics.

## Customized training and evaluating dataset
Please refer to `datautils/eval_only.py` and `datautils/asvspoof_2019.py` for other eval dataset. For augmentation strategies, please refer to `datautils/asvspoof_2019_augall_3.py`.
# Reference
* [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)
* [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts)
* [audio_augmentor](https://github.com/josebeo2016/audio_augmentor)
* [SupContrast](https://github.com/HobbitLong/SupContrast)
