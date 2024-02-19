#!/bin/bash
########################
# Script for training
# Usage:
# 
# bash 02_train.sh <seed> <config> <data_path> <comment>
# <seed>: random seed to initialize the weight
# <config>: path to config file
#           e.g. configs/conf-3-linear.yaml
# <data_path>: path to the database
#           e.g. DATA/asvspoof_2019_supcon
# <comment>: comment for the training
#           the subfolder name of the model
#########################




# the random seed
SEED=$1
# the name of the training config file 
CONFIG=$2
# path to the directory of the model
DATABASE_PATH=$3
# flag
CMT=$4

if [ "$#" -ne 4 ]; then
    echo -e "Invalid input arguments. Please check the doc of script."
    exit 1;
fi

# Enter conda environment
eval "$(conda shell.bash hook)"

conda activate fairseq
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Cannot load fairseq, please run 00_envsetup.sh first"
    exit 1
fi

echo -e "${RED}Training starts${NC}"
echo -e "Training log are writing to $PWD/logs/model_80_1_1e-07_${CMT}"
echo -e "Model save to $PWD/out/model_80_1_1e-07_${CMT}"
com="CUDA_VISIBLE_DEVICES=0 python main.py
    --seed ${SEED}
    --config ${CONFIG}
    --database_path ${DATABASE_PATH}
    --batch_size 1
    --comment "${CMT}"
    --num_epochs 80
    --lr 0.0000001"

echo ${com}
eval ${com}
echo -e "Training process finished"



