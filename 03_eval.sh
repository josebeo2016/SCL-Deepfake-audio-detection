#!/bin/bash
########################
# Script for evaluation
# Usage:
# 
# bash 03_eval <config> <data_path> <batch_size> <model_path> <eval_output>
# <config>: path to config file
#           e.g. configs/conf-3-linear.yaml
# <data_path>: path to the database
#           e.g. DATA/asvspoof_2019_supcon
# <batch_size>: batch size for evaluation
# <model_path>: path to the model
#           e.g. out/model_80_1_1e-07_conf-3-linear/epoch_80.pth
# <eval_output>: path to the output of evaluation
#           e.g. docs/eval_output.txt
#########################




# the name of the training config file (remember to modify the config file before evaluating)
CONFIG=$1
# CONFIG='configs/conf-5-linear.yaml'
# evaluation data path
DATABASE_PATH=$2
# DATABASE_PATH='DATA/asvspoof_2019_supcon'
# path to the directory of the model
BATCH_SIZE=$3
# BATCH_SIZE=64
# Model path
MODEL_PATH=$4
# MODEL_PATH='out/model_80_1_1e-07_conf-5-linear/epoch_80.pth'
# output path
EVAL_OUTPUT=$5
# EVAL_OUTPUT='docs/conf-5-linear_epoch_80.txt'



if [ "$#" -ne 5 ]; then
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

echo -e "${RED}Evaluation starts${NC}"


# Note that you should change the config file before evaluating
com="CUDA_VISIBLE_DEVICES=0 python main.py
    --config ${CONFIG}
    --database_path ${DATABASE_PATH}
    --batch_size ${BATCH_SIZE}
    --eval
    --model_path ${MODEL_PATH}
    --eval_output ${EVAL_OUTPUT}"

echo ${com}
eval ${com}
echo -e "Evaluation process finished"
echo -e "Please calculate the EER by yourself refer to Result.ipynb"

