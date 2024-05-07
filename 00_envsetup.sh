#!/bin/bash
# Install dependency for fairseq

# Name of the conda environment
ENVNAME=fairseq

eval "$(conda shell.bash hook)"
conda activate ${ENVNAME}
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Install conda environment ${ENVNAME}"
    
    # conda env
    conda create -n ${ENVNAME} python=3.9 pip --yes
    conda activate ${ENVNAME}

    # install pytorch
    echo "===========Install pytorch==========="
    # conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
    # pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118



    # git clone fairseq
    #  fairseq 0.10.2 on pip does not work
    # git clone https://github.com/pytorch/fairseq
    # cd fairseq
    pip install git+https://github.com/facebookresearch/fairseq.git@a54021305d6b3c4c5959ac9395135f63202db8f1

    # install scipy
    pip install scipy==1.7.3

    # install pandas
    pip install pandas==1.3.5

    # install protobuf
    pip install protobuf==3.20.3

    # install tensorboard
    pip install tensorboard==2.6.0
    pip install tensorboardX==2.6

    # install librosa
    pip install librosa==0.10.0

    # install pydub
    pip install pydub==0.25.1

else
    echo "Conda environment ${ENVNAME} has been installed"
fi

