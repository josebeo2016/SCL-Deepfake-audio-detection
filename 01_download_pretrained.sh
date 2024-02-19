#!/bin/bash

# usage: bash 01_download_pretrained.sh
#
# URL of the file that needs to be downloaded
# Download w2v model
url="https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"

# The folder where the downloaded file should be stored
directory="model/pretrained/"
filename="xlsr2_300m.pt"
# Check if the directory exists. If not, create the directory
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi

# Move to the directory
cd "$directory"

# check if the file exists. If yes, pass
if [ -f "$filename" ]; then
  echo "$filename exists."
  exit 0
fi

