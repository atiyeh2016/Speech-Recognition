#!/bin/bash

DATA_ROOT=./voxforge/enlang
DATA_MAIN=./data

DATA_SRC="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit" 
DATA_TGZ=${DATA_ROOT}/tgz
DATA_EXTRACT=${DATA_ROOT}/extracted

mkdir -p ${DATA_TGZ} 2>/dev/null

# Check if the executables needed for this script are present in the system
command -v wget >/dev/null 2>&1 ||\
    { echo "\"wget\" is needed but not found"'!'; exit 1; }

echo "--- Starting VoxForge data download (may take some time) ..."
wget -P ${DATA_TGZ} -l 1 -N -nd -c -e robots=off -A tgz -r -np ${DATA_SRC} || \
    { echo "WGET error"'!' ; exit 1 ; }

mkdir -p ${DATA_EXTRACT}

echo "--- Starting VoxForge archives extraction ..."
for a in ${DATA_TGZ}/*.tgz; do
    tar -C ${DATA_EXTRACT} -xf $a
done

mkdir -p ${DATA_MAIN}

mkdir -p ${DATA_MAIN}/train
mkdir -p ${DATA_MAIN}/train/wav
mkdir -p ${DATA_MAIN}/train/txt

mkdir -p ${DATA_MAIN}/test
mkdir -p ${DATA_MAIN}/test/wav
mkdir -p ${DATA_MAIN}/test/txt

mkdir -p ${DATA_MAIN}/val
mkdir -p ${DATA_MAIN}/val/wav
mkdir -p ${DATA_MAIN}/val/txt
# rm -rf ${DATA_TGZ}


# create csv files
python prepare_data.py

# load data
python load_data.py --train-manifest-list ./data_train.csv --valid-manifest-list ./data_val.csv  --test-manifest-list  ./data_test.csv




 