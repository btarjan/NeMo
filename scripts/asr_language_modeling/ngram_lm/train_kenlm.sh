#!/usr/bin/env bash

train_kenlm()
{
  python train_kenlm.py \
    --nemo_model_file am_models/QuartzNet15x5_hu.nemo \
    --train_file txt/"${1}".txt \
    --kenlm_bin_path decoders/kenlm/build/bin/  \
    --kenlm_model_file lm/"${1}"_"${2}"gram_kenlm.bin \
    --ngram_length "${2}"
}

train_kenlm train-114 3
train_kenlm train-114 4
train_kenlm train-114 5
