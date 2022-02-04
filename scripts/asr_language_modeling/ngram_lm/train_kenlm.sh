#!/usr/bin/env bash

train_kenlm()
{
  acm=${1}
  ngram=${2}

  python train_kenlm.py \
    --nemo_model_file am_models/"${acm}".nemo \
    --train_file /data/BEA-Base.json/train-114.json \
    --kenlm_bin_path decoders/kenlm/build/bin/  \
    --kenlm_model_file lm/BEA_"${acm}"_"${ngram}"gram_kenlm.bin \
    --ngram_length "${ngram}"
}

train_kenlm QuartzNet15x5_hu 3
train_kenlm Conformer-CTC-Char 3
train_kenlm Conformer-CTC-BPE 3
