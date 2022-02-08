#!/usr/bin/env bash

train_kenlm()
{
  train=${1}
  ngram=${2}
  acm=${3}

  python train_kenlm.py \
    --nemo_model_file am_models/"${acm}".nemo \
    --train_file "${train}" \
    --kenlm_bin_path decoders/kenlm/build/bin/  \
    --kenlm_model_file lm/"$(basename "${train%.*}")"_"${acm}"_"${ngram}"gram_kenlm.bin \
    --ngram_length "${ngram}"
}

#train_kenlm /data/BEA-Base.json/train-114.json 3 Conformer_large-CTC-BPE_pretrained
train_kenlm txt/spok.txt 3 Conformer_large-CTC-BPE_pretrained
train_kenlm txt/spok.txt 4 Conformer_large-CTC-BPE_pretrained
train_kenlm txt/spok.txt 5 Conformer_large-CTC-BPE_pretrained
