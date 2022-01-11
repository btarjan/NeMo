#!/usr/bin/env bash

eval_ngram()
{
  python eval_beamsearch_ngram.py \
    --nemo_model_file am_models/QuartzNet15x5_hu.nemo \
    --input_manifest  "${TEST}" \
    --kenlm_model_file ./lm/"${LM}".bin \
    --beam_width 80 \
    --beam_alpha 1.6 1.8 2.0 2.2 2.4 2.6 3.0 \
    --beam_beta  0.0   \
    --preds_output_folder results/amp_sp_metasen_from_pretrained \
    --decoding_mode beamsearch_ngram \
    > results/"${LM}"__"$(basename ${TEST/%.*})".log
}

build_kenlm_binary()
{
  if [ ! -f lm/"${LM}".bin ]; then
    python build_kenlm_binary.py \
      --arpa_file ./lm/"${LM}".lm \
      --kenlm_model_file ./lm/"${LM}".bin \
      --kenlm_bin_path decoders/kenlm/build/bin/
  else
    echo -e "${LM}.bin already exists!\nBinarization skipped!"
  fi
}

test_dev_spont=/data/BEA-1/dev-spont-indep/dev-spont-indep.json
test_dev_planned=/data/BEA-1/dev-planned-indep/dev-planned-indep.json
test_eval_spont=/data/BEA-1/eval-spont-indep/eval-spont-indep16.json
test_eval_planned=/data/BEA-1/eval-planned-indep/eval-planned-indep16.json

TEST=${test_dev_spont}
LM=train-114_3gram

# build kenlm binary if necessary
build_kenlm_binary

# eval with n-gram model
eval_ngram

