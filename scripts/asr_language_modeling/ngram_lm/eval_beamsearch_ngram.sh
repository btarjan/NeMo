#!/usr/bin/env bash

eval_ngram()
{
  mkdir -p results
  mkdir -p results/preds
  mkdir -p results/preds/"${LM}"__"$(basename "${TEST%.*}")"
  python eval_beamsearch_ngram.py \
    --nemo_model_file am_models/QuartzNet15x5_hu.nemo \
    --input_manifest  "${TEST}" \
    --kenlm_model_file ./lm/"${LM}".bin \
    --beam_width "${BEAM_WIDTH}" \
    --beam_alpha 1.6 \
    --beam_beta 0.0 \
    --preds_output_folder results/preds/"${LM}"__"$(basename "${TEST%.*}")" \
    --decoding_mode beamsearch_ngram \
    --acoustic_batch_size "${ACM_BS}" \
    --beam_batch_size "${BEAM_BS}" \
    > results/"${LM}"__"$(basename "${TEST%.*}")".log
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

find_best_WER()
{
  python find_best_WER.py results/"${LM}"__"$(basename "${TEST%.*}")".log \
  > results/"${LM}"__"$(basename "${TEST%.*}")".log2
}

# Test sets
test_dev_spont=/data/BEA-1/dev-spont-indep/dev-spont-indep.json
test_dev_planned=/data/BEA-1/dev-planned-indep/dev-planned-indep.json
test_eval_spont=/data/BEA-1/eval-spont-indep/eval-spont-indep16.json
test_eval_planned=/data/BEA-1/eval-planned-indep/eval-planned-indep16.json
test_dev_spont_no_empty=../manifests/dev-spont-indep-no_empty.json
test_eval_spont_no_empty=../manifests/eval-spont-indep16-no_empty.json

#TEST=${test_eval_spont}
#LM=train-114_3gram

# List of test sets
#declare -a test_list=("${test_dev_spont}" "${test_dev_planned}" \
#  "${test_eval_spont}" "${test_eval_planned}")

declare -a test_list=("${test_dev_spont_no_empty}" "${test_eval_spont_no_empty}")

## List of LMs to test
#declare -a LM_list=(train-114_3gram train-114_4gram train-114_5gram \
#  spok_norm_10-1_obh_postproc_3gram spok_norm_10-1_obh_postproc_4gram \
#  spok_norm_10-1_obh_postproc_5gram train-114__spok_norm_10-1_obh_postproc__ip061_3gram \
#  train-114__spok_norm_10-1_obh_postproc__ip061_3gram_PRUNED \
#  train-114__spok_norm_10-1_obh_postproc__ip061_4gram train-114__spok_norm_10-1_obh_postproc__ip061_5gram)

# List of LMs to test
declare -a LM_list=(train-114__spok_norm_10-1_obh_postproc__ip061_3gram)

# acoustic batch size
ACM_BS=32
# beam batch size
BEAM_BS=128
# beam width
BEAM_WIDTH=80

for TEST in "${test_list[@]}"; do
  for LM in "${LM_list[@]}"; do
    # build kenlm binary if necessary
    build_kenlm_binary
    # eval with n-gram model
    eval_ngram
    # sort parameters by WER
    find_best_WER
  done
done
