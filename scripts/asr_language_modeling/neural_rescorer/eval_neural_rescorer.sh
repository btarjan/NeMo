#!/usr/bin/env bash

neural_rescore()
{
  if [ -z "${ALPHA}" ] && [ -z "${BETA}" ]; then
    python eval_neural_rescorer.py \
      --lm_model "${NEURAL_LM}" \
      --beams_file ../ngram_lm/results/preds/${MODEL_AND_TEST}/${PREDS_CONFIG}.tsv \
      --beam_size "${BEAM_SIZE}" \
      --eval_manifest "${TEST}" \
      --scores_output_file ${MODEL_AND_TEST}.tsv \
      > ${MODEL_AND_TEST}.log
  else
    python eval_neural_rescorer.py \
      --lm_model "${NEURAL_LM}" \
      --beams_file ../ngram_lm/results/preds/${MODEL_AND_TEST}/${PREDS_CONFIG}.tsv \
      --beam_size "${BEAM_SIZE}" \
      --eval_manifest "${TEST}" \
      --alpha ${ALPHA} \
      --beta ${BETA} \
      --scores_output_file ${MODEL_AND_TEST}_manual.tsv \
      > ${MODEL_AND_TEST}_manual.log
  fi
}

# Test sets
test_dev_spont=/data/BEA-1/dev-spont-indep/dev-spont-indep.json
test_dev_planned=/data/BEA-1/dev-planned-indep/dev-planned-indep.json
test_eval_spont=/data/BEA-1/eval-spont-indep/eval-spont-indep16.json
test_eval_planned=/data/BEA-1/eval-planned-indep/eval-planned-indep16.json
test_dev_spont_no_empty=../manifests/dev-spont-indep-no_empty.json
test_eval_spont_no_empty=../manifests/eval-spont-indep16-no_empty.json

NEURAL_LM=NYTK/text-generation-news-gpt2-small-hungarian
BEAM_SIZE=80

#############
TEST=${test_dev_spont_no_empty}
MODEL_AND_TEST=train-114__spok_norm_10-1_obh_postproc__ip061_3gram__dev-spont-indep-no_empty
PREDS_CONFIG=preds_out_width80_alpha1.6_beta0.0.tsv
ALPHA=
BETA=

neural_rescore

#############
TEST=${test_eval_spont_no_empty}
MODEL_AND_TEST=train-114__spok_norm_10-1_obh_postproc__ip061_3gram__eval-spont-indep16-no_empty
PREDS_CONFIG=preds_out_width80_alpha1.6_beta0.0.tsv
ALPHA=
BETA=

neural_rescore