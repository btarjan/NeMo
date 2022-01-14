#!/usr/bin/env bash

neural_rescore()
{
  MODEL_AND_TEST=${MODEL}__"$(basename "${TEST%.*}")"
  NGRAM_CONFIG=width${BEAM_SIZE}_alpha${NGRAM_ALPHA}_beta${NGRAM_BETA}
  mkdir -p results
  mkdir -p results/"${MODEL_AND_TEST}"

  if [ -z "${ALPHA}" ] && [ -z "${BETA}" ]; then
    python eval_neural_rescorer.py \
      --lm_model "${NEURAL_LM}" \
      --beams_file ../ngram_lm/results/preds/"${MODEL_AND_TEST}"/preds_out_"${NGRAM_CONFIG}".tsv \
      --beam_size "${BEAM_SIZE}" \
      --eval_manifest "${TEST}" \
      --scores_output_file results/"${MODEL_AND_TEST}"/preds_out_"${NGRAM_CONFIG}".tsv \
      > results/"${MODEL_AND_TEST}"/"${NGRAM_CONFIG}".log
  else
    NNLM_CONFIG=${NGRAM_CONFIG}_n-alpha${ALPHA}_n-beta${BETA}
    python eval_neural_rescorer.py \
      --lm_model "${NEURAL_LM}" \
      --beams_file ../ngram_lm/results/preds/"${MODEL_AND_TEST}"/preds_out_"${NGRAM_CONFIG}".tsv \
      --beam_size "${BEAM_SIZE}" \
      --eval_manifest "${TEST}" \
      --alpha "${ALPHA}" \
      --beta "${BETA}" \
      --scores_output_file results/"${MODEL_AND_TEST}"/preds_out_"${NNLM_CONFIG}".tsv \
      > results/"${MODEL_AND_TEST}"/"${NNLM_CONFIG}".log
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
NGRAM_ALPHA=1.6
NGRAM_BETA=0.0

MODEL=train-114__spok_norm_10-1_obh_postproc__ip061_3gram

##############
TEST=${test_dev_spont_no_empty}
ALPHA=
BETA=
neural_rescore

#############
TEST=${test_eval_spont_no_empty}
ALPHA=
BETA=
neural_rescore

#############
TEST=${test_eval_spont_no_empty}
ALPHA=0.369
BETA=0.726
neural_rescore
