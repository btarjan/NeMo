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

#NEURAL_LM=NYTK/text-generation-news-gpt2-small-hungarian
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM/2022-01-15_09-49-46/checkpoints/TransformerLM.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM/2022-01-15_09-49-46/checkpoints/TransformerLM--val_PPL=446.0944-epoch=1.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_FT_train-114/2022-01-20_12-35-49/checkpoints/TransformerLM_FT_train-114.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_PT_spok/2022-01-20_23-08-10/checkpoints/TransformerLM_PT_spok.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_PT_spok/2022-01-20_15-42-00/checkpoints/TransformerLM_PT_spok--val_PPL=464.9954-epoch=2.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_PT_spok/2022-01-21_10-52-36/checkpoints/TransformerLM_PT_spok.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_PT_spok/2022-01-21_10-52-36/checkpoints/TransformerLM_PT_spok--val_PPL=524.5120-epoch=7.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_PT_spok/2022-01-21_00-16-02/checkpoints/TransformerLM_PT_spok.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_FT_train-114/2022-01-21_20-14-27/checkpoints/TransformerLM_FT_train-114--val_PPL=157.5620-epoch=14.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_FT_train-114/2022-01-21_20-14-27/checkpoints/TransformerLM_FT_train-114.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_FT_train-114/2022-01-22_18-07-39/checkpoints/TransformerLM_FT_train-114--val_PPL=96.2513-epoch=11.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_FT_train-114/2022-01-22_18-16-21/checkpoints/TransformerLM_FT_train-114--val_PPL=202.8741-epoch=23.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_FT_train-114_GPT2_small/2022-01-24_10-31-58/checkpoints/TransformerLM_FT_train-114_GPT2_small--val_PPL=146.5031-epoch=26.nemo
#NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_FT_train-114_GPT2_small/2022-01-25_08-52-01/checkpoints/TransformerLM_FT_train-114_GPT2_small--val_PPL=142.2549-epoch=26.nemo
NEURAL_LM=/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_FT_train-114_GPT2_small/2022-01-28_12-24-19/checkpoints/TransformerLM_FT_train-114_GPT2_small.nemo
BEAM_SIZE=80
NGRAM_ALPHA=1.6
NGRAM_BETA=0.0

MODEL=train-114__spok_norm_10-1_obh_postproc__ip061_3gram

###############
#TEST=${test_dev_spont_no_empty}
#ALPHA=
#BETA=
#neural_rescore
#
##############
#TEST=${test_eval_spont_no_empty}
#ALPHA=
#BETA=
#neural_rescore

###################
TEST=${test_eval_spont_no_empty}
ALPHA=0.98
BETA=0.691
neural_rescore
