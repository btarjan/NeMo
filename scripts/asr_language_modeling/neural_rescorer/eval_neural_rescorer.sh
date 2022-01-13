

neural_rescore()
{
  python eval_neural_rescorer.py \
    --lm_model "${NEURAL_LM}" \
    --beams_file ../ngram_lm/results/preds/${MODEL_AND_TEST}/${PREDS_CONFIG}.tsv \
    --beam_size "${BEAM_SIZE}" \
    --eval_manifest "${TEST}" \
    --scores_output_file ${MODEL_AND_TEST}.tsv \
    > ${MODEL_AND_TEST}.log
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
TEST=${test_dev_spont_no_empty}
MODEL_AND_TEST=train-114_3gram__dev-spont-indep-no_empty
PREDS_CONFIG=preds_out_width80_alpha2.2_beta-0.5

neural_rescore
