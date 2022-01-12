
python eval_neural_rescorer.py \
    --lm_model NYTK/text-generation-news-gpt2-small-hungarian \
    --beams_file ../ngram_lm/results/preds/train-114__spok_norm_10-1_obh_postproc__ip061_3gram__dev-spont-indep/preds_out_width80_alpha1.6_beta0.0.tsv \
    --beam_size 80 \
    --eval_manifest /data/BEA-1/dev-spont-indep/dev-spont-indep.json
