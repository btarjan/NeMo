#!/usr/bin/env bash

eval_ngram()
{
  mkdir -p results
  mkdir -p results/preds
  mkdir -p results/preds/"${LM}"__"$(basename "${TEST%.*}")"
  mkdir -p results/"${ACM}"
  python eval_beamsearch_ngram.py \
    --nemo_model_file am_models/"${ACM}".nemo \
    --input_manifest "${TEST}" \
    --kenlm_model_file ./lm/"${LM}".bin \
    --beam_width "${BEAM_WIDTH}" \
    --beam_alpha 0.6 0.8 1.0 1.2 1.4 \
    --beam_beta 0.5 1.0 1.5 2.0 2.5 \
    --preds_output_folder results/preds/"${LM}"__"$(basename "${TEST%.*}")" \
    --decoding_mode beamsearch_ngram \
    --acoustic_batch_size "${ACM_BS}" \
    --beam_batch_size "${BEAM_BS}" \
    > results/"${ACM}"/"${LM}"__"$(basename "${TEST%.*}")".log
}

#    --beam_alpha 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 \
#    --beam_beta -0.5 0.0 0.5 1.0 1.5 2.0 2.5 \

eval_ngram_with_weight()
{
  mkdir -p results
  mkdir -p results/preds
  mkdir -p results/preds/"${LM}"__"$(basename "${TEST%.*}")"
  mkdir -p results/"${ACM}"
  python eval_beamsearch_ngram.py \
    --nemo_model_file am_models/"${ACM}".nemo \
    --input_manifest "${TEST}" \
    --kenlm_model_file ./lm/"${LM}".bin \
    --beam_width "${BEAM_WIDTH}" \
    --beam_alpha ${ALPHA} \
    --beam_beta ${BETA} \
    --preds_output_folder results/preds/"${LM}"__"$(basename "${TEST%.*}")" \
    --decoding_mode beamsearch_ngram \
    --acoustic_batch_size "${ACM_BS}" \
    --beam_batch_size "${BEAM_BS}" \
    > results/"${ACM}"/"${LM}"__"$(basename "${TEST%.*}")".log
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
  python find_best_WER.py results/"${ACM}"/"${LM}"__"$(basename "${TEST%.*}")".log \
  > results/"${ACM}"/"${LM}"__"$(basename "${TEST%.*}")".log2
}

# Test sets
test_dev_spont=/data/BEA-Base.json/dev-spont.json
test_dev_repet=/data/BEA-Base.json/dev-repet.json
test_eval_spont=/data/BEA-Base.json/eval-spont.json
test_eval_repet=/data/BEA-Base.json/eval-repet.json

# List of test sets
declare -a test_list=("${test_dev_spont}" "${test_eval_spont}")

# List of LMs to test
declare -a LM_list=(train-114_Citrinet-1024-8x-Stride_3gram_kenlm \
                    train-114_Citrinet-1024-8x-Stride_4gram_kenlm \
                    train-114_Citrinet-1024-8x-Stride_5gram_kenlm \
                    train-114_Citrinet-1024-8x-Stride_6gram_kenlm )

# acoustic batch size
ACM_BS=32
# beam batch size
BEAM_BS=128
# beam width
BEAM_WIDTH=80
# acoustic model
ACM=Citrinet-1024-8x-Stride

for LM in "${LM_list[@]}"; do
  for TEST in "${test_list[@]}"; do
    # build kenlm binary if necessary
    build_kenlm_binary
    # eval with n-gram model
    eval_ngram
    # sort parameters by WER
    find_best_WER
  done
done

LM=train-114_spok_Conformer_large-CTC-BPE_pretrained_6gram_kenlm
ACM=Conformer_large-CTC-BPE_pretrained
TEST=${test_eval_repet}
ALPHA=1.6
BETA=2.0

#eval_ngram_with_weight
