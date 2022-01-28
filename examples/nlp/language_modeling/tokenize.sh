
VOCAB_SIZE=10000
TOKENIZER_TYPE="spe"  # can be wpe or spe
SPE_TYPE="bpe"  # can be bpe or unigram

mkdir -p tokenizers

python ../../../scripts/tokenizers/process_asr_text_tokenizer.py \
   --data_file="txts/train-114.txt" \
   --data_root="tokenizers" \
   --tokenizer=${TOKENIZER_TYPE} \
   --spe_type=${SPE_TYPE} \
   --no_lower_case \
   --log \
   --vocab_size=${VOCAB_SIZE}
#   --spe_bos
#   --spe_eos
