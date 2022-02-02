
VOCAB_SIZE=128
TOKENIZER_TYPE="spe"  # can be wpe or spe
SPE_TYPE="unigram"  # can be bpe or unigram
TARGET_DIR=../../examples/asr/asr_ctc/tokenizers

mkdir -p ${TARGET_DIR}

python ./process_asr_text_tokenizer.py \
   --manifest="/data/BEA-Base.json/train-114.json" \
   --data_root=${TARGET_DIR} \
   --tokenizer=${TOKENIZER_TYPE} \
   --spe_type=${SPE_TYPE} \
   --no_lower_case \
   --log \
   --vocab_size=${VOCAB_SIZE}
#   --spe_bos
#   --spe_eos
