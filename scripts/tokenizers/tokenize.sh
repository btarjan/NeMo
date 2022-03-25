
VOCAB_SIZE=128
TOKENIZER_TYPE="spe"  # can be wpe or spe
SPE_TYPE="unigram"  # can be bpe or unigram
TARGET_DIR=../../examples/asr/tokenizers

# train tokenizer
#mkdir -p ${TARGET_DIR}
#python ./process_asr_text_tokenizer.py \
#   --manifest="/data/BEA-Base.json/train-114.json" \
#   --data_root=${TARGET_DIR} \
#   --tokenizer=${TOKENIZER_TYPE} \
#   --spe_type=${SPE_TYPE} \
#   --no_lower_case \
#   --log \
#   --vocab_size=${VOCAB_SIZE}

# use tokenizer
mkdir -p tokenized_vocabs
python tokenize_vocab.py \
  ../../examples/asr/tokenizers/tokenizer_spe_unigram_v128/tokenizer.model \
  ../../scripts/asr_language_modeling/ngram_lm/txt/train-114_spok.txt > \
  tokenized_vocabs/train-114_spok-spe_unigram_v128-1best.dic
