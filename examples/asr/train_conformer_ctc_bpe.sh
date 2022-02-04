#python ./speech_to_text_bpe.py \
#    --config-path=conf/conformer \
#    --config-name=conformer_ctc_bpe_medium \
#    model.train_ds.manifest_filepath="/data/BEA-Base.json/train-114.json" \
#    model.validation_ds.manifest_filepath="/data/BEA-Base.json/dev-spont.json" \
#    model.test_ds.manifest_filepath="/data/BEA-Base.json/eval-spont.json" \
#    model.tokenizer.dir="tokenizers/tokenizer_spe_unigram_v128" \
#    model.tokenizer.type="bpe" \
#    hydra.run.dir="." \
#    trainer.gpus=2 \
#    trainer.max_epochs=100 \
#    model.train_ds.batch_size=32

python ./speech_to_text_bpe.py \
    --config-path=conf/conformer \
    --config-name=conformer_ctc_bpe_medium \
    model.train_ds.manifest_filepath="/data/BEA-Base.json/train-114.json" \
    model.validation_ds.manifest_filepath="/data/BEA-Base.json/dev-spont.json" \
    model.test_ds.manifest_filepath="/data/BEA-Base.json/eval-spont.json" \
    model.tokenizer.dir="tokenizers/tokenizer_spe_unigram_v128" \
    model.tokenizer.type="bpe" \
    hydra.run.dir="." \
    trainer.gpus=2 \
    trainer.max_epochs=100 \
    model.train_ds.batch_size=32 \
    +init_from_pretrained_model="stt_en_conformer_ctc_medium"

