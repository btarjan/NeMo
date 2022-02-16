
## 512 from pretrained
python ./speech_to_text_bpe.py \
    --config-path=conf/citrinet \
    --config-name=citrinet_512 \
    model.train_ds.manifest_filepath="/data/BEA-Base.json/train-114.json" \
    model.validation_ds.manifest_filepath="/data/BEA-Base.json/dev-spont.json" \
    model.test_ds.manifest_filepath="/data/BEA-Base.json/eval-spont.json" \
    model.tokenizer.dir="tokenizers/tokenizer_spe_unigram_v1024" \
    model.tokenizer.type="bpe" \
    hydra.run.dir="." \
    trainer.gpus=2 \
    trainer.max_epochs=100 \
    model.train_ds.batch_size=32 \
    +init_from_pretrained_model="stt_en_citrinet_512"

    #model.optim.lr=2.0 \
    #trainer.accumulate_grad_batches=2

## Large from pretrained
#python ./speech_to_text_rnnt_bpe.py \
#    --config-path=conf/conformer \
#    --config-name=conformer_transducer_bpe \
#    model.train_ds.manifest_filepath="/data/BEA-Base.json/train-114.json" \
#    model.validation_ds.manifest_filepath="/data/BEA-Base.json/dev-spont.json" \
#    model.test_ds.manifest_filepath="/data/BEA-Base.json/eval-spont.json" \
#    model.tokenizer.dir="tokenizers/tokenizer_spe_unigram_v1024" \
#    model.tokenizer.type="bpe" \
#    hydra.run.dir="." \
#    trainer.gpus=2 \
#    trainer.max_epochs=100 \
#    model.train_ds.batch_size=8 \
#    model.optim.lr=2.0 \
#    +init_from_pretrained_model="stt_en_conformer_transducer_large" \
#    trainer.accumulate_grad_batches=4
