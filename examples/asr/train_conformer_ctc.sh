python ./speech_to_text.py \
    --config-path=conf/conformer \
    --config-name=conformer_ctc_char_hu \
    model.train_ds.manifest_filepath="/data/BEA-Base.json/train-114.json" \
    model.validation_ds.manifest_filepath="/data/BEA-Base.json/dev-spont.json" \
    model.test_ds.manifest_filepath="/data/BEA-Base.json/eval-spont.json" \
    hydra.run.dir="." \
    trainer.gpus=2 \
    trainer.max_epochs=100 \
    model.train_ds.batch_size=32 \
