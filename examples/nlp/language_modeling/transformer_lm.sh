#!/usr/bin/env bash

mkdir -p results

python transformer_lm.py \
  -cn transformer_lm_config \
  trainer.gpus=2 \
  trainer.max_epochs=10 \
  model.train_ds.tokens_in_batch=16384 \
  +exp_manager.exp_dir=./results \
  +exp_manager.create_tensorboard_logger=True \
  +exp_manager.create_checkpoint_callback=True \
  +exp_manager.checkpoint_callback_params.monitor=val_PPL \
  +exp_manager.checkpoint_callback_params.mode=min \
  +exp_manager.checkpoint_callback_params.save_top_k=5 \
  model.train_ds.file_name=txts/spok_norm_10-1_obh_postproc.txt \
  model.validation_ds.file_name=txts/dev-spont-indep.txt \
  model.test_ds.file_name=txts/eval-spont-indep.txt \
  model.tokenizer.tokenizer_model=tokenizers/spok_norm_10-1_obh_postproc_30k \
  +model.head.use_transformer_init=True \
  +trainer.val_check_interval=0.1 \

