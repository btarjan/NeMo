#!/usr/bin/env bash

mkdir -p results

# pretraining - SPOK tokenizer
##############################
#python transformer_lm.py \
#  -cn transformer_lm_config \
#  trainer.gpus=2 \
#  trainer.max_epochs=10 \
#  model.train_ds.tokens_in_batch=16384 \
#  +exp_manager.exp_dir=./results \
#  +exp_manager.create_tensorboard_logger=True \
#  +exp_manager.create_checkpoint_callback=True \
#  +exp_manager.checkpoint_callback_params.monitor=val_PPL \
#  +exp_manager.checkpoint_callback_params.mode=min \
#  +exp_manager.checkpoint_callback_params.save_top_k=5 \
#  model.train_ds.file_name=txts/spok_norm_10-1_obh_postproc.txt \
#  model.validation_ds.file_name=txts/dev-spont-indep.txt \
#  model.test_ds.file_name=txts/eval-spont-indep.txt \
#  model.tokenizer.tokenizer_model=tokenizers/spok_norm_10-1_obh_postproc_30k \
#  +model.head.use_transformer_init=True \
#  +trainer.val_check_interval=0.1 \

# pretraining - BEA tokenizer (50 M parameters)
###############################################
#python transformer_lm.py \
#  -cn transformer_lm_config \
#  trainer.gpus=2 \
#  trainer.max_epochs=10 \
#  model.train_ds.tokens_in_batch=32768 \
#  exp_manager.name=TransformerLM_PT_spok \
#  +exp_manager.exp_dir=./results \
#  +exp_manager.create_tensorboard_logger=True \
#  +exp_manager.create_checkpoint_callback=True \
#  +exp_manager.checkpoint_callback_params.monitor=val_PPL \
#  +exp_manager.checkpoint_callback_params.mode=min \
#  +exp_manager.checkpoint_callback_params.save_top_k=5 \
#  model.train_ds.file_name=txts/spok_norm_10-1_obh_postproc.txt \
#  model.validation_ds.file_name=txts/dev-spont-indep.txt \
#  model.test_ds.file_name=txts/eval-spont-indep.txt \
#  model.tokenizer.tokenizer_model=tokenizers/train-114_10k \
#  +model.head.use_transformer_init=True \
#  model.optim.sched.name=CosineAnnealing \
#  +trainer.val_check_interval=0.1

# pretraining - BEA tokenizer (150 M parameters - GPT2 small)
#############################################################
python transformer_lm.py \
  -cn transformer_lm_config_gpt2_small \
  trainer.gpus=2 \
  trainer.max_epochs=10 \
  model.train_ds.tokens_in_batch=16384 \
  exp_manager.name=TransformerLM_PT_spok_GPT2_small \
  +exp_manager.exp_dir=./results \
  +exp_manager.create_tensorboard_logger=True \
  +exp_manager.create_checkpoint_callback=True \
  +exp_manager.checkpoint_callback_params.monitor=val_PPL \
  +exp_manager.checkpoint_callback_params.mode=min \
  +exp_manager.checkpoint_callback_params.save_top_k=5 \
  model.train_ds.file_name=txts/spok_norm_10-1_obh_postproc.txt \
  model.validation_ds.file_name=txts/dev-spont-indep.txt \
  model.test_ds.file_name=txts/eval-spont-indep.txt \
  model.tokenizer.tokenizer_model=tokenizers/train-114_10k \
  +model.head.use_transformer_init=True \
  +trainer.val_check_interval=0.1


# finetuning
#python transformer_lm_ft.py \
#  -cn transformer_lm_config \
#  trainer.gpus=2 \
#  trainer.max_epochs=40 \
#  exp_manager.name=TransformerLM_FT_train-114 \
#  +exp_manager.exp_dir=./results \
#  +exp_manager.create_tensorboard_logger=True \
#  +exp_manager.create_checkpoint_callback=True \
#  +exp_manager.checkpoint_callback_params.monitor=val_PPL \
#  +exp_manager.checkpoint_callback_params.mode=min \
#  +exp_manager.checkpoint_callback_params.save_top_k=5 \
