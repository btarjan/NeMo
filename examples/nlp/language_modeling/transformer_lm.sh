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

# pretraining - BEA tokenizer (160 M parameters - GPT2 small)
#############################################################
#python transformer_lm.py \
#  -cn transformer_lm_config_gpt2_small \
#  trainer.gpus=2 \
#  trainer.max_epochs=20 \
#  model.train_ds.tokens_in_batch=8192 \
#  exp_manager.name=TransformerLM_PT_spok_GPT2_small \
#  +exp_manager.exp_dir=./results \
#  +exp_manager.create_tensorboard_logger=True \
#  +exp_manager.create_checkpoint_callback=True \
#  +exp_manager.checkpoint_callback_params.monitor=val_PPL \
#  +exp_manager.checkpoint_callback_params.mode=min \
#  +exp_manager.checkpoint_callback_params.save_top_k=5 \
#  model.train_ds.file_name=txts/spok_norm_10-1_obh_postproc.txt \
#  model.validation_ds.file_name=txts/dev-spont-indep.txt \
#  model.test_ds.file_name=txts/eval-spont-indep.txt \
#  model.tokenizer.tokenizer_name=yttm \
#  model.tokenizer.tokenizer_model=tokenizers/train-114_20k \
#  +model.head.use_transformer_init=True \
#  +trainer.val_check_interval=0.1 \
#  model.optim.lr=1e-4


# finetuning
############
export PRETRAINED_PATH=results/TransformerLM_PT_spok_GPT2_small/2022-01-28_00-06-41/checkpoints/TransformerLM_PT_spok_GPT2_small.nemo

python transformer_lm_ft.py \
  -cn transformer_lm_config_gpt2_small \
  trainer.gpus=2 \
  trainer.max_epochs=20 \
  model.train_ds.tokens_in_batch=4096 \
  exp_manager.name=TransformerLM_FT_train-114_GPT2_small \
  +exp_manager.exp_dir=./results \
  +exp_manager.create_tensorboard_logger=True \
  +exp_manager.create_checkpoint_callback=True \
  +exp_manager.checkpoint_callback_params.monitor=val_PPL \
  +exp_manager.checkpoint_callback_params.mode=min \
  +exp_manager.checkpoint_callback_params.save_top_k=5 \
  model.train_ds.file_name=txts/train-114.txt \
  model.validation_ds.file_name=txts/dev-spont-indep.txt \
  model.test_ds.file_name=txts/eval-spont-indep.txt \
  model.tokenizer.tokenizer_model=tokenizers/train-114_20k \
  +model.head.use_transformer_init=True \
  model.optim.lr=1e-5
