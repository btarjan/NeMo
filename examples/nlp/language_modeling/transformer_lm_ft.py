
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.language_modeling import TransformerLMModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import copy

nemo_path = "/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_PT_spok/2022-01-20_23-08-10/checkpoints/TransformerLM_PT_spok.nemo"
ckpt_path="/home/tarjanb/NeMo/examples/nlp/language_modeling/results/TransformerLM_PT_spok/2022-01-20_23-08-10/checkpoints/TransformerLM_PT_spok--val_PPL_849.3261-epoch_9-last.ckpt"

@hydra_runner(config_path="conf", config_name="transformer_lm_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = TransformerLMModel.restore_from(restore_path=nemo_path, map_location='cuda')

    params = model.cfg
    params['train_ds']['file_name'] = 'txts/train-114.txt'
    params['validation_ds']['file_name'] = 'txts/dev-spont-indep.txt'
    params['test_ds']['file_name'] = 'txts/eval-spont-indep.txt'
    # params['test_ds']['tokens_in_batch'] = 8192
    # params['train_ds']['batch_size'] = 8
    new_opt = copy.deepcopy(params.optim)
    new_opt.lr = 0.00005
    # new_opt.sched.max_steps = 200
    # new_opt.sched.name = "ReduceLROnPlateau"
    # new_opt.sched.warmup_ratio = 0.0

    model.setup_training_data(train_data_config=params['train_ds'])
    model.setup_validation_data(val_data_config=params['validation_ds'])
    model.setup_test_data(test_data_config=params['test_ds'])
    model.setup_optimization(optim_config=new_opt)
    print(model.cfg)

    trainer.fit(model)
    trainer.test(model)
    trainer.test(ckpt_path="best")


if __name__ == '__main__':
    main()
