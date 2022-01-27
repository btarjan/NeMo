
import os
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.language_modeling import TransformerLMModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="transformer_lm_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = TransformerLMModel(cfg.model, trainer=trainer)
    pretrained_model = TransformerLMModel.restore_from(restore_path=os.environ["PRETRAINED_PATH"], map_location='cuda')
    model.encoder.load_state_dict(pretrained_model.encoder.state_dict(), strict=True)

    trainer.fit(model)
    trainer.test(model)
    trainer.test(ckpt_path="best")


if __name__ == '__main__':
    main()
