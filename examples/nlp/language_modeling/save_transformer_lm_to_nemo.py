
from nemo.collections.nlp.models.language_modeling import TransformerLMModel


def cpkt_to_nemo(checkpoint_path: str):
    model = TransformerLMModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.save_to(checkpoint_path.replace("ckpt", "nemo"))


cpkt_to_nemo("results/TransformerLM_FT_train-114/2022-01-22_18-16-21/checkpoints/TransformerLM_FT_train-114--val_PPL=202.8741-epoch=23.ckpt")
