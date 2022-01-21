
from nemo.collections.nlp.models.language_modeling import TransformerLMModel

model = TransformerLMModel.load_from_checkpoint(
    checkpoint_path="results/TransformerLM_PT_spok/2022-01-21_10-52-36/checkpoints/TransformerLM_PT_spok--val_PPL=524.5120-epoch=7.ckpt",
    hparams_file="results/TransformerLM_PT_spok/2022-01-21_10-52-36/hparams.yaml",
    map_location=None,
)

model.save_to("results/TransformerLM_PT_spok/2022-01-21_10-52-36/checkpoints/TransformerLM_PT_spok--val_PPL=524.5120-epoch=7.nemo")
