
import pytorch_lightning as pl
from nemo.collections.nlp.models.language_modeling import TransformerLMModel

nemo_path = "results/TransformerLM_PT_spok/2022-01-20_15-42-00/checkpoints/TransformerLM_PT_spok--val_PPL=464.9954-epoch=2.nemo"

model = TransformerLMModel.restore_from(restore_path=nemo_path, map_location='cuda')
params = model.cfg

params['train_ds']['file_name'] = 'txts/train-114.txt'
params['validation_ds']['file_name'] = 'txts/dev-spont-indep.txt'
params['test_ds']['file_name'] = 'txts/eval-spont-indep.txt'

model.setup_training_data(train_data_config=params['train_ds'])
model.setup_validation_data(val_data_config=params['validation_ds'])
model.setup_test_data(test_data_config=params['test_ds'])

trainer = pl.Trainer(gpus=1)
trainer.validate(model)
trainer.test(model)
