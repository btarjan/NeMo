
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE


def cpkt_to_nemo(checkpoint_path: str):
    model = EncDecCTCModelBPE.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.save_to(checkpoint_path.replace("ckpt", "nemo"))


cpkt_to_nemo("nemo_experiments/Conformer-CTC-BPE/2022-02-09_01-07-41/checkpoints/Conformer-CTC-BPE--val_wer=0.1675-epoch=92.ckpt")
