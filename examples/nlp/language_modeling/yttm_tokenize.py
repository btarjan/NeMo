
import youtokentome as yttm

data = "txts/spok_norm_10-1_obh_postproc.txt"
model = "tokenizers/spok_norm_10-1_obh_postproc_30k"
vocab_size = 30000
yttm.BPE.train(data, model, vocab_size)
