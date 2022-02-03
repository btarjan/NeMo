
import youtokentome as yttm

data = "../../examples/asr/tokenizers/text_corpus/document.txt"
model = "../../examples/asr/tokenizers/train-114_128"
vocab_size = 128
yttm.BPE.train(data, model, vocab_size)
