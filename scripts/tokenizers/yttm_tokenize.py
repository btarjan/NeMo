
import youtokentome as yttm

data = "../../examples/asr/asr_ctc/tokenizers/text_corpus/document.txt"
model = "../../examples/asr/asr_ctc/tokenizers/train-114_128"
vocab_size = 128
yttm.BPE.train(data, model, vocab_size)
