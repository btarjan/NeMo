
import sys
import sentencepiece as spm

model_path = sys.argv[1]
text_path = sys.argv[2]

s = spm.SentencePieceProcessor(model_file=model_path)

token_set = set()
with open(text_path, 'r') as text:
    for line in text:
        line = line.rstrip()
        token_list = line.split()
        for token in token_list:
            token_set.add(token)

for token in token_set:
    encoded_token = s.encode(token, out_type=str)
    print("{}\t{}".format(token, encoded_token))

