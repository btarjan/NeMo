
import sys
import re

log_path = sys.argv[1]

decoding_param_pattern = re.compile("preds_out_(.+).tsv")
WER_pattern = re.compile(" = ([\d\.]+)%/")
result_dict = {}
i = 0

with open(log_path) as log_file:
    for line in log_file:
        i += 1
        if "Stored the predictions of beam search decoding at" in line:
            decoding_param_match = decoding_param_pattern.search(line)
            decoding_param = decoding_param_match.group(1)
            match_index = i
        if "WER/CER with beam search decoding and N-gram model" in line:
            assert i == match_index + 1
            WER_match = WER_pattern.search(line)
            WER = WER_match.group(1)
            result_dict[decoding_param] = float(WER)

        sorted_results = sorted(result_dict.items(), key=lambda kv: kv[1])

print(sorted_results)