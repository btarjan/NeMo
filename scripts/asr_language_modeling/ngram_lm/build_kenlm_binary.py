
import logging
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Build binary KenLM model from an ARPA to be used with beam search decoder of ASR models.'
    )
    parser.add_argument(
        "--arpa_file",
        required=True,
        type=str,
        help="Path to the ARPA LM file that should be converted",
    )
    parser.add_argument(
        "--kenlm_model_file", required=True, type=str, help="The path to store the KenLM binary model file"
    )
    parser.add_argument("--kenlm_bin_path", required=True, type=str, help="The path to the bin folder of KenLM")
    args = parser.parse_args()

    logging.info(f"Running binary_build command on \n\n{args.arpa_file}\n\n")
    kenlm_args = [
        os.path.join(args.kenlm_bin_path, "build_binary"),
        "trie",
        args.arpa_file,
        args.kenlm_model_file,
    ]
    ret = subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)

    if ret.returncode != 0:
        raise RuntimeError("Building KenLM binary was not successful!")


if __name__ == '__main__':
    main()
