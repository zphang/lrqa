import argparse
import numpy as np
import tqdm.auto as tqdm
import torch


def load_vectors(fname, max_lines):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    arr_data = np.zeros([max_lines, 300])
    keys = []
    for i, line in tqdm.tqdm(enumerate(fin), total=max_lines):
        if i == max_lines:
            break
        tokens = line.rstrip().split(' ')
        arr_data[i] = np.array(list(map(float, tokens[1:])))
        keys.append(tokens[0])
    return {
        "keys": keys,
        "arr_data": arr_data,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasttext_data_path",
        help="e.g. /path/to/crawl-300d-2M.vec",
    )
    parser.add_argument(
        "--num_lines",
        help="Number of words in vocab to use",
        default=100_000,
    )
    parser.add_argument(
        "--output_path",
        help="Location to write fastText embeddings",
    )
    args = parser.parse_args()
    fasttext_embeddings = load_vectors(args.fasttext_data_path, max_lines=args.num_lines)
    torch.save(fasttext_embeddings, args.output_path)


if __name__ == "__main__":
    main()
