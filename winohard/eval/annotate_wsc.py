"""
This script prepares the wsc data into a csv file, ready for annotationg
"""

import argparse
from winohard.utils import read_jsonl


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--in_data", type=str, help="jsonl file",
                       default="/home/lazary/workspace/thesis/winohard/data/wsc/wsc.jsonl")
    parse.add_argument("--out_data", type=str, help="jsonl file",
                       default="/home/lazary/workspace/thesis/winohard/data/wsc/wsc.tsv")

    args = parse.parse_args()

    data = read_jsonl(args.in_data)

    # ignoring the triplet sentence, to keep the format valid
    data = data[:254] + data[255:]

    out_data = []

    for i in range(0, len(data) - 1, 2):
        out_data.append([data[i]['qID'], data[i]['sentence'], data[i]['option1'], data[i]['option2'], data[i]['answer'],
                         data[i + 1]['qID'], data[i + 1]['sentence'], data[i + 1]['option1'], data[i + 1]['option2'],
                         data[i + 1]['answer']])

    with open(args.out_data, 'w') as f:
        f.write('\t'.join(['qID1', 'sentence1', 'option11', 'option12', 'answer1',
                              'qID2', 'sentence2', 'option21', 'option22', 'answer2']) + '\n')
        for obj in out_data:
            f.write('\t'.join(obj) + '\n')


if __name__ == '__main__':
    main()
