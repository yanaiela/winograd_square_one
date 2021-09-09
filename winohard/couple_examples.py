"""
This script couples sentences from the winograd dataset.
It assumes that the pairs, if existing, are adjacent to each other.
It uses nltk to check the edit distance of the two sentences, and assign them to be a pair, if the edit distance
is at least 70% of the shorter sentence.

In the case of wsc dataset, the pairs are already ordered, so using that as the pairing method.
"""

import argparse
import json
from copy import deepcopy
import nltk
from winohard.utils import read_jsonl, write_jsonl


def is_minimal_pair(s1, s2):
    tokens1 = s1.split()
    tokens2 = s2.split()
    edit_distance = nltk.edit_distance(tokens1, tokens2)

    min_len = min(len(tokens1), len(tokens2))

    if edit_distance / min_len > 0.3:
        return False
    else:
        return True


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--in_data", type=str, help="jsonl file",
                       default="data/wsc/wsc.jsonl")
    parse.add_argument("--out_data", type=str, help="jsonl file",
                       default="data/winohard/pairs/dev.jsonl")
    parse.add_argument("--wsc", action="store_true", help="use wsc data? (default is False)")

    args = parse.parse_args()

    data = read_jsonl(args.in_data)

    pair_data = []
    if args.wsc:
        pid = 0
        # ignoring the triplet sentence, to keep the format valid
        data = data[:254] + data[255:]

        for i in range(0, len(data) - 1, 2):
            cur_obj = deepcopy(data[i])
            next_obj = deepcopy(data[i + 1])
            cur_obj['pair_id'] = pid
            next_obj['pair_id'] = pid

            pair_data.append(cur_obj)
            pair_data.append(next_obj)
            pid += 1

    else:
        pid = 0
        for i in range(len(data) - 1):
            cur_obj = data[i]
            next_obj = data[i + 1]
            cur_sentence = cur_obj['sentence']
            next_sentence = next_obj['sentence']

            pair = is_minimal_pair(cur_sentence, next_sentence)
            if pair:
                cur_obj['pair_id'] = pid
                next_obj['pair_id'] = pid
                pair_data.append(cur_obj)
                pair_data.append(next_obj)
                pid += 1

    print('collected {} pairs, out of {} total original examples. that makes it {}%'.format(pid, len(data),
                                                                                            (pid * 2) / len(data)))

    write_jsonl(pair_data, args.out_data)


if __name__ == '__main__':
    main()
