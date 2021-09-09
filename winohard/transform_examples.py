"""
This script prepares the wsc data into a csv file, ready for annotationg
"""

import argparse

from winohard.utils import read_jsonl, write_jsonl


def find_unique(s1, s2):
    s1_split = s1.split()
    s2_split = s2.split()

    for w1, w2 in zip(s1_split, s2_split):
        if w1 != w2:
            return w1, w2


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--in_data", type=str, help="jsonl file",
                       default="/home/lazary/workspace/thesis/winohard/data/wsc/wsc.jsonl")
    parse.add_argument("--out_data", type=str, help="jsonl file",
                       default="/home/lazary/workspace/thesis/winohard/data/lms/wsc_lm.jsonl")

    args = parse.parse_args()

    data = read_jsonl(args.in_data)

    # ignoring the triplet sentence, to keep the format valid
    data = data[:254] + data[255:]


    coupled_data = []
    for i in range(0, len(data) - 1, 2):
        couple = data[i], data[i + 1]
        coupled_data.append(couple)

    out_data = []

    for c1, c2 in coupled_data:
        s1 = c1['sentence']
        s2 = c2['sentence']
        w1, w2 = find_unique(s1, s2)
        s1_replace = s1.replace(w1, '[MASK]')
        option1 = c1['option1']
        s1_replace = s1_replace.replace('_', option1)

        s2_replace = s2.replace(w2, '[MASK]')
        option1 = c2['option2']
        s2_replace = s2_replace.replace('_', option1)

        if not s1_replace.endswith('.'):
            s1_replace += '.'
        if not s2_replace.endswith('.'):
            s2_replace += '.'

        if w1.endswith('.'):
            w1 = w1[:-1]
        if w2.endswith('.'):
            w2 = w2[:-1]

        # print(w1, s1_replace)
        # print(w2, s2_replace)
        if s1_replace.count('[MASK]') > 1:
            # print(w1, s1_replace)
            continue
        if s2_replace.count('[MASK]') > 1:
            continue

        out_data.append({'s1': s1_replace, 's2': s2_replace,
                         'ans1': w1, 'ans2': w2})

    write_jsonl(out_data, args.out_data)


if __name__ == '__main__':
    main()
