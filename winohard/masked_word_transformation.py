"""
Creates a version of winograd-like examples, that can be evaluated with MLM
"""

import argparse
from collections import defaultdict
from copy import deepcopy

from winohard.utils import read_jsonl, write_jsonl


def find_unique(s1, s2):
    s1_split = s1.split()
    s2_split = s2.split()
    pair = None
    for w1, w2 in zip(s1_split, s2_split):
        # in case of more than a single word that differs. e.g.
        # The painting in Mark's living room shows an oak tree. _ is to the right of the bookcase.
        # The painting in Mark's living room shows an oak tree. _ is to the right of a house.
        if pair and w1 != w2:
            # print(s1)
            # print(s2)
            return None
        if w1 != w2:
            pair = w1, w2
    return pair


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--paired_data", type=str, help="jsonl file",
                       default="data/wsc/wsc.jsonl")
    parse.add_argument("--out_lm_file", type=str, help="jsonl file",
                       default="data/lms/wsc_lm.jsonl")
    parse.add_argument("--out_wino_file", type=str, help="jsonl file",
                       default="data/winohard/wsc_transformed.jsonl")

    args = parse.parse_args()

    data = read_jsonl(args.paired_data)

    coupled_d_data = defaultdict(list)
    for d in data:
        coupled_d_data[d['pair_id']].append(d)

    coupled_data = []
    for k, vals in coupled_d_data.items():
        if len(vals) != 2:
            continue
        coupled_data.append(vals)

    ordered_coupled_data = []
    for k, vals in coupled_d_data.items():
        if vals[0]['answer'] == "1":
            ordered_coupled_data.append(vals)
        else:
            ordered_coupled_data.append([vals[1], vals[0]])

    out_data = []
    orig_transformed_data = []

    for c1, c2 in ordered_coupled_data:
        s1 = c1['sentence']
        s2 = c2['sentence']
        tup = find_unique(s1, s2)
        if tup is None:
            continue
        w1, w2 = tup

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

        if s1_replace.count('[MASK]') > 1:
            # print(w1, s1_replace)
            continue
        if s2_replace.count('[MASK]') > 1:
            continue

        out_data.append({'s1': s1_replace, 's2': s2_replace,
                         'ans1': w1, 'ans2': w2})

        s1_replace = deepcopy(s1_replace).replace('[MASK]', '_')
        s2_replace = deepcopy(s2_replace).replace('[MASK]', '_')

        c1t = deepcopy(c1)
        c1t['sentence'] = s1_replace
        c1t['option1'] = w1
        c1t['option2'] = w2
        # c1t['answer'] = '1'

        c2t = deepcopy(c2)
        c2t['sentence'] = s2_replace
        c2t['option1'] = w1
        c2t['option2'] = w2
        # c2t['answer'] = '2'

        orig_transformed_data.append(c1t)
        orig_transformed_data.append(c2t)

    print('original data size: {}, converted data: {}. that makes it {:.2f}%.'.format(len(data),
                                                                                      len(orig_transformed_data),
                                                                                      100. *
                                                                                      len(orig_transformed_data) / len(
                                                                                          data)
                                                                                      ))
    write_jsonl(out_data, args.out_lm_file)
    write_jsonl(orig_transformed_data, args.out_wino_file)


if __name__ == '__main__':
    main()
