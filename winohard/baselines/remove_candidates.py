"""
This script is one of the proposed baseline of our work.
It remove the candidates (or replace them with masks) from the original sentence
It takes input file of the form of winogrande data, and creates a new file, in the same format
"""

import argparse
from collections import defaultdict

from tqdm import tqdm
from winohard.utils import read_jsonl, write_jsonl


def get_entity(sentence, option):
    if sentence.count(option) == 1:
        return option
    # handling capitalized the
    if option.startswith('The') and option not in sentence:
        option_temp = option.replace('The', 'the')
        if sentence.count(option_temp) == 1:
            return option_temp
    if option not in sentence and option.startswith('the'):
        option_temp = option.replace('the', 'The')
        if sentence.count(option_temp) == 1:
            return option_temp

    if option.startswith('the'):
        no_the = option.replace('the ', '')
        if sentence.count(no_the) == 1:
            return no_the
    if option.startswith('The'):
        no_the = option.replace('The ', '')
        if sentence.count(no_the) == 1:
            return no_the

    # default
    return None


def remove_candidates(sentence, option1, option2):

    matching1 = get_entity(sentence, option1)
    matching2 = get_entity(sentence, option2)
    if matching1 is None or matching2 is None:
        return None

    if sentence.count(matching1) != 1 or sentence.count(matching2) != 1:
        # print(sentence, option1, option2)
        return None

    no_candidates_sentence = sentence.replace(matching1, '').replace(matching2, '').replace('  ', ' ')
    return no_candidates_sentence


def replace_candidates_with_mask(sentence, option1, option2):

    matching1 = get_entity(sentence, option1)
    matching2 = get_entity(sentence, option2)
    if matching1 is None or matching2 is None:
        return None

    if sentence.count(matching1) != 1 or sentence.count(matching2) != 1:
        # print(sentence, option1, option2)
        return None

    m1_tokens = matching1.split()
    m2_tokens = matching2.split()
    no_candidates_sentence = sentence.replace(matching1, ' '.join(['[MASK]'] * len(m1_tokens)))\
                                     .replace(matching2, ' '.join(['[MASK]'] * len(m2_tokens)))\
                                     .replace('  ', ' ')
    return no_candidates_sentence


def filter_singletons(data):
    pairs = defaultdict(int)
    for row in data:
        pairs[row['pair_id']] += 1

    filtered_data = []
    for row in data:
        if pairs[row['pair_id']] == 2:
            filtered_data.append(row)
    return filtered_data


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--in_data", type=str, help="jsonl file",
                       default="data/winohard/pairs/wsc.jsonl")
    parse.add_argument("--out_data", type=str, help="jsonl file",
                       default="winohard/half/wsc.jsonl")
    parse.add_argument("--masked", action='store_true')

    args = parse.parse_args()

    data = read_jsonl(args.in_data)

    no_cands_data = []
    for obj in tqdm(data):
        sentence = obj['sentence']
        # print(sentence)
        opt1 = obj['option1']
        opt2 = obj['option2']
        if args.masked:
            no_cands = replace_candidates_with_mask(sentence, opt1, opt2)
        else:
            no_cands = remove_candidates(sentence, opt1, opt2)

        if no_cands is not None:
            no_cands_data.append({'qID': obj['qID'],
                                  'sentence': no_cands,
                                  'option1': opt1,
                                  'option2': opt2,
                                  'answer': obj['answer'],
                                  'pair_id': obj['pair_id']
                                  })

            # print(sentence, no_cands)

    paired_no_cands_data = filter_singletons(no_cands_data)

    print('collected {} out of {} total original examples. that makes it {}%'.format(len(no_cands_data), len(data),
                                                                                     len(no_cands_data) / len(data)))
    print('final data after removing singletons: {}'.format(len(paired_no_cands_data)))

    write_jsonl(paired_no_cands_data, args.out_data)


if __name__ == '__main__':
    main()
