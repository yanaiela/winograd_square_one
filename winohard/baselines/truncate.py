"""
This script is one of the proposed baseline of our work.
It splits sentences based on some discourse markers (e.g. `so', `but', `because', etc.)
It takes input file of the form of winogrande data, and creates a new file, in the same format
"""

import argparse

import spacy
from tqdm import tqdm

from winohard.utils import read_jsonl, write_jsonl

nlp = spacy.load("en_core_web_sm")


def sentence_split(sentence):
    doc = nlp(sentence)

    tokens = [x.text for x in doc]
    comp_ind = tokens.index('_')

    discourse_markers = ['so', 'but', 'and',
                         'because', 'although', 'though', 'due', 'since',
                         '.', ',', ';', '?',
                         ]

    for i in range(comp_ind - 1, -1, -1):
        w = doc[i]
        if w.text in discourse_markers:
            rest = doc[w.i + 1:].text
            return rest

    return None


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--in_data", type=str, help="jsonl file",
                       default="data/winohard/pairs/wsc.jsonl")
    parse.add_argument("--out_data", type=str, help="jsonl file",
                       default="winohard/truncated/wsc.jsonl")

    args = parse.parse_args()

    data = read_jsonl(args.in_data)

    truncate_data = []
    for obj in tqdm(data):
        sentence = obj['sentence']
        # print(sentence)
        opt1 = obj['option1']
        opt2 = obj['option2']
        subsequence = sentence_split(sentence)
        if subsequence is not None:
            truncate_data.append({'qID': obj['qID'],
                                  'sentence': subsequence,
                                  'option1': opt1,
                                  'option2': opt2,
                                  'answer': obj['answer'],
                                  'pair_id': obj['pair_id']
                                  })

    print('collected {} out of {} total original examples. that makes it {}%'.format(len(truncate_data), len(data),
                                                                                     len(truncate_data) / len(data)))
    write_jsonl(truncate_data, args.out_data)


if __name__ == '__main__':
    main()
