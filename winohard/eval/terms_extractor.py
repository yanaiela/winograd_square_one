"""
This script is one of the proposed baseline of our work.
It remove the candidates (or replace them with masks) from the original sentence
It takes input file of the form of winogrande data, and creates a new file, in the same format
"""

import argparse

import spacy
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from typing import List, Dict
from functools import lru_cache
from winohard.utils import read_jsonl, write_jsonl


@lru_cache(maxsize=None)
def get_nlp():
    nlp = spacy.load("en_core_web_sm")
    return nlp


@lru_cache(maxsize=None)
def annotate_sentence(nlp, text):
    doc = nlp(text)
    return doc


def extract_terms(instance):
    sentence = instance['sentence']
    opt1 = instance['option1'].lower().split()
    opt2 = instance['option2'].lower().split()
    entities = opt1 + opt2
    nlp = get_nlp()
    # doc = nlp(sentence.replace('_', 'it'))
    doc = annotate_sentence(nlp, sentence.replace('_', 'it'))
    terms = []
    for w in doc:
        if w.text.lower() in entities:
            continue
        if w.text.lower() in ['that']:
            continue
        if w.pos_ in ['VERB', 'NOUN', 'ADJ', 'SCONJ']:
            terms.append(w.text)
        elif w.lemma_ in ['not']:
            terms.append(w.text)
    # print(sentence, opt1, opt2, terms)
    return terms


def get_dataset_terms(data) -> List[Dict]:
    data_terms_augmented = []

    for datum in tqdm(data):
        terms = extract_terms(datum)
        datum_copy = deepcopy(datum)
        datum_copy['terms'] = terms
        data_terms_augmented.append(datum_copy)
    return data_terms_augmented


def find_most_similar_instance(test_instance, train_data):
    test_terms = test_instance['terms']
    test_term_len = len(test_terms)
    max_intersection = 0
    best_train_intersection = None
    for train_datum in train_data:
        train_terms = train_datum['terms']
        intersection_len = len(set(train_terms) & set(test_terms))
        if intersection_len > max_intersection:
            max_intersection = intersection_len
            best_train_intersection = train_datum
        if max_intersection == test_term_len:
            best_train_intersection = train_datum
            max_intersection = intersection_len
            break
    return best_train_intersection, max_intersection / test_term_len


def find_terms_overlaps(train_data_terms, test_data_terms):
    overlap = []
    high_overlap_data = []
    low_overlap_data = []
    for test_datum in test_data_terms:

        best_train_intersection, overlap_percantage = find_most_similar_instance(test_datum, train_data_terms)
        if overlap_percantage > 0.5:
            high_overlap_data.append(test_datum)
        else:
            low_overlap_data.append(test_datum)
        overlap.append(overlap_percantage)
    return overlap, low_overlap_data, high_overlap_data


def filter_singletons(data):
    pair_ids_count = defaultdict(int)

    for datum in data:
        pair_ids_count[datum['pair_id']] += 1

    filter_data = []
    for datum in data:
        if pair_ids_count[datum['pair_id']] == 2:
            filter_data.append(datum)
    return filter_data


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--train_file", type=str, help="jsonl file",
                       default="data/winogrande/train_xl.jsonl")
    parse.add_argument("--test_file", type=str, help="jsonl file",
                       default="data/winohard/pairs/wsc.jsonl")
    parse.add_argument("--out_dir", type=str, help="jsonl file",
                       default="data/winohard/overlaps/")

    args = parse.parse_args()

    test_data = read_jsonl(args.test_file)
    train_data = read_jsonl(args.train_file)

    terms = extract_terms(test_data[0])
    print(terms)

    test_terms = get_dataset_terms(test_data)
    train_terms = get_dataset_terms(train_data)

    overlaps, low_overlap_test, high_overlap_test = find_terms_overlaps(train_terms, test_terms)
    # print()
    high_overlap_sum = sum([x > 0.5 for x in overlaps])
    print(high_overlap_sum, len(overlaps), high_overlap_sum / len(overlaps))

    twin_high_overlap = filter_singletons(high_overlap_test)
    twin_low_overlap = filter_singletons(low_overlap_test)
    print('high overlap total: {}, after filtering: {}'.format(len(high_overlap_test), len(twin_high_overlap)))
    print('low overlap total: {}, after filtering: {}'.format(len(low_overlap_test), len(twin_low_overlap)))

    data_name = args.test_file.split('/')[-1].split('.')[0]

    write_jsonl(twin_low_overlap, args.out_dir + f'{data_name}_low_overlap2.jsonl')
    write_jsonl(twin_high_overlap, args.out_dir + f'{data_name}_high_overlap2.jsonl')


if __name__ == '__main__':
    main()
