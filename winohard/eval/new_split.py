"""
Creating a new splits of data, while keeping more data for evaluation.
Taking parts of the training data for eval, while making sure it from the debiased-data.
"""

import argparse
from collections import defaultdict
from typing import List, Set, Dict, Tuple

from winohard.filter2single_tokens_lms import get_tokenizers, filter_data
from winohard.utils import read_jsonl, write_jsonl


def create_test_set(train_data: List[Dict], debiased_ids: Set[str]) -> Tuple[List[Dict], List[Dict]]:
    debiased_test = []
    new_train = []

    debiased_pairs = defaultdict(list)

    for instance in train_data:
        if instance['qID'] in debiased_ids:
            debiased_pairs[instance['pair_id']].append(instance)
        else:
            new_train.append(instance)

    for instance_id, instances in debiased_pairs.items():
        if len(instances) == 2:
            debiased_test.append(instances[0])
            debiased_test.append(instances[1])
        else:
            assert len(instances) == 1
            new_train.append(instances[0])

    # leaving 2000 examples (1000 pairs) to the new test, the rest goes to training
    if len(debiased_test) > 2000:
        rest = debiased_test[2000:]
        debiased_test = debiased_test[:2000]
        new_train.extend(rest)
    return new_train, debiased_test


def create_train_sizes(train_data: List[Dict]) -> List[List[Dict]]:
    size_range = list(range(2000, len(train_data), 2000))
    train_sizes = [100, 500, 1000] + size_range

    varied_training_data = []

    # creating different splits
    for size in train_sizes:
        varied_training_data.append(train_data[:size])

    # # in case the full training size did not get into the range from above
    # if len(train_data) % 1000 != 0:
    #     varied_training_data.append(train_data)
    return varied_training_data


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--train_data", type=str, help="jsonl file",
                       default="data/winohard/transformed/transformation/train_xl.jsonl")
    parse.add_argument("--dev_data", type=str, help="jsonl file",
                       default="data/winohard/transformed/transformation/dev.jsonl")
    parse.add_argument("--debiased_train", type=str, help="jsonl file",
                       default="data/winogrande/train-debiased.jsonl")
    parse.add_argument("--out_folder", type=str, help="path of folder",
                       default="data/winohard/new_splits/")
    parse.add_argument("--model_names", type=str, help="model type (out of MLM from huggingface)",
                       default="roberta-large,roberta-base,bert-base-cased,bert-large-cased,albert-base-v2,albert-xxlarge-v2")

    args = parse.parse_args()

    train_data = read_jsonl(args.train_data)
    dev_data = read_jsonl(args.dev_data)
    debiased_train_data = read_jsonl(args.debiased_train)

    model_names = args.model_names.split(',')
    tokenizers = get_tokenizers(model_names)

    filter_train = filter_data(train_data, tokenizers, model_names)
    filter_dev = filter_data(dev_data, tokenizers, model_names)

    debiased_ids = set([x['qID'] for x in debiased_train_data])
    new_train, new_test = create_test_set(filter_train, debiased_ids)

    varied_training_sizes = create_train_sizes(new_train)

    write_jsonl(filter_dev, args.out_folder + 'dev.jsonl')
    write_jsonl(new_test, args.out_folder + 'test.jsonl')

    for ind, train in enumerate(varied_training_sizes):
        write_jsonl(train, args.out_folder + 'train_{}.jsonl'.format(ind))


if __name__ == '__main__':
    main()
