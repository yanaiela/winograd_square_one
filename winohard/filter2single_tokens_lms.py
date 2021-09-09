import argparse
from typing import List, Dict
from copy import deepcopy

from transformers import AutoTokenizer

from winohard.utils import read_jsonl, write_jsonl


def get_tokenizer_by_name(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_tokenizers(model_names: List[str]):
    tokenizers = [get_tokenizer_by_name(x) for x in model_names]
    return tokenizers


def filter_oov(data, tokenizer, model_name):
    if 'roberta' in model_name or 'albert' in model_name or 'longformer' in model_name:
        all_vocab = list(tokenizer.get_vocab().keys())
        vocab = [tokenizer.convert_tokens_to_string(x).strip() for x in all_vocab]
    else:
        vocab = list(tokenizer.vocab.keys())

    filt_data = []

    for row in data:
        if 'option1' in row:
            ans1 = row['option1']
            ans2 = row['option2']
        else:
            ans1 = row['ans1']
            ans2 = row['ans2']


        if 'roberta' in model_name or 'albert' in model_name or 'longformer' in model_name:
            ans1 = ' ' + ans1
            ans2 = ' ' + ans2
        if len(tokenizer.tokenize(ans1)) > 1 or len(tokenizer.tokenize(ans2)) > 1:
        #if ans1 not in vocab or ans2 not in vocab:
            print(ans1, ans2)
            continue
        filt_data.append(row)
    return filt_data


def filter_data(origin_data: List[Dict], tokenizers, model_names) -> List[Dict]:
    filter_data = deepcopy(origin_data)

    for tokenizer_name, tokenizer in zip(model_names, tokenizers):
        before = len(filter_data)
        filter_data = filter_oov(filter_data, tokenizer, tokenizer_name)
        after = len(filter_data)
        print(f'{tokenizer_name} filtered out {before - after} examples')
    return filter_data


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--in_data", type=str, help="jsonl file",
                       default="data/winohard/lms/wsc_lm.jsonl")
    parse.add_argument("--model_names", type=str, help="model type (out of MLM from huggingface)",
                       default="roberta-base,bert-base-cased")
    parse.add_argument("--out_file", type=str, help="output jsonl file path",
                       default="data/winohard/transformation/wsc_transformed_filtered.jsonl")

    args = parse.parse_args()

    data = read_jsonl(args.in_data)

    model_names = args.model_names.split(',')
    tokenizers = get_tokenizers(model_names)

    data = filter_data(data, tokenizers, model_names)

    write_jsonl(data, args.out_file)


if __name__ == '__main__':
    main()
