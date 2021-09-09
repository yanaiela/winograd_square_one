import argparse
from typing import Dict
from collections import defaultdict
import torch
import wandb
from tqdm import tqdm
from transformers import pipeline, Pipeline

from winohard.filter2single_tokens_lms import filter_oov
from winohard.utils import read_jsonl
from winohard.eval.zero_shot_lm import build_model_by_name, fill_mask


def log_wandb(args):
    lm = args.model
    data = args.in_data.split('/')[-1].split('_')[0]
    config = dict(
        lm=lm,
        data=data
    )

    wandb.init(
        name=f'zero-shot-v2_{lm}_{data}',
        project="winohard",
        tags=["eval", 'zero-shot', lm],
        config=config,
    )


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--in_data", type=str, help="jsonl file",
                       default="data/winohard/lms/wsc_lm.jsonl")
    parse.add_argument("--model", type=str, help="model type (out of MLM from huggingface)",
                       default="roberta-base")
    parse.add_argument("--device", type=int, help="device to run on",
                       default=-1)

    args = parse.parse_args()

    log_wandb(args)

    print('model: {}. dataset: {}'.format(args.model, args.in_data.split('/')[-1].split('.')[0]))

    data = read_jsonl(args.in_data)

    lm_model = build_model_by_name(args.model, args.device)

    filt_data = filter_oov(data, lm_model.tokenizer, args.model)

    paired_acc = defaultdict(int)
    acc = 0
    double_acc = 0
    for example in tqdm(filt_data):
        sentence = example['sentence']
        ans1 = example['option1']
        ans2 = example['option2']
        if 'roberta' in args.model or 'albert' in args.model:
            ans1 = ' ' + ans1
            ans2 = ' ' + ans2

        sentence = sentence.replace('_', '[MASK]')

        result = fill_mask(lm_model, sentence, [ans1, ans2])
        if 'roberta' in args.model or 'albert' in args.model:
            ans1 = lm_model.tokenizer.tokenize(ans1)[0]
            ans2 = lm_model.tokenizer.tokenize(ans2)[0]

        true_answer = example['answer']

        is_correct = 0
        if result[ans1] > result[ans2] and true_answer == '1':
            is_correct = 1
        elif result[ans2] > result[ans1] and true_answer == '2':
            is_correct = 1

        paired_acc[example['pair_id']] += is_correct

    reg_acc = sum(list(paired_acc.values())) / len(filt_data)
    pair_acc = len([x for x in list(paired_acc.values()) if x == 2]) / len(paired_acc)
    print('reg acc:', reg_acc)
    print('double acc:', pair_acc)
    wandb.run.summary['reg_acc'] = reg_acc
    wandb.run.summary['pair_acc'] = pair_acc


if __name__ == '__main__':
    main()
