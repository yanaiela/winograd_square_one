import argparse
from typing import Dict

import torch
import wandb
from tqdm import tqdm
from transformers import pipeline, Pipeline

from winohard.filter2single_tokens_lms import filter_oov
from winohard.utils import read_jsonl


def log_wandb(args):
    lm = args.model
    data = args.in_data.split('/')[-1].split('_lm')[0]
    config = dict(
        lm=lm,
        data=data
    )

    wandb.init(
        name=f'zero-shot_{lm}_{data}',
        project="winohard",
        tags=["eval", 'zero-shot', lm],
        config=config,
    )


def build_model_by_name(lm: str, device) -> Pipeline:
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """

    if not torch.cuda.is_available():
        device = -1

    model = pipeline("fill-mask", model=lm, device=device)
    return model


def fill_mask(model, sentence, candidates) -> Dict:
    words = model(sentence.replace('[MASK]', model.tokenizer.mask_token), targets=candidates)

    ans_dic = {}
    for w in words:
        ans_dic[w['token_str']] = w['score']
    return ans_dic


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

    acc = 0
    double_acc = 0
    for example in tqdm(filt_data):
        s1 = example['s1']
        s2 = example['s2']
        ans1 = example['ans1']
        ans2 = example['ans2']
        if 'roberta' in args.model or 'albert' in args.model or 'longformer' in args.model:
            ans1 = ' ' + ans1
            ans2 = ' ' + ans2

        result1 = fill_mask(lm_model, s1, [ans1, ans2])
        result2 = fill_mask(lm_model, s2, [ans2, ans1])
        if 'roberta' in args.model or 'albert' in args.model or 'longformer' in args.model:
            ans1 = lm_model.tokenizer.tokenize(ans1)[0]
            ans2 = lm_model.tokenizer.tokenize(ans2)[0]

        if result1[ans1] > result1[ans2]:
            acc += 1

        if result2[ans2] > result2[ans1]:
            acc += 1

        if result1[ans1] > result1[ans2] and result2[ans2] > result2[ans1]:
            double_acc += 1

    print('reg acc:', acc / (len(filt_data) * 2))
    print('double acc:', double_acc / len(filt_data))
    wandb.run.summary['reg_acc'] = acc / (len(filt_data) * 2)
    wandb.run.summary['pair_acc'] = double_acc / len(filt_data)


if __name__ == '__main__':
    main()
