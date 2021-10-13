""" Finetuning BERT/RoBERTa models on WinoGrande. """

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random
import math

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import pathlib
import statistics

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertForMultipleChoice,
                          BertTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaForMultipleChoice,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer, AlbertConfig, AlbertForMultipleChoice, AlbertTokenizer)

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule

# from scripts.adhoc_models import RobertaForMultipleChoice
# from scripts.roberta_mc import RobertaForMultipleChoice
# from scripts.utils import (compute_metrics, convert_examples_to_features,
#                                 output_modes, processors,
#                                 convert_multiple_choice_examples_to_features)
# from roberta_mc import RobertaForMultipleChoice
from Fine_tuned_model_evaluation.utils import (compute_metrics, convert_examples_to_features,
                                               output_modes, processors,
                                               convert_multiple_choice_examples_to_features)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'bert_mc': (BertConfig, BertForMultipleChoice, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'roberta_mc': (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),
    'albert_mc': (AlbertConfig, AlbertForMultipleChoice, AlbertTokenizer)
}


def winohard_evaluation(predictions, labels, pair_ids):
    print('prediction:', predictions, len(predictions))
    print('labels:', labels, len(labels))
    print('pair_ids:', pair_ids, len(pair_ids))

    all_correctness = list()
    correctness_by_pair = dict()
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            tmp_correctness = 1
        else:
            tmp_correctness = 0
        all_correctness.append(tmp_correctness)
        if pair_ids[i] != -1:
            if str(pair_ids[i]) not in correctness_by_pair:
                correctness_by_pair[str(pair_ids[i])] = list()
            correctness_by_pair[str(pair_ids[i])].append(tmp_correctness)
    print(correctness_by_pair)
    print('Accuracy by question:', sum(all_correctness) / len(all_correctness))
    if len(correctness_by_pair) == 0:
        print('We do not have pair labels for this dataset')
    else:
        pair_correctness = list()
        for tmp_pair in correctness_by_pair:
            if sum(correctness_by_pair[tmp_pair]) == len(correctness_by_pair[tmp_pair]):
                pair_correctness.append(1)
            else:
                pair_correctness.append(0)
        print('Accuracy by pair:', sum(pair_correctness) / len(pair_correctness))

    return sum(all_correctness) / len(all_correctness), sum(pair_correctness) / len(pair_correctness)


def set_seed(args):
    if args.seed == -1:
        seed = random.randint(1, 10000)
    else:
        seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    # Added here for reproductibility (even between python 2 and 3)
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.output_dir)

    processor = processors[args.task_name]()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_pct is None:
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=math.floor(args.warmup_pct*t_total), t_total=t_total)
        scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=math.floor(args.warmup_pct * t_total),
                                         num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    # set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0],
                              mininterval=10, ncols=100)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet', 'bert_mc'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, processor, eval_split="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, processor, prefix="", eval_split=None):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    assert eval_split is not None

    results = {}
    if os.path.exists("/output/metrics.json"):
        with open("/output/metrics.json", "r") as f:
            existing_results = json.loads(f.read())
        f.close()
        results.update(existing_results)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, eval_split=eval_split)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} on {} *****".format(prefix, eval_split))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        pair_ids = None
        score_list = []

        for batch in tqdm(eval_dataloader, desc="Evaluating", mininterval=10, ncols=100):
            model.eval()

            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': None,  # XLM don't use segment_ids
                          'labels': batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            written_scores = logits.to('cpu').numpy()
            score_list.extend(written_scores.tolist())

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                # if not eval_split == "test":
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                pair_ids = batch[4].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                # if not eval_split == "test":
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                pair_ids = np.append(pair_ids, batch[4].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        # print(preds)
        if args.output_mode == "classification" or args.output_mode == "multiple_choice":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        acc_question, acc_pair = winohard_evaluation(preds, out_label_ids, pair_ids)
        results['acc_by_question'] = acc_question
        results['acc_by_pair'] = acc_pair

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, eval_split="train"):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if args.data_cache_dir is None:
        data_cache_dir = args.data_dir
    else:
        data_cache_dir = args.data_cache_dir

    cached_features_file = os.path.join(data_cache_dir, 'cached_{}_{}_{}_{}'.format(
        eval_split,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.train_data_file)
        label_list = processor.get_labels()

        if eval_split == "train":
            examples = processor.get_examples(args.train_data_file, args.num_train_examples)
        elif eval_split == "dev":
            examples = processor.get_examples(args.dev_data_file, -1)
        elif eval_split == "test":
            examples = processor.get_examples(args.test_data_file, -1)
        else:
            raise Exception("eval_split should be among train / dev / test")

        if output_mode == "multiple_choice":
            features = convert_multiple_choice_examples_to_features(
                examples, label_list, args.max_seq_length, tokenizer, output_mode,
                cls_token_at_end=bool(args.model_type in ['xlnet']),  # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args.model_type in ['roberta', "roberta_mc"]),
                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(args.model_type in ['xlnet']),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0
            )
        else:
            features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args.model_type in ['roberta', "roberta_mc"]),
                                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    if output_mode == "multiple_choice":
        def _select_field(features, field):
            return [
                [
                    choice[field]
                    for choice in feature.choices_features
                ]
                for feature in features
            ]

        all_input_ids = torch.tensor(_select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(_select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(_select_field(features, 'segment_ids'), dtype=torch.long)
        all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
        all_pair_ids = torch.tensor([f.pair_id for f in features], dtype=torch.long)
    else:
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        all_pair_ids = torch.tensor([f.pair_id for f in features], dtype=torch.long)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            if args.data_cache_dir is not None:
                pathlib.Path(args.data_cache_dir).mkdir(parents=True, exist_ok=True)
            torch.save(features, cached_features_file)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_pair_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--dev_data_file", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--test_data_file", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut names")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    # parser.add_argument("--data_dir", default=None, type=str, required=True,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--data_cache_dir", default=None, type=str,
                        help="Cache dir if it needs to be diff from data_dir")

    parser.add_argument("--gpu", default=5, type=int,
                        help="GPU to run the experiments with")

    ## Other parameters
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=80, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_prediction", action='store_true',
                        help="Whether to run prediction on the test set. (Training will not be executed.)")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--run_on_test', action='store_true')

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_examples", default=-1, type=int,
                        help="Total number of training examples.")
    parser.add_argument("--num_trail", default=1, type=int,
                        help="Total number of training trails.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_pct", default=0.1, type=float,
                        help="Linear warmup over warmup_pct*total_steps.")

    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=10000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=11262,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd
    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
    # # print(args.gpu)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.n_gpu=1
    # args.device = device
    # print('current device:', device)
    # print(device)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(
    #     args.config_name if args.config_name else args.model_name_or_path,
    #     num_labels=1 if args.model_type in ["roberta_mc"] else num_labels,
    #     finetuning_task=args.task_name
    # )
    # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    # model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=True)

    config = config_class.from_pretrained(args.model_name_or_path, num_labels=1, finetuning_task="winogrande")
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Prediction (on test set)
    if args.do_prediction:
        results = {}
        logger.info("Prediction on the test set (note: Training will not be executed.) ")
        evaluate(args, model, tokenizer, processor, prefix="", eval_split="test")
        # result = dict((k, v) for k, v in result.items())
        # results.update(result)
        logger.info("***** Experiment finished *****")
        # return results

    # Training
    performance_by_question = list()
    performance_by_pair = list()

    for i in range(args.num_trail):
        if args.do_train:
            set_seed(args)
            train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            model.to(args.device)

        result = evaluate(args, model, tokenizer, processor, prefix="0", eval_split="dev")
        performance_by_question.append(result['acc_by_question'])
        performance_by_pair.append(result['acc_by_pair'])

    performance_by_question_mean = statistics.mean(performance_by_question)
    performance_by_question_std = statistics.stdev(performance_by_question)
    performance_by_pair_mean = statistics.mean(performance_by_pair)
    performance_by_pair_std = statistics.stdev(performance_by_pair)

    result_log = {'performance_by_question_mean': performance_by_question_mean,
                  'performance_by_question_std': performance_by_question_std,
                  'performance_by_pair_mean': performance_by_pair_mean,
                  'performance_by_pair_std': performance_by_pair_std}
    print(result_log)

    # # Evaluation
    # results = {}
    # checkpoints = [args.output_dir]
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         result = evaluate(args, model, tokenizer, processor, prefix=global_step, eval_split="dev")
    #         result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         results.update(result)
    #
    # # Run on test
    # if args.run_on_test and args.local_rank in [-1, 0]:
    #     checkpoint = checkpoints[0]
    #     global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #     model = model_class.from_pretrained(checkpoint)
    #     model.to(args.device)
    #     result = evaluate(args, model, tokenizer, processor, prefix=global_step, eval_split="test")
    #     result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #     results.update(result)

    logger.info("***** Experiment finished *****")
    # return results


if __name__ == "__main__":
    main()
