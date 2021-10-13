import ujson as json


""" Utility for finetuning BERT/RoBERTa models on WinoGrande. """

# from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import json
import random

logger = logging.getLogger(__name__)


def read_jsonl(in_f):
    data = []
    with open(in_f, 'r') as f:
        for line in f.readlines():
            obj = json.loads(line)
            data.append(obj)
    return data


def write_jsonl(data, out_f):
    with open(out_f, 'w') as f:
        for obj in data:
            json.dump(obj, f)
            f.write('\n')

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class MCInputExample(object):
    def __init__(self, guid, options, label, pair_id):
        self.guid = guid
        self.options = options
        self.label = label
        self.pair_id = pair_id


class MLMInputExample(object):
    def __init__(self, guid, options, label, pair_id):
        self.guid = guid
        self.options = options
        self.label = label
        self.pair_id = pair_id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, pair_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.pair_id = pair_id


class MultipleChoiceFeatures(object):
    def __init__(self,
                 example_id,
                 option_features,
                 label=None,
                 pair_id=-1):
        self.example_id = example_id
        self.option_features = self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in option_features
        ]
        self.label = int(label)
        self.pair_id = pair_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, num_train_examples):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a tab separated value file."""
        records = []
        with open(input_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                records.append(json.loads(line))
            return records


class WinograndeProcessor(DataProcessor):

    def get_train_examples(self, data_dir, num_train_examples):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")))

    def get_examples(self, data_file, num_example):

        if num_example <= 0:
            return self._create_examples(self._read_jsonl(data_file))
        else:
            raw_data = self._read_jsonl(data_file)
            random.shuffle(raw_data)
            return self._create_examples(raw_data[:num_example])

    def get_labels(self):
        return ["1", "2"]

    def _create_examples(self, records):
        examples = []
        for (i, record) in enumerate(records):
            guid = record['qID']
            sentence = record['sentence']

            name1 = record['option1']
            name2 = record['option2']
            if not 'answer' in record:
                # This is a dummy label for test prediction.
                # test.jsonl doesn't include the `answer`.
                label = "1"
            else:
                label = record['answer']

            if not 'pair_id' in record:
                pair_id = -1
            else:
                pair_id = record['pair_id']

            conj = "_"
            idx = sentence.index(conj)
            context = sentence[:idx]
            option_str = "_ " + sentence[idx + len(conj):].strip()

            option1 = option_str.replace("_", name1)
            option2 = option_str.replace("_", name2)

            mc_example = MCInputExample(
                guid=guid,
                options=[
                    {
                        'segment1': context,
                        'segment2': option1
                    },
                    {
                        'segment1': context,
                        'segment2': option2
                    }
                ],
                label=label,
                pair_id=pair_id
            )
            examples.append(mc_example)

        return examples


class SuperGlueWscProcessor(DataProcessor):
    """Processor for the SuperGLUE-WSC """

    def get_train_examples(self, data_dir, num_train_examples):
        if num_train_examples <= 0:
            return self._create_examples(
                self._read_jsonl(os.path.join(data_dir, "train.jsonl")))
        else:
            return self._create_examples(
                self._read_jsonl(os.path.join(data_dir, "train.jsonl"))[:num_train_examples])

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")))

    def get_examples(self, data_file):
        return self._create_examples(self._read_jsonl(data_file))

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, records):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, record) in enumerate(records):
            guid = record['idx']
            text_a = record["sentence1"]
            text_b = record["sentence2"]
            label = record["label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MLMProcessor(DataProcessor):

    def get_train_examples(self, data_dir, num_train_examples):
        if num_train_examples <= 0:
            return self._create_examples(
                self._read_jsonl(os.path.join(data_dir, "train.jsonl")))
        else:
            return self._create_examples(
                self._read_jsonl(os.path.join(data_dir, "train.jsonl"))[:num_train_examples])

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")))

    def get_examples(self, data_file):
        return self._create_examples(self._read_jsonl(data_file))

    def get_labels(self):
        return ["1", "2"]

    def _create_examples(self, records):
        examples = []
        for (i, record) in enumerate(records):
            guid = record['qID']
            sentence = record['sentence']

            name1 = record['option1']
            name2 = record['option2']
            if not 'answer' in record:
                # This is a dummy label for test prediction.
                # test.jsonl doesn't include the `answer`.
                label = "1"
            else:
                label = record['answer']

            if not 'pair_id' in record:
                pair_id = -1
            else:
                pair_id = record['pair_id']

            conj = "_"
            idx = sentence.index(conj)
            context = sentence[:idx]
            option_str = "_ " + sentence[idx + len(conj):].strip()

            option1 = option_str.replace("_", name1)
            option2 = option_str.replace("_", name2)

            mc_example = MCInputExample(
                guid=guid,
                options=[
                    {
                        'segment1': context,
                        'segment2': option1
                    },
                    {
                        'segment1': context,
                        'segment2': option2
                    }
                ],
                label=label,
                pair_id=pair_id
            )
            examples.append(mc_example)

        return examples

def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 output_mode,
                                 cls_token_at_end=False,
                                 pad_on_left=False,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_token=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=1,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        special_tokens_count = 3 if sep_token_extra else 2
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"

            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count - 1)
        else:
            # Account for [CLS] and [SEP] with "- 2" or "-3" for RoBERTa
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          pair_id=example.pair_id))
    return features


def convert_multiple_choice_examples_to_features(examples, label_list, max_seq_length,
                                                 tokenizer, output_mode,
                                                 cls_token_at_end=False, pad_on_left=False,
                                                 cls_token='[CLS]', sep_token='[SEP]', sep_token_extra=False, pad_token=0,
                                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        option_features = []
        for option in example.options:

            context_tokens = tokenizer.tokenize(option['segment1'])

            option_tokens = tokenizer.tokenize(option['segment2'])
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(context_tokens, option_tokens, max_seq_length - special_tokens_count)

            tokens = context_tokens + [sep_token]

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)
            tokens += option_tokens + [sep_token]

            segment_ids += [sequence_b_segment_id] * (len(option_tokens) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if output_mode != "multiple_choice":
                raise KeyError(output_mode)

            option_features.append((tokens, input_ids, input_mask, segment_ids))

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"example_id: {example.guid}")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(
                    option_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            logger.info(f"label: {label_id}")
            logger.info(f"label: {example.pair_id}")

        features.append(
            MultipleChoiceFeatures(
                example_id=example.guid,
                option_features=option_features,
                label=label_id,
                pair_id=example.pair_id
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "winogrande":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "superglue-wsc":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

processors = {
    "winogrande": WinograndeProcessor,
    "superglue-wsc": SuperGlueWscProcessor,
    "MLM": MLMProcessor
}

output_modes = {
    "winogrande": "multiple_choice",
    "superglue-wsc": "classification",
    "MLM": "MLM"
}

GLUE_TASKS_NUM_LABELS = {
    "winogrande": 2,
    "superglue-wsc": 2,
}
