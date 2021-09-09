# Winograd Schema: Back to Square One

This repository accompanies the paper: [Back to Square One: Artifact Detection, Training and Commonsense Disentanglement in the Winograd Schema](https://arxiv.org/abs/2104.08161)
which was published at EMNLP2021

## Data
We publish all the transformations we conducted with the proposed baselines, the twin pairing,
etc, under  [data](data/)

## Reproducing the Results

### Setup
First, run the coupling script. 
This script couples pairs from WSC and winogrande, for all splits of the data

`bash runs/couple.sh`


To filter out associative examples, and produce the wsc subset
to match the `WSC-na` from the paper:

```shell
python winohard/associate_associative.py \
        --wsc data/winohard/pairs/wsc.jsonl \
        --wsc_associative data/associatives/WSC_associative_label.json \
        --out_data data/winohard/pairs/wsc_non_associative.jsonl
```

### Baselines data creation

We also published all the transformed [data](data/), in case you just want to have a look,
or directly evaluate on it.

#### *no-cands* baseline:

`bash runs/no_cands.sh`

#### *part-sent* baseline:

`bash runs/truncate.sh`


### Transforming data to be compatible with zero-shot evaluation

`bash runs/zero_shot.sh`

once creating the data, filtering examples that have a shared vocabulary
across the different tested models

`bash runs/unify_zero_shot.sh`

In order to make a fair comparison with the transformed model,
filtering the same subset that is used for the
transformation task, for the original task
`bash runs/comparable_original_2_new_format.sh`

### Evaluation
Evaluate against dataset and models:

`bash runs/eval_zero_shot.sh`


