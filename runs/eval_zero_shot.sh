for model in bert-base-cased bert-large-cased roberta-base roberta-large albert-base-v1 albert-xxlarge-v2
do
  for data in 'dev' 'wsc' 'wsc_non_associative'
  do
    python winohard/eval/zero_shot_lm.py \
           --model $model \
           --in_data data/winohard/transformed/zero_shot/${data}_lms.jsonl \
           --device -1
  done
done