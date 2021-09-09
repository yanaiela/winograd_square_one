for data in 'train_xs' 'train_s' 'train_m' 'train_l' 'train_xl' 'dev' 'wsc' 'wsc_non_associative'
do
  python winohard/filter2single_tokens_lms.py \
         --in_data data/winohard/transformed/zero_shot/$data.jsonl \
         --model_names roberta-large,roberta-base,bert-base-cased,bert-large-cased,albert-base-v2,albert-xxlarge-v2 \
         --out_file data/winohard/transformed/zero_shot/${data}_lms.jsonl
done