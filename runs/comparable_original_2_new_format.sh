for data in 'train_xs' 'train_s' 'train_m' 'train_l' 'train_xl' 'dev' 'wsc' 'wsc_non_associative'
do
  python winohard/comparable_filtering.py \
        --original_data data/winohard/pairs/$data.jsonl \
        --subset_data data/winohard/transformed/transformation/$data.jsonl \
        --out_data data/winohard/transformed/original/$data.jsonl
done