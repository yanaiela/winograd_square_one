for data in 'train_xs' 'train_s' 'train_m' 'train_l' 'train_xl' 'dev' 'wsc' 'wsc_non_associative'
do
  python winohard/baselines/remove_candidates.py \
        --in_data data/winohard/pairs/$data.jsonl \
        --out_data data/winohard/masked_cands/$data.jsonl \
        --masked
done

for data in 'train_xs' 'train_s' 'train_m' 'train_l' 'train_xl' 'dev' 'wsc' 'wsc_non_associative'
do
  python winohard/baselines/remove_candidates.py \
        --in_data data/winohard/pairs/$data.jsonl \
        --out_data data/winohard/no_cands/$data.jsonl
done