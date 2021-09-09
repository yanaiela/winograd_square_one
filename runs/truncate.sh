for data in 'train_xs' 'train_s' 'train_m' 'train_l' 'train_xl' 'dev' 'wsc' 'wsc_non_associative'
do
  python winohard/baselines/truncate.py \
        --in_data data/winohard/pairs/$data.jsonl \
        --out_data data/winohard/truncated/$data.jsonl
done