for data in 'train_xs' 'train_s' 'train_m' 'train_l' 'train_xl' 'dev' 'wsc' 'wsc_non_associative'
do
  python winohard/masked_word_transformation.py \
         --paired_data data/winohard/pairs/$data.jsonl \
         --out_lm_file data/winohard/transformed/zero_shot/$data.jsonl \
         --out_wino_file data/winohard/transformed/transformation/$data.jsonl
done