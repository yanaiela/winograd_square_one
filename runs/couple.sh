mkdir -p data/winohard/pairs/

# WSC
python winohard/couple_examples.py \
       --in_data data/wsc/wsc.jsonl \
       --out_data data/winohard/pairs/wsc.jsonl \
       --wsc

# Winogrande
for data in 'train_xs' 'train_s' 'train_m' 'train_l' 'train_xl' 'dev' 'test'
do
  python winohard/couple_examples.py \
         --in_data data/winogrande/$data.jsonl \
         --out_data data/winohard/pairs/$data.jsonl
done
