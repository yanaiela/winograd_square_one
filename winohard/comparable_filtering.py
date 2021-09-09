"""

"""

import argparse

from winohard.utils import read_jsonl, write_jsonl


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--original_data", type=str, help="jsonl file",
                       default="data/winohard/pairs/dev.jsonl")
    parse.add_argument("--subset_data", type=str, help="jsonl file",
                       default="data/winohard/transformed/transformation/dev.jsonl")
    parse.add_argument("--out_data", type=str, help="jsonl file",
                       default="data/winohard/transformed/original/dev_pairs.jsonl")

    args = parse.parse_args()

    data = read_jsonl(args.original_data)
    subset_data = read_jsonl(args.subset_data)

    subset_ids = set([x['qID'] for x in subset_data])
    out_data = [x for x in data if x['qID'] in subset_ids]

    print('collected {} examples, out of {} total original examples.'
          ' that makes it {}%'.format(len(out_data), len(data),
                                      len(out_data) / len(
                                          data)))

    write_jsonl(out_data, args.out_data)


if __name__ == '__main__':
    main()
