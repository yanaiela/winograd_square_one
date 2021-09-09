"""

"""

import argparse

from winohard.utils import read_jsonl, write_jsonl, read_json


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--wsc", type=str, help="jsonl file",
                       default="data/winohard/pairs/wsc.jsonl")
    parse.add_argument("--wsc_associative", type=str, help="json file",
                       default="data/associatives/WSC_associative_label.json")
    parse.add_argument("--out_data", type=str, help="json file",
                       default="data/winohard/pairs/wsc_non_associative.jsonl")

    args = parse.parse_args()

    wsc = read_jsonl(args.wsc)
    wsc_associative_labels = read_json(args.wsc_associative)

    associatives = [x['index'] for x in wsc_associative_labels if x['is_associative']]
    non_associative_wsc = []
    for i in range(0, len(wsc) - 1, 2):
        qid1 = int(wsc[i]['qID'][3:])
        qid2 = int(wsc[i + 1]['qID'][3:])
        if qid1 in associatives or qid2 in associatives:
            continue
        non_associative_wsc.append(wsc[i])
        non_associative_wsc.append(wsc[i + 1])

    print('collected {} examples, out of {} total original examples.'
          ' that makes it {}%'.format(len(non_associative_wsc), len(wsc),
                                      len(non_associative_wsc) / len(
                                          wsc)))

    write_jsonl(non_associative_wsc, args.out_data)


if __name__ == '__main__':
    main()
