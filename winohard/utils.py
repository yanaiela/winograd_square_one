import json


def read_json(in_f):
    with open(in_f, 'r') as f:
        data = json.load(f)
    return data


def read_jsonl(in_f):
    data = []
    with open(in_f, 'r') as f:
        for line in f.readlines():
            obj = json.loads(line)
            data.append(obj)
    return data


def write_jsonl(data, out_f):
    with open(out_f, 'w') as f:
        for obj in data:
            json.dump(obj, f)
            f.write('\n')
