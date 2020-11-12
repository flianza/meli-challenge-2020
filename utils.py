import gzip
import json

def jl_to_list(filename):
    output = []
    with gzip.open(filename, 'rb') as f:
        for line in f:
            output.append(json.loads(line))
    return output

def drop_duplicates(items):
    seen = set()
    seen_add = seen.add
    return [x for x in items if not (x in seen or seen_add(x))]
