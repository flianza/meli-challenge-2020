import gzip
import json
import pandas as pd
from datetime import datetime

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

def export_results(results):
    df = pd.DataFrame(data=results)
    filename = datetime.now().strftime('%Y%m%d%H%M') + '.csv'
    df.to_csv('./results/' + filename, sep=',', index=False, header=False)
