from sklearn.model_selection import train_test_split
from utils import jl_to_list

samples = None
rows = jl_to_list('data/train_dataset.jl.gz')
if samples is not None:
    rows = rows[:samples]
rows_train, rows_test = train_test_split(rows, test_size=0.2, random_state=42)

item_data = jl_to_list('data/item_data.jl.gz')

test_dataset = []
if samples is None:
    test_dataset = jl_to_list('data/test_dataset.jl.gz')

from tqdm import tqdm

metadata = {x['item_id']:x for x in tqdm(item_data)}
all_items = list(metadata.keys())
y_true = [row['item_bought'] for row in tqdm(rows_test)]
y_full_true = [row['item_bought'] for row in tqdm(rows)]

sold_items_ids = {row['item_bought'] for row in tqdm(rows)}
sold_prices = [float(metadata[item_id]['price'] if metadata[item_id]['price'] is not None else 99999) for item_id in tqdm(sold_items_ids)]

from models import TopViewedItemsByMostFrequentDomainBaseline, LastViewedBaseline
from order_models import PriceBasedOrder

fill_model = TopViewedItemsByMostFrequentDomainBaseline(all_items, metadata, verbose=False)
baseline = LastViewedBaseline(all_items, fill_model=fill_model)
order = PriceBasedOrder(metadata, sold_prices)

baseline.fit(rows)
y_pred = baseline.predict(test_dataset)
y_pred_ordered = order.predict(y_pred)

import pandas as pd
df = pd.DataFrame(data=y_pred)
df.to_csv("./results/not_ordered.csv",sep=',',index=False,header=False)
df = pd.DataFrame(data=y_pred_ordered)
df.to_csv("./results/ordered.csv",sep=',',index=False,header=False)