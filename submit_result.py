from utils import jl_to_list
from models import TopViewedItemsByMostFrequentDomainBaseline, LastViewedBaseline
import pandas as pd

samples = None
train_dataset = jl_to_list('data/train_dataset.jl.gz')
item_data = jl_to_list('data/item_data.jl.gz')
metadata = {x['item_id']:x for x in item_data}
all_items = list(metadata.keys())
test_dataset = jl_to_list('data/test_dataset.jl.gz')

fill_model = TopViewedItemsByMostFrequentDomainBaseline(all_items, metadata, verbose=False)
baseline = LastViewedBaseline(all_items, fill_model=fill_model)
baseline.fit(train_dataset)
y_pred = baseline.predict(test_dataset)

df = pd.DataFrame(data=y_pred)
df.to_csv("./results/fill_model.csv",sep=',',index=False,header=False)