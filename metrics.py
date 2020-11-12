
import statistics
import numpy as np
from utils import drop_duplicates

def ndcg(y_pred, y_true, metadata):
    ndcg_values = []
    
    y_pred_cleaned = clean(y_pred)
    
    for i in range(len(y_pred_cleaned)):
        dcg_value = dcg(y_pred_cleaned[i], y_true[i], metadata)
        idcg_value = idcg(y_pred_cleaned[i], y_true[i])
        ndcg_values.append(dcg_value / idcg_value)
    
    return statistics.mean(ndcg_values)

def clean(y_pred):
    results = []
    
    for items in y_pred:
        items_clean = drop_duplicates(items)
        missing_values = 10 - len(items_clean)
        items_clean = items_clean + [0] * missing_values
        results.append(items_clean)
    
    return results
        
def dcg(y_pred, y_true, metadata):
    values = []
    for i in range(len(y_pred)):
        rel_value = rel(y_pred[i], y_true, metadata)
        dcg_value = rel_value / np.log(i + 2.0)
        values.append(dcg_value)
    return np.sum(values)
    
def idcg(y_pred, y_true):
    return 22.42461597

def rel(y_hat, y, metadata):
    if y_hat == y:
        return 12.0
    if domain(y_hat, metadata) == domain(y, metadata):
        return 1.0
    return 0.0

def domain(item, metadata):
    return metadata[item]['domain_id']
