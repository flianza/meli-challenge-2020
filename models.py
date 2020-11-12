import math
import random
from collections import Counter, defaultdict
from tqdm.notebook import tqdm
from utils import drop_duplicates

class AbstractBaseline(object):
    def __init__(self, all_items, fill_model=None, k=10, verbose=True):
        self.all_items = all_items
        self.fill_model = fill_model
        self.k = k
        self.verbose = verbose
    
    def fit(self, X=None, y=None):
        self._fit(X, y)
        
        if self.fill_model is not None:
            self.fill_model.fit(X, y)
            
        return self
    
    def _fit(self, X, y):
        pass
    
    def predict(self, X=None):
        y_pred = []
        
        for row in (tqdm(X) if self.verbose else X):
            recommendation = self._predict_one(row)
            recommendation = self._fill_missing_values(row, recommendation)
            y_pred.append(recommendation)
        return y_pred
    
    def _predict_one(self, row):
        pass
    
    def _fill_missing_values(self, row, recommendation):
        recommendation = drop_duplicates(recommendation)[:10]
        
        missing_items = self.k - len(recommendation)
        fill_items = []
        if self.fill_model is None:
            fill_items = random.choices(self.all_items, k=missing_items)
        else:
            fill_items = self.fill_model.predict([row])[0][:missing_items]
        
        return recommendation + fill_items

class LastViewedBaseline(AbstractBaseline):
    def __init__(self, all_items, fill_model=None, k=10, verbose=True):
        super().__init__(all_items, fill_model=fill_model, k=k, verbose=verbose)
    
    def _predict_one(self, row):
        viewed = [ev for ev in row['user_history'] if ev['event_type'] == 'view']
        viewed = sorted(viewed, key=lambda x: x['event_timestamp'], reverse=True)
        viewed = [ev['event_info'] for ev in viewed]

        recommendation = []

        for item in viewed:
            if item not in recommendation:
                recommendation.append(item)
        
        return recommendation

class TopViewedItemsByMostFrequentDomainBaseline(AbstractBaseline):
    def __init__(self, all_items, metadata, fill_model=None, k=10, max_views=30, verbose=True):
        super().__init__(all_items, fill_model=fill_model, k=k, verbose=verbose)
        self.max_views = max_views
        self.metadata = metadata
        self.viewed_items_by_domain = None
    
    def _fit(self, X=None, y=None):
        self.viewed_items_by_domain = defaultdict(lambda: defaultdict(int))

        for row in tqdm(X):
            viewed = [ev['event_info'] for ev in row['user_history'] if ev['event_type'] == 'view']
            for item in viewed:
                domain = self.metadata[item]['domain_id']
                self.viewed_items_by_domain[domain][item] += 1
                
        return self
    
    def _predict_one(self, row):
        viewed = [ev['event_info'] for ev in row['user_history'] if ev['event_type'] == 'view']
        if len(viewed) == 0:
            return []
        domain = self.__visited_domains(row)
        domain = domain.most_common(1)[0][0]
        return self.__top_items(domain)
    
    def __visited_domains(self, row):
        domains = Counter()
        viewed = [ev['event_info'] for ev in row['user_history'] if ev['event_type'] == 'view']
        if len(viewed) > self.max_views:
            viewed = viewed[:self.max_views]
        for item in viewed:
            domain = self.metadata[item]['domain_id']
            domains[domain] += 1
        return domains
    
    def __top_items(self, domain):
        top = self.viewed_items_by_domain[domain]
        top = Counter(top)
        top = top.most_common(self.k)
        recommendation = [x[0] for x in top]
        
        return recommendation
