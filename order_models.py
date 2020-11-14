from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF

class PriceBasedOrder(object):
    def __init__(self, metadata, sold_prices, verbose=True):
        self.metadata = metadata
        self.ecdf = ECDF(sold_prices)
        self.verbose = verbose
        
    def fit(self, X=None, y=None):
        return self
    
    def predict(self, X=None):
        results = []
        for items in (tqdm(X) if self.verbose else X):
            scores = self._scores(items)
            results.append([item for _, item in sorted(zip(scores, items), reverse=True)])
        return results
    
    def _scores(self, items):
        scores = []
        for item in items:
            price = float(self.metadata[item]['price'] if self.metadata[item]['price'] is not None else 99999)
            score = self.ecdf(price) - self.ecdf(max(0, price - 10))
            scores.append(score)
        return scores
