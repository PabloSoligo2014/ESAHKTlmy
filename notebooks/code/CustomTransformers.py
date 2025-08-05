from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TimeSeriesFreqRegularization(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['value']):
        super().__init__()
        self.columns = columns
        
    def fit(self, X, y=None):
        # the index should be a datetime index
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex")
        
        #estimated frequency
        deltas = X.index.to_series().diff().dt.total_seconds()
        # frequency in seconds
        self.most_frequent_frequency_ = int(deltas.mode()[0])  # Frecuencia más común en segundos
        return self
    
    def get_feature_names_out(self, input_features=None):
        return self.columns
        
    def transform(self, X, y=None):
        # Copy
        Xc  = X.copy()
        # Reindex to the most frequent frequency
        idx = Xc.index.union(pd.date_range(Xc.index.min(), Xc.index.max(), freq=f'{self.most_frequent_frequency_}s'))
        Xc   = Xc.reindex(idx)
        
        # Interpolate missing values
        Xc[self.columns] = Xc[self.columns].interpolate(method='time')
        #Remove extra rows
        Xc = Xc[Xc.index.isin(pd.date_range(Xc.index.min(), Xc.index.max(), freq=f'{self.most_frequent_frequency_}s'))]
        Xc.index = pd.DatetimeIndex(Xc.index)
        Xc.index.freq = f'{self.most_frequent_frequency_}s'
        return Xc