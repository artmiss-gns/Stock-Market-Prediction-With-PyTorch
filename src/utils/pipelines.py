from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

from utils.dataset import MakeSequence


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        super().__init__()
        self.columns = columns

    def fit(self, X):
        return self

    def transform(self, X:pd.DataFrame):
        return X.drop(columns=self.columns)
    

class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self.feature = feature
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[[self.feature]])
        return self

    def transform(self, X):
        X[[self.feature]] = self.scaler.transform(X[[self.feature]])
        return X

class SequencePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, sequence_length, target):
        self.sequence_length = sequence_length
        self.target = target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return MakeSequence(X, self.sequence_length, target=self.target)
    

if __name__ == '__main__':
    from utils.train_test_split import train_test_split

    SEQUENCE_NUMBER = 7
    BATCH_SIZE = 16
    TRAIN_SIZE = 0.85

    preprocess_pipeline = Pipeline([
        ('drop_columns', DropColumns(['Open', 'High', 'Low', 'Adj Close', 'Date'])),
        ('scale_close', Scaler(feature='Close')),
        ('scale_volume', Scaler(feature='Volume')),
        ('make_sequence', SequencePipeline(sequence_length=SEQUENCE_NUMBER, target='Close'))
    ])

    data = pd.read_csv('./data/REMEDY.HE.csv')
    train, test = train_test_split(data, train_size_split=0.85)
    train_data, test_data = preprocess_pipeline.fit_transform(train), preprocess_pipeline.fit_transform(test)