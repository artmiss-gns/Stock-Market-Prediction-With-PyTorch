import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.pipeline import Pipeline

from utils.pipelines import DropColumns, Scaler, SequencePipeline
from utils.train_test_split import train_test_split
from models import LSTM

SEQUENCE_NUMBER = 7
BATCH_SIZE = 16
TRAIN_SIZE = 0.85
HIDDEN_SIZE = 64
NUM_LAYERS = 4
INPUT_SIZE = 2
LR = 0.001
EPOCH = 15

preprocess_pipeline = Pipeline([
    ('drop_columns', DropColumns(['Open', 'High', 'Low', 'Adj Close', 'Date'])),
    ('scale_close', Scaler(feature='Close')),
    ('scale_volume', Scaler(feature='Volume')),
    ('make_sequence', SequencePipeline(sequence_length=SEQUENCE_NUMBER, target='Close'))
])

# load and preprocess
data = pd.read_csv('./data/REMEDY.HE.csv')
train, test = train_test_split(data, train_size_split=0.85)
train_data = preprocess_pipeline.fit_transform(train)
test_data = preprocess_pipeline.fit_transform(test)

# dataloader
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True, # ! set it to True, the results become pretty good!
    drop_last=True,
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

for epoch in range(EPOCH):
    pass