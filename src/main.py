import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.pipeline import Pipeline

from utils.pipelines import DropColumns, Scaler, SequencePipeline
from utils.train_test_split import train_test_split
from models.LSTM import LSTM
from src.train import train
from src.eval import eval

SEQUENCE_NUMBER = 7
BATCH_SIZE = 16
TRAIN_SIZE = 0.85
HIDDEN_SIZE = 64
NUM_LAYERS = 4
INPUT_SIZE = 2
LR = 0.001
EPOCH = 15

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def preprocess():
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
        shuffle=True, # ! setting it to True, the results become pretty good!
        drop_last=True,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )

    return train_dataloader, test_dataloader

def main():
    train_dataloader, test_dataloader = preprocess()

    model = LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_size=BATCH_SIZE).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loss_values = list()
    test_loss_values = list()
    for epoch in range(EPOCH):
        print("train...")
        train_loss = train(model, criterion, optimizer, train_dataloader, device=device)
        print('test...')
        test_loss = eval(model, criterion, test_dataloader, device=device)

        train_loss_values.append(train_loss)
        test_loss_values.append(test_loss)

        print("EPOCH:", epoch)
        print("train-loss: {0:.6f}".format(train_loss_values[epoch]))
        print("test-loss: {0:.6f}".format(test_loss_values[epoch]))


if __name__ == '__main__':
    main()