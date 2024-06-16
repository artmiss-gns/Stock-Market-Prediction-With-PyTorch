from torch import nn
import torch

def train(model, criterion, optimizer, dataloader, device='cpu'):
    model = model.to(device)
    model.train()
    batch_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        output, (h_n, c_n) = model(x)
        loss = criterion(output.squeeze(dim=0), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()

    batch_loss /= (batch+1) 
    return batch_loss
