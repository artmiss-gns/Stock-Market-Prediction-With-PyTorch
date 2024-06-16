
from torch import nn
import torch

def eval(model, criterion, dataloader, device='cpu'):
    model = model.to(device)
    model.eval()
    batch_loss = 0
    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            # x = x.permute(1, 0).unsqueeze(dim=0)
            x = x.to(device)
            y = y.to(device)

            output, (h_n, c_n) = model(x)
            loss = criterion(output.squeeze(dim=0), y)

            batch_loss += loss.item()

    batch_loss /= (batch+1) 
    return batch_loss