import torch.nn as nn
from torch import optim
import model
import tqdm
import torch
import scheduler

def evaluate_model(test_loader, device, hyperparameters):
    model = torch.load('../src/model.pt')
    model = model.to(device)
    print("Reading validation dataset...")
    test_loss = 0
    correct = 0
