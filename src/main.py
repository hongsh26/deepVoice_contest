import torch
import train
import model
import preprocessing

device = ("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
print(">> Training Mode <<")
hyperparameters = {"batch_size": 10, "num_epochs": 11, "learning_rate": 1e-5}
print(hyperparameters)
model = model.LeNet_5().to(device)
dataloader = preprocessing.data_loader()
result = train.train(dataloader, device, hyperparameters, model)
print(result)
