import pickle

import torch
import train
import model
import preprocessing
import test

device = ("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# print(">> Training Mode <<")
# hyperparameters = {"batch_size": 10, "num_epochs": 11, "learning_rate": 1e-5}
# print(hyperparameters)
# model = model.LeNet_5().to(device)
# dataloader = preprocessing.data_loader()
# result = train.train(dataloader, device, hyperparameters, model)
# print(result)


print("data loading")
print(">> Test Mode <<")
test_loader, file_names = preprocessing.load_test_data('../data/test_image.pkl')
result = test.test(test_loader, file_names)
print("여기야")
print(result)
test.save_results(result, '../data/test_results.csv')