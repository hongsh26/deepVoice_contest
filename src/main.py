import pickle
import torch
import train
import model
import model1
import preprocessing
import test
from efficientnet_pytorch import EfficientNet

device = ("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
#
# print(">> Training Mode <<")
# hyperparameters = {"batch_size": 16, "num_epochs": 11, "learning_rate": 1e-5}
# print(hyperparameters)
# model = model.LeNet_5().to(device)
# # model = model1.EFNet_L2(1).to(device)
# # model = EfficientNet.from_pretrained("efficientnet-b7", num_classes=2).from_name('efficientnet-b0')
# # model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
# dataloader = preprocessing.data_loader()
# result = train.train(dataloader, device, hyperparameters, model)
# print(result)


print("data loading")
print(">> Test Mode <<")
test_loader, file_names = preprocessing.load_test_data('../data/denoising_test_image.pkl')
result = test.test(test_loader, file_names)
print("여기야")
print(result)
test.save_results(result, '../data/test_results.csv')