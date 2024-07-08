import preprocessing
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import model
import fadam
import train_model
import pandas as pd
import test
import train_spec
import argparse

fake_train_path = '../data/fake_train.csv'
real_train_path = '../data/real_train.csv'
fake_val_path = '../data/fake_val.csv'
real_val_path = '../data/real_val.csv'
test_data_path = '../data/test_feature.csv'

parser = argparse.ArgumentParser(description='Training settings')
parser.add_argument('--batchSize', type=int, default=16, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for training')
args = parser.parse_args()

device = ("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

train_data = preprocessing.combine_train(real_train_path, fake_train_path)
valid_data = preprocessing.combine_train(real_val_path, fake_val_path)
test_data = pd.read_csv(test_data_path)

X_train = train_data.iloc[:, 1:-1].values
y_train = train_data.iloc[:, -1].values

X_valid = valid_data.iloc[:, 1:-1].values
y_valid = valid_data.iloc[:, -1].values

X_test = test_data.iloc[:, 1:].values
# print(X_test)

# 데이터 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
X_valid_scaled = scaler.fit_transform(X_valid)
valid_dataset = TensorDataset(torch.tensor(X_valid_scaled, dtype=torch.float32),
                              torch.tensor(y_valid, dtype=torch.long))

X_test_scaled = scaler.fit_transform(X_test)
test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))

train_loader1 = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
test_loader1, file_names = preprocessing.load_test_data('../data/test_image.pkl')
input_dim = X_test_scaled.shape[1]  # 여기가 중요합니다. X_train_scaled.shape[1]가 8이어야 합니다.
print(f'Input dimension: {input_dim}')

models = [
    model.DNN(input_dim).to(device),
    model.LeNet_5().to(device),
]
trained_models = []

for i, model_instance in enumerate(models):
    epoch = 11
    if (i == 1):
        train_loader = preprocessing.data_loader()
        result = train_spec.train(train_loader, args.batchSize, epoch, model_instance, learning_rate=args.lr)
        torch.save(model_instance.state_dict(), f"model_{i}.pth")
        trained_models.append(model_instance)
    else:
        # 데이터 샘플링 (랜덤 서브샘플링)
        sample_size = int(0.8 * len(train_dataset))  # 전체 데이터의 80%를 사용
        train_subset, _ = random_split(train_dataset, [sample_size, len(train_dataset) - sample_size])
        sampled_train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)

        model_instance.apply(model_instance.init_weights)

        criterion = nn.CrossEntropyLoss()
        # optimizer = fadam.FAdam(model_instance.parameters())
        optimizer = optim.Adam(model_instance.parameters(), lr=1e-5)
        train_model.train(model_instance, train_loader1, val_loader, criterion, optimizer, epoch)
        torch.save(model_instance.state_dict(), f"model_{i}.pth")
        trained_models.append(model_instance)

# result = test.test(test_loader, input_dim)
results = test.ensemble_test(test_loader, test_loader1, trained_models, device)
# print(result)
test.save_result(results, '../data/output.csv')
