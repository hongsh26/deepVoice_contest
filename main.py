import os
import torch
import test
import train
import preprocessing
import model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# CSV 파일 불러오기
device = ("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

print(">> Training Mode <<")
# data = pd.read_csv("../src/train.csv")
# X = data.drop(columns=['id', 'label'])
# y = data['label']

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# dtrain = xgb.DMatrix(X_train, label=y_train)
# dval = xgb.DMatrix(X_val, label=y_val)
#
# param_grid = {
#     'max_depth': [6, 7, 8],
#     'learning_rate': [0.015, 0.1, 0.15],
#     'n_estimators': [390, 395, 400],
#     'subsample': [0.55, 0.6, 0.65],
#     'colsample_bytree': [0.878, 0.88, 0.882]
# }
# xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
#
# # GridSearchCV 초기화
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1)
# # Grid Search 수행
# grid_search.fit(X_train, y_train)
#
# # 최적의 하이퍼파라미터 출력
# print(f'Best parameters found: {grid_search.best_params_}')
#
# # 최적의 모델로 예측 및 평가
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_val)
#
# accuracy = accuracy_score(y_val, y_pred)
# precision = precision_score(y_val, y_pred)
# recall = recall_score(y_val, y_pred)
# f1 = f1_score(y_val, y_pred)
#
# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'F1 Score: {f1}')
hyperparameters = {"batch_size": 16, "num_epochs": 20, "learning_rate": 1e-4}

print(hyperparameters)
train_loader, valid_loader, shape = preprocessing.data_loader(hyperparameters)

# 예시 데이터 로드 (train_loader에서 하나의 배치를 가져와서 확인)
for inputs, labels in train_loader:
    inputs = inputs.numpy()
    print("입력 데이터의 최소값:", np.min(inputs))
    print("입력 데이터의 최대값:", np.max(inputs))
    print("입력 데이터의 평균:", np.mean(inputs))
    print("입력 데이터의 표준편차:", np.std(inputs))
    break
for inputs, _ in train_loader:
    print(inputs)
    input_length = inputs.shape[1]
    print(input_length)
    break
model = model.CNN(filters_1=32, kernel_size_1=3, filters_2=64, kernel_size_2=3, filters_3=128, kernel_size_3=3, filters_4=256, kernel_size_4=3, units=256, input_length=shape)
result, graph  = train.train(train_loader, valid_loader, device, hyperparameters, model)
print(result)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for i, data in enumerate(graph):
    epoch = i + 1
    train_acc, train_loss, val_acc, val_loss = data

    if i == 0:
        ax1.plot(epoch, train_acc, 'bo-', label="Train Accuracy")
        ax1.plot(epoch, val_acc, 'yo-', label="Validation Accuracy")
        ax2.plot(epoch, train_loss, 'bo-', label="Train Loss")
        ax2.plot(epoch, val_loss, 'yo-', label="Validation Loss")
    else:
        ax1.plot(epoch, train_acc, 'bo-')
        ax1.plot(epoch, val_acc, 'yo-')
        ax2.plot(epoch, train_loss, 'bo-')
        ax2.plot(epoch, val_loss, 'yo-')

ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()

print("test start ---------")
test_loader, file_names = preprocessing.load_test_data(hyperparameters)
for input in test_loader:
    print(input[0])
    break
print('집에 보내줘', file_names)
result = test.test(test_loader, file_names)
print("여기야")
test.save_results(result, '../data/test_results.csv')
