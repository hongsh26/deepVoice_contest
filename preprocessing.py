import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import scipy
import sklearn.preprocessing
import torchvision
import torch
from PIL import Image
import pandas as pd
import numpy as np
import csv
from torchvision import transforms
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from scipy.io.wavfile import write
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ast
import re

device = ("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return sample, label

def convert_to_array(column):
    return np.array(eval(column))


def scale_column(data, column, scaler):
    X = np.vstack(data[column].values)
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def data_loader(hyperparameter):
    # 데이터 로드
    train_csv = '../src/train_data.csv'
    df = pd.read_csv(train_csv)

    # def process_series(series):
    #     # 불필요한 공백 제거
    #     series = re.sub(r'\s+', '', series)
    #     # 숫자와 공백 사이에 콤마 추가
    #     series = re.sub(r'(?<=\d)\s+(?=\d)', ',', series)
    #     # `e` 또는 `E` 뒤에 있는 `+` 기호와 숫자 2개 뒤에 콤마 추가
    #     series = re.sub(r'([eE][+-]\d{2})', r'\1,', series)
    #     # 적절히 형식을 변경하여 파싱 가능하게 함
    #     return np.array(ast.literal_eval(series))
    #
    # # 각 시리즈 데이터를 처리하여 배열로 변환합니다.
    # df['mfccs'] = df['mfccs'].apply(process_series)
    # df['chroma'] = df['chroma'].apply(process_series)
    # df['spectral_centroid'] = df['spectral_centroid'].apply(process_series)
    # df['zero_crossing_rate'] = df['zero_crossing_rate'].apply(process_series)

    # 피쳐를 하나의 배열로 결합합니다.
    X = np.hstack([
        np.vstack(df['mfccs']),
        np.vstack(df['chroma']),
        np.vstack(df['spectral_centroid']),
        np.vstack(df['zero_crossing_rate'])
    ])

    # 레이블을 추출합니다.
    y = df['label'].values

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, stratify=y, random_state=42)

    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=hyperparameter['batch_size'], shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=hyperparameter['batch_size'], shuffle=False)
    return train_loader, val_loader, X_train.shape[1]


def load_test_data(hyperparameter):
    # 데이터 로드
    train_csv = '../src/test.csv'
    data = pd.read_csv(train_csv)

    # 특성과 레이블 분리
    X = data.drop(columns=['id']).values
    name = data['id'].values
    print(name)
    # 데이터 정규화 (Min-Max 정규화)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    test_dataset = TensorDataset(X_tensor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(test_loader)
    return test_loader, name


#make some noise~
def make_noise(file):
    y, sr = librosa.load('../data/train/' + file, sr=None)
    noise_factor = 0.005
    noise = np.random.normal(len(y))
    y_noisy = y + noise_factor * noise
    write('../data/noise_train/' + file, sr, y_noisy)


def label_data():
    csv_path = '../data/train.csv'
    data = pd.read_csv(csv_path)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

    train_data.to_csv('train_set.csv', index=False)
    val_data.to_csv('val_set.csv', index=False)


# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


def pad_features(features, target_length):
    if features.shape[0] >= target_length:
        return features[:target_length, :]
    else:
        padding = np.zeros((target_length - features.shape[0], features.shape[1]))
        return np.vstack((features, padding))


# def feature(csv_path):
#     count = 0
#     audio_array = pd.read_csv(csv_path)
#     paths = audio_array['path'].values
#     labels = audio_array['label'].values
#     print(labels)
#     all_features = []
#     target_length = 102
#     for path in paths:
#         count += 1
#         print(f"{count}/{len(paths)}")
#         try:
#             path = '../data/test_nonBack/' + path
#             audio, sr = librosa.load(path, sr=16000)  # Load audio at 16kHz
#             inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
#             with torch.no_grad():
#                 features = model(**inputs).last_hidden_state
#
#             features = features.squeeze(0)  # Remove batch dimension
#             features_np = features.cpu().numpy()
#             features_np = pad_features(features_np, target_length)  # Pad or truncate features
#
#             print(f"Features shape: {features_np.shape}")
#             all_features.append(features_np)
#         except FileNotFoundError:
#             print(f"File not found: {path}")
#         except Exception as e:
#             print(f"Error processing file {path}: {e}")
#
#     if all_features:
#         all_features_np = np.vstack([f[:features_np.shape[1]] for f in all_features])
#         df_features = pd.DataFrame(all_features_np)
#         # **라벨 맞추기**
#         df_features['file_name'] = np.repeat(paths, target_length)  # Repeat labels
#         df_features.to_csv("test_features.csv", index=False)
#     else:
#         print("No features were extracted. Please check the input files.")


def extract_features(file_path, sr=16000):
    y, sr = librosa.load(file_path, sr=sr)  # 샘플링 레이트 16000으로 로드
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(mfccs)
    print("shape")
    print(mfccs.shape)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).T
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    features = {
        'id': file_path[19:],
        'mfccs': mfccs.tolist(),
        'chroma': chroma.tolist(),
        'spectral_centroid': spectral_centroid.tolist(),
        'zero_crossing_rate': zero_crossing_rate.tolist(),
    }
    return features


def process_and_label_files(directory, label, sr=16000):
    count = 0
    files = os.listdir(directory)
    features = []
    for file in files:
        try:
            count += 1
            print(f"{count}/{len(files)}")
            file_path = os.path.join(directory, file)
            feature = extract_features(file_path, sr)
            feature['label'] = label
            features.append(feature)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            pass
    return features


# fake_features = process_and_label_files('../data/fake_train', 1)
# real_features = process_and_label_files('../data/real_train', 0)
#
# all_features = fake_features + real_features
#
# df = pd.DataFrame(all_features)
#
# df.to_csv('../src/train_data.csv', index=False)
#
# print(f"{len(all_features)} features saved to '../src/train.csv'")
