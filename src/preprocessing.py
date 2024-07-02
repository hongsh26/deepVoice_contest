import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import torchvision
import torch
from PIL import Image
import pandas as pd
import numpy as np
import csv
from torchvision import transforms
import pickle
from torch.utils.data import DataLoader, TensorDataset

device = ("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def save_spectrogram(path, save):
    audio, sample_rate = librosa.load(path)
    n_fft = 2048  # 창 크기
    hop_length = n_fft // 4  # 홉 크기
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    mel_spect = librosa.power_to_db(S, ref=np.max)

    mel_spect = torch.tensor(mel_spect).to(device)

    dir_path = save
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = os.path.basename(path).replace('.ogg', '.png')
    save_path = os.path.join(dir_path, file_name)

    mel_spect = mel_spect.cpu().numpy()
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(mel_spect, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    print(save_path)
    plt.savefig(save_path)
    plt.close()


def load_and_label_data(dir, csv):
    data = []
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor()
    ])

    try:
        train = pd.read_csv(csv)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return data

    # Load images
    print("start --------")
    count = 0
    for filename in os.listdir(dir):
        count += 1
        print(count)
        try:
            filepath = os.path.join(dir, filename)
            image = Image.open(filepath)
            image = transform(image)
            image = image.view(-1).numpy()  # Flatten the image

            # result = train[train['id'] == filename]
            result = train[train['id'] == filename.replace('.png', '')]
            label = result['label'].values[0]
            print(result)
            print(result['label'].values[0])
            data.append((image.tolist(), label))
        except Exception as e:
            print(f"Skipping file {filename} due to error: {e}")
    return data


def save_csv(data):
    f = open('../data/train_set.csv', 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    header = ['label'] + [f'pixel{i}' for i in range(28 * 28)]
    wr.writerow(header)

    for tensor, label in data:
        wr.writerow([tensor, label])


def data_loader():
    train_csv = '../data/train_set.csv'
    train_df = pd.read_csv(train_csv)
    train_labels = train_df.iloc[:, 1].apply(lambda x: 1 if x == 'real' else 0)
    train_labels = torch.tensor(train_labels.values, dtype=torch.float32)

    train_data = train_df.iloc[:, 0].apply(
        lambda x: np.array(eval(x), dtype=np.float32) if isinstance(x, str) else np.array(x, dtype=np.float32))
    train_data = np.stack(train_data.values)  # 리스트를 numpy 배열로 변환
    train_data = torch.tensor(train_data).view(-1, 1, 28, 28)
    train_data = train_data.repeat(1, 3, 1, 1)

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader

def load_test_data(file_path):
    # test_image.pkl 파일을 로드
    with open(file_path, 'rb') as f:
        image_data = pickle.load(f)

    images = []
    file_names = []
    for file_name, image_tensor in image_data.items():
        images.append(image_tensor)
        file_names.append(file_name)

    # 텐서 데이터셋 생성
    images_tensor = torch.stack(images)
    dataset = TensorDataset(images_tensor, torch.tensor(range(len(images_tensor))))  # dummy targets
    return DataLoader(dataset, batch_size=32, shuffle=False), file_names


def preprocess_and_save_images(image_dir, output_file):
    trans = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image_data = {}

    for file_name in os.listdir(image_dir):
        print(file_name)
        image_path = os.path.join(image_dir, file_name)
        image = Image.open(image_path).convert('RGB')
        image_tensor = trans(image)
        image_data[file_name] = image_tensor

    with open(output_file, 'wb') as f:
        pickle.dump(image_data, f)


# preprocess_and_save_images('../data/test_spectrogram', '../data/test_image.pkl')
# train_data = pd.read_csv('../data/train.csv')
# files = os.listdir('../data/test')
# data = load_and_label_data('../data/train_spectrogram','../data/train.csv')
# save_csv(data)
# count = 0
# for file in files:
#     count += 1
#     print(f"{count}번째 파일 진행중")
#     print(file)
#     save_spectrogram('../data/test/' + file, '../data/test_spectrogram')
# print("end------")

# count = 0
# for file in files:
#     find_row = train_data.loc[train_data['id'] == file[:8]]
#     print(find_row)
