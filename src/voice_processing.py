import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import librosa
import librosa.display


def read_audio(file_path):
    sample_rate, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = data[:, 0]
    return sample_rate, data


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def extract_features(data, sample_rate):
    stft = np.abs(librosa.stft(data))
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate)

    features = np.concatenate((mfccs, chroma, mel, contrast, tonnetz), axis=0)
    return features

def extract_and_plot_features(data, sample_rate, title):
    plt.figure(figsize=(12, 8))
    n_fft = min(len(data), 1024)
    hop_length = n_fft // 4
    # 스펙트로그램
    plt.subplot(3, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title} - Spectrogram')

    # MFCC
    plt.subplot(3, 1, 2)
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time', hop_length=hop_length)
    plt.colorbar()
    plt.title(f'{title} - MFCC')

    # Chroma
    plt.subplot(3, 1, 3)
    chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    librosa.display.specshow(chroma, sr=sample_rate, x_axis='time', y_axis='chroma', hop_length=hop_length)
    plt.colorbar()
    plt.title(f'{title} - Chroma')

    plt.tight_layout()
    plt.show()


# 오디오 파일 읽기
human_sample_rate, human_data = read_audio('converted_audio_real1.wav')
deepvoice_sample_rate, deepvoice_data = read_audio('converted_audio_fake1.wav')

# 주파수 범위 설정
lowcut1, highcut1 = 200, 400
lowcut2, highcut2 = 800, 1000
lowcut3, highcut3 = 1500, 2000

# 필터링
filtered_human_data1 = bandpass_filter(human_data, lowcut1, highcut1, human_sample_rate)
filtered_deepvoice_data1 = bandpass_filter(deepvoice_data, lowcut1, highcut1, deepvoice_sample_rate)
filtered_human_data2 = bandpass_filter(human_data, lowcut2, highcut2, human_sample_rate)
filtered_deepvoice_data2 = bandpass_filter(deepvoice_data, lowcut2, highcut2, deepvoice_sample_rate)
filtered_human_data3 = bandpass_filter(human_data, lowcut3, highcut3, human_sample_rate)
filtered_deepvoice_data3 = bandpass_filter(deepvoice_data, lowcut3, highcut3, deepvoice_sample_rate)

# 특징 추출
human_features1 = extract_features(filtered_human_data1, human_sample_rate)
deepvoice_features1 = extract_features(filtered_deepvoice_data1, deepvoice_sample_rate)
human_features2 = extract_features(filtered_human_data2, human_sample_rate)
deepvoice_features2 = extract_features(filtered_deepvoice_data2, deepvoice_sample_rate)
human_features3 = extract_features(filtered_human_data3, human_sample_rate)
deepvoice_features3 = extract_features(filtered_deepvoice_data3, deepvoice_sample_rate)

# CNN 모델을 위한 데이터 준비 (필요시 각 범위별로 따로 수행)
# 데이터를 적절한 형태로 변환하여 모델 학습에 사용

extract_and_plot_features(human_features1, human_sample_rate, 'Human voice')
extract_and_plot_features(deepvoice_features1, deepvoice_sample_rate, 'Ai voice')