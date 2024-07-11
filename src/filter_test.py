import os
import numpy as np
import librosa
import pyroomacoustics as pra
from pyAudioAnalysis import audioSegmentation as aS
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def detect_speakers(audio_path, n_clusters=2):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    features = features.T  # Transpose to shape (n_frames, n_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_

    return len(set(labels))


def separate_speakers(audio_path, output_dir):
    y, sr = librosa.load(audio_path, sr=None)
    n_sources = 2  # Assuming maximum 2 speakers

    N = 512  # STFT window length
    L = N // 2  # Filter size

    # STFT
    Y = pra.transform.stft.analysis(y, N, L)

    # Apply the binary mask
    X_hat = pra.bss.auxiva(Y, n_sources)

    # ISTFT
    estimates = [pra.transform.stft.synthesis(X_hat[j], N, L) for j in range(n_sources)]

    # Save the separated audio files
    for i, estimate in enumerate(estimates):
        output_file = os.path.join(output_dir, f"speaker_{i + 1}.ogg")
        librosa.output.write_wav(output_file, estimate, sr)

    return [os.path.join(output_dir, f"speaker_{i + 1}.ogg") for i in range(n_sources)]


def remove_noise(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    noise_profile = librosa.effects.split(y, top_db=30)
    y_denoised = []
    for interval in noise_profile:
        y_denoised.extend(y[interval[0]:interval[1]])
    librosa.output.write_wav(output_path, np.array(y_denoised), sr)


def save_spectrum(audio_path, output_dir):
    y, sr = librosa.load(audio_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(os.path.join(output_dir, os.path.basename(audio_path).replace('.ogg', '_spectrum.png')))
    plt.close()


def process_audio(input_dir, output_dir):
    for audio_file in os.listdir(input_dir):
        if audio_file.endswith('.ogg'):
            audio_path = os.path.join(input_dir, audio_file)

            # Detect number of speakers
            n_speakers = detect_speakers(audio_path)

            if n_speakers > 1:
                # Separate speakers
                separated_files = separate_speakers(audio_path, output_dir)
            else:
                separated_files = [audio_path]

            for sep_file in separated_files:
                # Remove noise
                noise_removed_file = sep_file.replace('.ogg', '_denoised.wav')
                remove_noise(sep_file, noise_removed_file)

                # Save spectrum
                save_spectrum(noise_removed_file, output_dir)


input_directory = '../data/test'
output_directory = '../data/test_nonBack'
os.makedirs(output_directory, exist_ok=True)
process_audio(input_directory, output_directory)