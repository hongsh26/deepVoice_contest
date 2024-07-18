import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio
import torchaudio
import os
from pyannote.audio import Pipeline
from pydub import AudioSegment
import librosa


def denoising(file):
    model = pretrained.dns64().to('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    file_path = os.path.join("../data/test", file)
    noisy, sr = torchaudio.load(file_path)

    noisy = convert_audio(noisy, sr, model.sample_rate, model.chin)
    noisy = noisy.to(device)
    # 음성 정제
    with torch.no_grad():
        denoised = model(noisy.unsqueeze(0)).squeeze(0).cpu()

    # 정제된 음성 저장
    denoised_file = f"../data/claer_{file[:10]}.wav"
    torchaudio.save(denoised_file, denoised, sample_rate=model.sample_rate)


# 음성 파일 로드
audio_path = '../data/claer_TEST_02688.wav'
audio, sr = librosa.load(audio_path, sr=16000)
