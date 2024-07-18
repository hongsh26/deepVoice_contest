import torch.nn as nn
from torch import optim
import model
import tqdm
import torch
import scheduler
import os
import pandas as pd
from pyAudioAnalysis import audioSegmentation as aS
import numpy as np


def test(test_loader, file_names):
    filters_1, kernel_size_1 = 16, 3
    filters_2, kernel_size_2 = 64, 3
    filters_3, kernel_size_3 = 32, 3
    filters_4, kernel_size_4 = 64, 3
    units = 512
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    for inputs in test_loader:
        input_length = inputs[0].shape[1]
        print(input_length)
        break
    results = []
    models = model.CNN(filters_1, kernel_size_1, filters_2, kernel_size_2, filters_3, kernel_size_3, filters_4,
                       kernel_size_4, units, input_length).to(device)
    # model = EFNet_L2(1)
    models.load_state_dict(torch.load('../src/fakaAudio-16-30-6-fadam.pth'))
    models = models.to(device)
    models.eval()
    with torch.no_grad():
        for i, (inputs) in enumerate(test_loader):
            inputs = inputs[0]
            inputs = inputs.unsqueeze(1)
            ai_output, human_output = models(inputs)
            print('결과표 : ai = ', ai_output.item(), ',  human : ', human_output.item())
            results.append((file_names[i], ai_output.item(), human_output.item()))
    return results


def save_results(results, output_csv_path):
    # sample_submission.csv 파일을 로드하여 결과 형식 맞추기
    sample_submission = pd.read_csv('../data/sample_submission.csv')
    # 예측 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results, columns=['id', 'fake', 'real'])
    results_df['id'] = results_df['id'].str.replace('.wav', '')
    print(results_df.head(5))
    # 결과를 sample_submission 형식에 맞추기
    final_submission = sample_submission.copy()
    final_submission['fake'] = final_submission['id'].map(results_df.set_index('id')['fake'])
    final_submission['real'] = final_submission['id'].map(results_df.set_index('id')['real'])

    # 결과 저장
    final_submission.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")


def division(file):
    audio_file = "../data/test/" + file
    segments, classes = aS.speaker_diarization(audio_file, 2)
    print(segments)
    print(classes)
