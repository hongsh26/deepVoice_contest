import torch.nn as nn
from torch import optim
from model import LeNet_5
import tqdm
import torch
import scheduler
import os
import pandas as pd

def test(test_loader, file_names):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    model = LeNet_5()
    model.load_state_dict(torch.load('../src/fakaAudio-11-5-fadam.pth'))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, indices = batch
            images = images.to(device)
            outputs = torch.sigmoid(model(images)).cpu().numpy()  # sigmoid로 확률 계산
            for idx, output in zip(indices, outputs):
                file_name = file_names[idx]
                results.append((file_name, 1 - output[0], output[0]))  # fake 확률, real 확률
                print('file_name: ', file_name)
                print('fake: ', 1 - output[0])
                print('real: ', output[0])
    return results

def save_results(results, output_csv_path):
    # sample_submission.csv 파일을 로드하여 결과 형식 맞추기
    sample_submission = pd.read_csv('../data/sample_submission.csv')

    # 예측 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results, columns=['id', 'fake', 'real'])
    results_df['id'] = results_df['id'].str.replace('.png', '')
    print(results_df.head(5))
    # 결과를 sample_submission 형식에 맞추기
    final_submission = sample_submission.copy()
    final_submission['fake'] = final_submission['id'].map(results_df.set_index('id')['fake'])
    final_submission['real'] = final_submission['id'].map(results_df.set_index('id')['real'])

    # 결과 저장
    final_submission.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")