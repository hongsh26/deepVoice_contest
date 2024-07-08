import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import model
from sklearn.preprocessing import StandardScaler
import tqdm
import torch
import os


def test(test_loader, input_dim):
    results = []
    models = model.DNN(input_dim)
    models.load_state_dict(torch.load('../src/fadam-batch16-epoch30.pth'))
    models.eval()
    files = os.listdir('../data/test')
    files = sorted(files)
    with torch.no_grad():
        for i, (X_batch,) in enumerate(test_loader):
            print(f'Input batch shape: {X_batch.shape}')
            outputs = models(X_batch)
            print("output")
            print(len(outputs))
            for j, output in (enumerate(outputs)):
                idx = files[i * test_loader.batch_size + j]
                file_name = idx.replace('.ogg', '')
                fake_prob, real_prob = output[0].item(), output[1].item()
                results.append((file_name, fake_prob, real_prob))                # print('file_name: ', file_name)
                # print('fake: ', 1 - output[0])
                # print('real: ', output[0])
    return results


def ensemble_test(test_loader_numeric, test_loader_spectrogram, models, device):
    results = []
    files = os.listdir('../data/test')
    files = sorted(files)
    models = [model.to(device) for model in models]

    with torch.no_grad():
        spectrogram_iter = iter(test_loader_spectrogram)

        for i, (X_batch_numeric,) in enumerate(test_loader_numeric):
            X_batch_numeric = X_batch_numeric.to(device)  # 데이터 디바이스 이동
            ensemble_outputs = torch.zeros(X_batch_numeric.size(0), 2).to(device)  # 결과 텐서 디바이스 이동

            # 첫 번째 모델 (수치 기반 모델)
            outputs_numeric = models[0](X_batch_numeric)
            ensemble_outputs += outputs_numeric

            # 두 번째 모델 (스펙트로그램 기반 모델)
            try:
                X_batch_spectrogram = next(spectrogram_iter)[0].to(device)  # 데이터 디바이스 이동
                if X_batch_spectrogram.size(0) != X_batch_numeric.size(0):
                    # 배치 크기를 맞추기 위한 코드 추가
                    while X_batch_spectrogram.size(0) < X_batch_numeric.size(0):
                        X_batch_spectrogram = torch.cat((X_batch_spectrogram, next(spectrogram_iter)[0].to(device)),
                                                        dim=0)
                    if X_batch_spectrogram.size(0) > X_batch_numeric.size(0):
                        X_batch_spectrogram = X_batch_spectrogram[:X_batch_numeric.size(0)]
                outputs_spectrogram = models[1](X_batch_spectrogram)

                # 두 출력의 크기를 맞추기 위한 코드 추가
                if outputs_spectrogram.size(1) != outputs_numeric.size(1):
                    outputs_spectrogram = outputs_spectrogram[:, :outputs_numeric.size(1)]  # 일치하도록 조정

            except StopIteration:
                break

            ensemble_outputs += outputs_spectrogram

            # 평균 앙상블
            ensemble_outputs /= len(models)

            for j, output in enumerate(ensemble_outputs):
                idx = files[i * test_loader_numeric.batch_size + j]
                file_name = idx.replace('.ogg', '')
                fake_prob, real_prob = output[0].item(), output[1].item()
                results.append((file_name, fake_prob, real_prob))

    return results

def save_result(results, output_csv_path):
    sample_submission = pd.read_csv('../data/sample_submission.csv')
    results_converted = [(id, fake, real) for id, fake, real in results]
    results_df = pd.DataFrame(results_converted, columns=['id', 'fake', 'real'])

    final_submission = sample_submission.drop(columns=['fake', 'real']).merge(results_df, on='id', how='left')

    final_submission.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
