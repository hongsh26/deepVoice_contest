import pandas as pd

data = pd.read_csv('../src/train_data.csv')
print("head(1)")
print(data.head(1))
print("head(1)['mfccs']")
print(data.head(1)['mfccs'].shape)
