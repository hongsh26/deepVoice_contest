from torch import nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 32)  # 추가된 층
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 16)  # 추가된 층
        self.fc9 = nn.Linear(16, 8)
        self.fc10 = nn.Linear(8, 8)   # 추가된 층
        self.fc11 = nn.Linear(8, 2)   # 출력 층
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))  # 추가된 층
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))  # 추가된 층
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x)) # 추가된 층
        x = self.softmax(self.fc11(x))  # 출력 층
        return x


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # Conv1d와 MaxPool1d 레이어를 거친 후의 출력 크기를 계산합니다.
        # 여기서 input_dim은 특징의 개수입니다.
        conv_output_dim = input_dim // 4  # MaxPool1d로 인해 크기가 두 번 절반으로 줄어듭니다.

        self.fc1 = nn.Linear(64 * conv_output_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = self.relu(self.conv1(x))  # (batch_size, 1, input_dim) -> (batch_size, 32, input_dim)
        x = self.pool(x)  # (batch_size, 32, input_dim) -> (batch_size, 32, input_dim / 2)
        x = self.relu(self.conv2(x))  # (batch_size, 32, input_dim / 2) -> (batch_size, 64, input_dim / 2)
        x = self.pool(x)  # (batch_size, 64, input_dim / 2) -> (batch_size, 64, input_dim / 4)
        x = x.view(x.size(0), -1)  # 펼치기 (batch_size, 64 * input_dim / 4)
        x = self.relu(self.fc1(x))  # (batch_size, 64)
        x = self.relu(self.fc2(x))  # (batch_size, 32)
        x = self.softmax(self.fc3(x))  # (batch_size, 2)
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)