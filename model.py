# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class CNN(nn.Module):
#     def __init__(self, filters_1, kernel_size_1, filters_2, kernel_size_2, filters_3, kernel_size_3, filters_4,
#                  kernel_size_4, units, input_length):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters_1, kernel_size=kernel_size_1,
#                                padding=kernel_size_1 // 2)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         self.dropout1 = nn.Dropout(p=0.2)
#
#         self.conv2 = nn.Conv1d(in_channels=filters_1, out_channels=filters_2, kernel_size=kernel_size_2,
#                                padding=kernel_size_2 // 2)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         self.dropout2 = nn.Dropout(p=0.2)
#
#         self.conv3 = nn.Conv1d(in_channels=filters_2, out_channels=filters_3, kernel_size=kernel_size_3,
#                                padding=kernel_size_3 // 2)
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         self.dropout3 = nn.Dropout(p=0.2)
#
#         self.conv4 = nn.Conv1d(in_channels=filters_3, out_channels=filters_4, kernel_size=kernel_size_4,
#                                padding=kernel_size_4 // 2)
#         self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         self.dropout4 = nn.Dropout(p=0.2)
#
#         # 최종 출력 크기 계산 (패딩 포함)
#         conv1_out_length = (input_length + 2 * (kernel_size_1 // 2) - kernel_size_1) // 1 + 1
#         pool1_out_length = (conv1_out_length + 2 * 1 - 2) // 2 + 1
#
#         conv2_out_length = (pool1_out_length + 2 * (kernel_size_2 // 2) - kernel_size_2) // 1 + 1
#         pool2_out_length = (conv2_out_length + 2 * 1 - 2) // 2 + 1
#
#         conv3_out_length = (pool2_out_length + 2 * (kernel_size_3 // 2) - kernel_size_3) // 1 + 1
#         pool3_out_length = (conv3_out_length + 2 * 1 - 2) // 2 + 1
#
#         conv4_out_length = (pool3_out_length + 2 * (kernel_size_4 // 2) - kernel_size_4) // 1 + 1
#         pool4_out_length = (conv4_out_length + 2 * 1 - 2) // 2 + 1
#
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(filters_4 * pool4_out_length, units)
#         self.dropout5 = nn.Dropout(p=0.5)
#
#         self.fc2_ai = nn.Linear(units, 1)
#         self.fc2_human = nn.Linear(units, 1)
#
#     def forward(self, x):
#         x = self.dropout1(F.relu(self.conv1(x)))
#         x = self.pool1(x)
#
#         x = self.dropout2(F.relu(self.conv2(x)))
#         x = self.pool2(x)
#
#         x = self.dropout3(F.relu(self.conv3(x)))
#         x = self.pool3(x)
#
#         x = self.dropout4(F.relu(self.conv4(x)))
#         x = self.pool4(x)
#
#         x = self.flatten(x)
#         x = self.dropout5(F.relu(self.fc1(x)))
#
#         ai_output = self.fc2_ai(x)
#         human_output = self.fc2_human(x)
#
#         return torch.sigmoid(ai_output), torch.sigmoid(human_output)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, filters_1, kernel_size_1, filters_2, kernel_size_2, filters_3, kernel_size_3, filters_4,
                 kernel_size_4, units, input_length):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters_1, kernel_size=kernel_size_1,
                               padding=kernel_size_1 // 2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(in_channels=filters_1, out_channels=filters_2, kernel_size=kernel_size_2,
                               padding=kernel_size_2 // 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(in_channels=filters_2, out_channels=filters_3, kernel_size=kernel_size_3,
                               padding=kernel_size_3 // 2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout3 = nn.Dropout(p=0.2)
        self.conv4 = nn.Conv1d(in_channels=filters_3, out_channels=filters_4, kernel_size=kernel_size_4,
                               padding=kernel_size_4 // 2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout4 = nn.Dropout(p=0.2)

        conv1_out_length = (input_length + 2 * (kernel_size_1 // 2) - kernel_size_1) // 1 + 1
        pool1_out_length = (conv1_out_length + 2 * 1 - 2) // 2 + 1
        conv2_out_length = (pool1_out_length + 2 * (kernel_size_2 // 2) - kernel_size_2) // 1 + 1
        pool2_out_length = (conv2_out_length + 2 * 1 - 2) // 2 + 1
        conv3_out_length = (pool2_out_length + 2 * (kernel_size_3 // 2) - kernel_size_3) // 1 + 1
        pool3_out_length = (conv3_out_length + 2 * 1 - 2) // 2 + 1
        conv4_out_length = (pool3_out_length + 2 * (kernel_size_4 // 2) - kernel_size_4) // 1 + 1
        pool4_out_length = (conv4_out_length + 2 * 1 - 2) // 2 + 1

        self.fc1 = nn.Linear(filters_4 * pool4_out_length, units)
        self.fc2 = nn.Linear(units, 2)  # 두 개의 독립적인 출력 (fake, real)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 두 개의 독립적인 출력
        return x

