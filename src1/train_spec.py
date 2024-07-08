import torch.nn as nn
from torch import optim
import tqdm
import torch
import fadam


def train(dataloader, batch_size, epochs, model, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)
    model.cuda()

    # criterion, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss().cuda()
    # optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    optimizer = fadam.FAdam(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    acc_list = []
    loss_list = []
    for epoch in range(epochs):
        with tqdm.tqdm(dataloader, unit="batch", leave=False) as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward calculation
                outputs = model(inputs)

                # 타겟을 출력 크기와 맞추기 위해 one-hot 인코딩을 사용
                labels_one_hot = torch.zeros(labels.size(0), 2).to(device)
                labels_one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)

                loss = criterion(outputs, labels_one_hot.float())

                # Backward calculation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accuracy calculation
                predicted = torch.round(torch.sigmoid(outputs))
                correct = (predicted == labels_one_hot).sum().item()
                total = labels.size(0) * 2  # 두 클래스의 총 예측 수
                accuracy = correct / total

                tepoch.set_postfix(
                    accuracy=f"{accuracy:.2f}", loss=f"{loss.item():.4f}"
                )
                acc_list.append(accuracy)
                loss_list.append(loss.item())

            scheduler.step()

    print("Training complete.")
    print(f"accuracy: {accuracy}, loss: {loss}")
