import torch.nn as nn
from torch import optim
import model
import tqdm
import torch
import scheduler
import fadam
import statistics


def train(dataloader1, dataloader2, device, hyperparameters, model):
    graph = []
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = fadam.FAdam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    for epoch in range(hyperparameters["num_epochs"]):
        acc_list = []
        loss_list = []
        model = model.to(device)
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        with tqdm.tqdm(dataloader1, unit="it", leave=False) as tepoch:
            for _, (inputs, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                print(inputs)

                inputs = inputs.to(device).unsqueeze(1)
                labels = labels.to(device).flaot()
                #labels = labels.unsqueeze(1)  # convert labels to same nn output shape


                # Forward calculation
                ai_output, human_output = model(inputs)
                # 각각의 출력에 대해 손실을 계산하고 합산
                loss_ai = criterion(ai_output, labels)
                loss_human = criterion(human_output, labels)
                loss = loss_ai + loss_human

                # Backward calculation and optimization
                loss.backward()
                optimizer.step()

                # Accuracy calculation
                running_loss += loss.item()

                # 예측 계산 (여기서는 ai_output을 사용)
                predicted = (ai_output >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = correct / total

                tepoch.set_postfix(accuracy=f"{accuracy:.2f}", loss=f"{loss.item():.4f}")
                acc_list.append(accuracy)
                loss_list.append(loss.item())

            scheduler.step()
            result, val_acc, val_loss = validate(dataloader2, device, model)
            train_acc = round(statistics.mean(acc_list), 3)
            train_loss = round(statistics.mean(loss_list), 3)
            graph.append((train_acc, train_loss, round(val_acc, 3), round(val_loss, 3)))

    print("Training complete. Saving model...")
    torch.save(model.state_dict(), f"fakaAudio-16-30-6-fadam.pth")
    print(f"Model saved. fakaAudio.pth")
    return {"acc": accuracy, "loss": loss}, graph


def validate(dataloader, device, model):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss().cuda()
    with torch.no_grad():
        with tqdm.tqdm(dataloader, unit="it", leave=False) as tepoch:
            for _, (inputs, labels) in enumerate(tepoch):
                inputs = inputs.to(device)
                inputs = inputs.unsqueeze(1)  # Add channel dimension
                labels = labels.to(device)
                labels = labels.unsqueeze(1)

                # Forward calculation
                ai_output, human_output = model(inputs)

                # 각각의 출력에 대해 손실을 계산하고 합산
                loss_ai = criterion(ai_output, labels)
                loss_human = criterion(human_output, labels)
                loss = loss_ai + loss_human

                val_loss += loss.item()

                # 예측 계산 (여기서는 ai_output을 사용)
                predicted = (ai_output >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            val_loss /= len(dataloader)
            return {"acc": accuracy, "loss": val_loss}, accuracy, val_loss
