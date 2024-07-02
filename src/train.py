import torch.nn as nn
from torch import optim
import model
import tqdm
import torch
import scheduler
import fadam

def train(dataloader, device, hyperparameters, model):
    model.train()
    model = model.to(device)
    model.cuda()

    # criterion, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss().cuda()
    # optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    optimizer = fadam.FAdam(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    acc_list =[]
    loss_list =[]
    for epoch in range(hyperparameters["num_epochs"]):
        with tqdm.tqdm(dataloader, unit="it", leave=False) as tepoch:
            for _, (inputs, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch + 1}")

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward calculation
                outputs = model(inputs)
                labels = labels.unsqueeze(1)  # convert labels to same nn output shape
                loss = criterion(outputs, labels.float())

                # Backward calculation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accuracy calculation
                predicted = torch.round(torch.sigmoid(outputs))
                correct = (predicted == labels).sum().item()
                total = labels.size(0)
                accuracy = correct / total

                tepoch.set_postfix(
                    accuracy=f"{accuracy:.2f}", loss=f"{loss.item():.4f}"
                )
                acc_list.append(accuracy)
                loss_list.append(loss.item())

            scheduler.step()


    print("Training complete. Saving model...")
    torch.save(model.state_dict(), f"fakaAudio-11-5-fadam.pth")
    print(f"Model saved. fakaAudio.pth")
    return {"acc": accuracy, "loss": loss}
