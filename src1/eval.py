import torch


def eval(model, loader, criterion, epoch, n_epochs, loss):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in loader:
            val_outputs = model(X_val_batch)
            val_loss += criterion(val_outputs, y_val_batch).item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            total += y_val_batch.size(0)
            correct += (val_predicted == y_val_batch).sum().item()

    val_accuracy = correct / total
    val_loss /= len(loader)
    print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
