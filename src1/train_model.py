import torch


def train(model, train_loader, val_loader, criterion, optimizer, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    n_epochs = epoch
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        val_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        model.eval()
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = model(X_val_batch)
                val_loss += criterion(val_outputs, y_val_batch).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                total += y_val_batch.size(0)
                correct += (val_predicted == y_val_batch).sum().item()

        val_accuracy = correct / total
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        scheduler.step(val_loss)
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), f"adam-batch16-epoch20.pth")
    print(f"Model saved. fakaAudio.pth")