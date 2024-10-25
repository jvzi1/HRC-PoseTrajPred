import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import CONFIG

def train_model(model, train_dataloader, val_dataloader):
    model = model.to(CONFIG.DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.LEARNING_RATE)

    for epoch in range(CONFIG.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(CONFIG.DEVICE), labels.to(CONFIG.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        # 验证集
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(CONFIG.DEVICE), labels.to(CONFIG.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels)
                total += labels.size(0)

        print(f'Epoch {epoch + 1}/{CONFIG.NUM_EPOCHS}, Loss: {running_loss / len(train_dataloader.dataset)}, Val Acc: {correct.double() / total}')
    
    return model
