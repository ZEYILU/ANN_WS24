import torch
import torch.optim as optim
import torch.nn as nn
from model import CNN1
from data_loader import get_data_loaders

def train_model(task="task1", batch_size=64, num_epochs=20, learning_rate=0.001):
    """
    训练 CNN1 模型并返回训练损失和准确率。

    参数:
        task (str): "task1" 或 "task2"，决定数据预处理和模型存储。
        batch_size (int): 训练时的 batch size。
        num_epochs (int): 训练轮数。
        learning_rate (float): 学习率。

    返回:
        train_losses (list): 每个 epoch 的训练损失。
        train_accs (list): 每个 epoch 的训练准确率。
    """
    preprocess_test = True if task == "task2" else False  # 任务 2 需要 TEST 预处理
    train_loader, test_loader, test2_loader = get_data_loaders(batch_size=batch_size, preprocess_test=preprocess_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_loader.dataset.classes)
    model = CNN1(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train_one_epoch(model, dataloader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    train_losses = []
    train_accs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    model_filename = f"{task}_model.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"✅ 训练完成！模型已保存为 {model_filename}")

    return model, train_losses, train_accs
