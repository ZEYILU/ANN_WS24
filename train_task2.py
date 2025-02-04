import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from model import CNN1
from data_loader import get_data_loaders

def plot_training_curves(train_losses, train_accs):
    """
    绘制训练损失和准确率曲线。

    参数:
        train_losses (list): 每个 epoch 的训练损失。
        train_accs (list): 每个 epoch 的训练准确率。
    """
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(12, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def train_task2(batch_size=64, num_epochs=20, learning_rate=0.001):
    """
    任务二：使用 TRAIN + TEST 作为训练数据，训练 CNN1 模型。

    参数:
        batch_size (int): 训练时的 batch size。
        num_epochs (int): 训练轮数。
        learning_rate (float): 学习率。

    返回:
        model (nn.Module): 训练后的 PyTorch 模型。
        train_losses (list): 每个 epoch 的训练损失。
        train_accs (list): 每个 epoch 的训练准确率。
    """
    print("🚀 任务二：使用 TRAIN + TEST 训练模型...")
    
    # 任务二的关键：让 `TEST` 数据也作为训练数据
    train_loader, test_loader, test2_loader = get_data_loaders(batch_size=batch_size, preprocess_test=True)

    # **拼接 `TRAIN` 和 `TEST` 数据集**
    full_train_loader = torch.utils.data.DataLoader(
        train_loader.dataset + test_loader.dataset, batch_size=batch_size, shuffle=True
    )

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
        train_loss, train_acc = train_one_epoch(model, full_train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    model_filename = "task2_model.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"✅ 训练完成！模型已保存为 {model_filename}")

    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs)

    return model, train_losses, train_accs
