import torch
import torch.nn as nn
from model import CNN1
from data_loader import get_data_loaders

def evaluate_model_task2(batch_size=64):
    """
    任务二：评估 CNN1 模型在 TEST 和 TEST2 数据集上的表现。

    参数:
        batch_size (int): 评估时的 batch size。

    返回:
        test_loss (float): TEST 数据集的损失。
        test_acc (float): TEST 数据集的准确率。
        test2_loss (float): TEST2 数据集的损失。
        test2_acc (float): TEST2 数据集的准确率。
    """
    print("📊 任务二：在 TEST 和 TEST2 数据集上评估模型...")

    train_loader, test_loader, test2_loader = get_data_loaders(batch_size=batch_size, preprocess_test=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_loader.dataset.classes)
    model = CNN1(num_classes=num_classes).to(device)

    model_filename = "task2_model.pth"
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    
    criterion = nn.CrossEntropyLoss()

    def evaluate(model, dataloader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    print("📊 评估 TEST 数据集：")
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"TEST - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    print("\n📊 评估 TEST2 数据集：")
    test2_loss, test2_acc = evaluate(model, test2_loader, criterion)
    print(f"TEST2 - Loss: {test2_loss:.4f}, Acc: {test2_acc:.4f}")

    return test_loss, test_acc, test2_loss, test2_acc
