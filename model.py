import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN1, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(10 * 14 * 14, num_classes)
    def forward(self,x):
        #print("Input shape:", x.shape)
        x = self.conv(x)
        #print("After Conv, shape:", x.shape)
        x = F.relu(x)
        x = self.pool(x)
        #print("After Pooling, shape:", x.shape)
        x = x.view(x.size(0), -1)
        #print("After Flatten, shape:", x.shape)
        x = self.fc(x)
        #print("After FC, shape:", x.shape)
        return x