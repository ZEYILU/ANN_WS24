import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.filters import threshold_otsu

def process_train_image(image_tensor):
    """ 训练数据的预处理（默认不修改）。"""
    return image_tensor

def process_test_image(image_tensor):
    """
    统一测试集图像颜色：
    - 使用 Otsu 方法计算最佳阈值，并判断背景是白底黑字还是黑底白字。
    - 若背景为白色，则反转图像，使其统一为黑底白字。

    参数:
        image_tensor (Tensor): 形状为 (1, H, W)，值归一化到 [0,1]。

    返回:
        处理后的 image_tensor，确保为黑底白字。
    """
    image_np = image_tensor.squeeze().cpu().numpy()
    otsu_thresh = threshold_otsu(image_np)
    proportion = np.mean(image_np > otsu_thresh)

    if proportion > 0.5:  # 白底黑字，需要翻转
        return 1.0 - image_tensor
    return image_tensor

class NumpyDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        """
        自定义数据集，支持 TRAIN, TEST, TEST2 数据加载。

        参数:
            data_dir (str): 数据集根目录，每个类别一个子文件夹。
            mode (str): 'train' 或 'test'，决定数据预处理方式。
        """
        self.data_dir = data_dir
        self.mode = mode
        self.samples = []
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_folder = os.path.join(data_dir, cls)
            for file in os.listdir(cls_folder):
                if file.endswith('.npy'):
                    file_path = os.path.join(cls_folder, file)
                    self.samples.append((file_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]
        image = np.load(file_path)[0]  # 取第一个通道作为灰度图

        # 确保数据是 float32
        image = image.astype(np.float32)

        # 只有数据最大值超过 1.0 才进行归一化
        if image.max() > 1.0:
            image /= 255.0

        image = torch.from_numpy(image).unsqueeze(0)  # 变为 (1, H, W)

        if self.mode == 'train':
            image = process_train_image(image)
        elif self.mode == 'test':
            image = process_test_image(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_data_loaders(batch_size=32, train_dir="D:/thws Mai/ann/TRAIN", test_dir="D:/thws Mai/ann/TEST", test2_dir="D:/thws Mai/ann/TEST2", preprocess_test=False):
    """
    获取数据加载器，并支持 `TEST` 预处理开关。

    参数:
        batch_size (int): 批量大小。
        train_dir (str): 训练数据目录（默认 "data/TRAIN"）。
        test_dir (str): 测试数据目录（默认 "data/TEST"）。
        test2_dir (str): 额外测试数据目录（默认 "data/TEST2"）。
        preprocess_test (bool): 是否对测试集进行颜色一致性处理。

    返回:
        train_loader, test_loader, test2_loader (如果提供 TEST2 数据)
    """
    train_dataset = NumpyDataset(train_dir, mode='train')
    test_dataset = NumpyDataset(test_dir, mode='test' if preprocess_test else 'train')  # 控制是否预处理 TEST

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test2_loader = None
    if os.path.exists(test2_dir) and os.path.isdir(test2_dir):
        test2_dataset = NumpyDataset(test2_dir, mode='test' if preprocess_test else 'train')
        test2_loader = DataLoader(test2_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test2_loader
