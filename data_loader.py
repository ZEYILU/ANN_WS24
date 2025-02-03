from torch.utils.data import DataLoader, Dataset
import torch
import os

class NumpyDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        """
        data_dir: 数据目录，每个子文件夹为一个类别
        mode: 'train' 或 'test'，决定采用哪种预处理函数
        transform: 其他额外变换（本例中未使用）
        """
        self.data_dir = data_dir
        self.mode = mode  # 'train' 或 'test'
        self.transform = transform
        self.samples = []  # 存储 (file_path, label) 对
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
        # 加载 .npy 文件，假定 shape 为 (C, H, W)
        image = np.load(file_path)
        # 取第一个通道作为灰度图（将 shape 从 (C, H, W) 变为 (H, W)）
        image = image[0]
        # 转换为 float32 并归一化到 [0,1]（假设原始数据范围为 0-255）
        image = image.astype(np.float32) / 255.0
        # 转换为 tensor，并加上通道维度，最终 shape 为 (1, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        
        # 根据 mode 调用不同的预处理函数
        if self.mode == 'train':
            image = process_train_image(image)
        elif self.mode == 'test':
            image = process_test_image(image)
        else:
            # 默认不处理
            pass
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label
