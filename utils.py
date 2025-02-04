import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.filters import threshold_otsu
import os
import random
import numpy as np
import matplotlib.pyplot as plt

def dataset_stats(dataset):
    """
    计算数据集的统计信息（最小值、最大值、均值、标准差）。

    参数:
        dataset (Dataset): PyTorch 数据集对象。

    返回:
        dict: 数据集的统计信息，包括 min, max, mean, std。
    """
    all_pixels = []
    for img, _ in dataset:
        all_pixels.append(img.flatten())
    all_pixels = torch.cat(all_pixels)  # 拼到一个大张量里

    return {
        'min': float(all_pixels.min()),
        'max': float(all_pixels.max()),
        'mean': float(all_pixels.mean()),
        'std':  float(all_pixels.std())
    }

def count_samples(data_dir):
    """
    统计数据集中每个类别的样本数量。

    参数:
        data_dir (str): 数据集路径，每个类别一个子文件夹。

    返回:
        dict: {类别: 样本数}
    """
    if not os.path.exists(data_dir):
        print(f"⚠️ Warning: 数据目录 {data_dir} 不存在！")
        return {}

    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    sample_counts = {category: len(os.listdir(os.path.join(data_dir, category))) for category in categories}

    return sample_counts


def visualize_samples(data_dir, num_images=5):
    """
    从数据集中每个类别随机选取 `num_images` 个样本进行可视化。

    参数:
        data_dir (str): 数据集路径，每个类别一个子文件夹。
        num_images (int): 每个类别随机显示的样本数量（默认 5）。

    """
    if not os.path.exists(data_dir):
        print(f"⚠️ Warning: 数据目录 {data_dir} 不存在！")
        return

    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for category in categories:
        category_path = os.path.join(data_dir, category)
        file_list = os.listdir(category_path)

        if len(file_list) == 0:
            continue  # 若类别下无样本，则跳过

        # 选取 num_images 张样本进行可视化
        n_images = min(num_images, len(file_list))
        chosen_files = random.sample(file_list, n_images)

        plt.figure(figsize=(3 * n_images, 3))
        plt.suptitle(f"Category: {category}")

        for i, file in enumerate(chosen_files):
            sample_file = os.path.join(category_path, file)
            sample = np.load(sample_file)  # 读取 .npy 文件
            image = sample[0]  # 取第一个通道作为灰度图

            plt.subplot(1, n_images, i + 1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')

        plt.show()


def process_train_image(image_tensor):
    """ 训练数据预处理（目前不做修改） """
    return image_tensor


def process_test_image(image_tensor):
    """
    测试集预处理：
    - 采用 Otsu 方法计算最佳二值化阈值
    - 统计高于阈值的像素占比
    - 若占比 > 0.5，则图像是白底黑字，需要反转为黑底白字

    参数:
        image_tensor: (1, H, W)，已归一化到 [0,1]
    返回:
        处理后的 image_tensor，确保为黑底白字
    """
    image_np = image_tensor.squeeze().cpu().numpy()
    otsu_thresh = threshold_otsu(image_np)
    proportion = np.mean(image_np > otsu_thresh)

    # 如果白色区域占比大于 50%，说明是白底黑字，进行翻转
    if proportion > 0.5:
        return 1.0 - image_tensor
    return image_tensor


class NumpyDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        """
        自定义数据集，用于加载 .npy 格式的图像数据。

        参数:
            data_dir: 数据所在的文件夹，每个类别一个子文件夹
            mode: 'train'（训练集）或 'test'（测试集）
        """
        self.data_dir = data_dir
        self.mode = mode  # train / test
        self.samples = []  # 存储 (文件路径, 标签)
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
        image = np.load(file_path)  # 加载 numpy 数组
        image = image[0]  # 取第一个通道作为灰度图 (H, W)

        # 归一化到 [0,1]
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # 变为 (1, H, W)

        # 根据 mode 选择不同的预处理方式
        if self.mode == 'train':
            image = process_train_image(image)
        elif self.mode == 'test':
            image = process_test_image(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label


def get_data_loaders(batch_size=32, train_dir="TRAIN", test_dir="TEST", test2_dir="TEST2"):
    """
    创建数据加载器，并使用默认数据路径。

    参数:
        batch_size: 批量大小
        train_dir: 训练数据路径（默认: "data/TRAIN"）
        test_dir: 测试数据路径（默认: "data/TEST"）
        test2_dir: 额外测试数据路径（默认: "data/TEST2"）

    返回:
        train_loader, test_loader, test2_loader (如果提供 TEST2 数据)
    """
    train_dataset = NumpyDataset(train_dir, mode='train')
    test_dataset = NumpyDataset(test_dir, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 只有当 TEST2 数据目录存在时才加载它
    test2_loader = None
    if os.path.exists(test2_dir) and os.path.isdir(test2_dir):
        test2_dataset = NumpyDataset(test2_dir, mode='test')
        test2_loader = DataLoader(test2_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test2_loader


# # ========== 示例用法 ==========
# if __name__ == "__main__":
#     train_loader, test_loader, test2_loader = get_data_loaders(batch_size=64)

#     # 打印数据集大小
#     print(f"Train dataset size: {len(train_loader.dataset)}")
#     print(f"Test dataset size: {len(test_loader.dataset)}")
#     if test2_loader:
#         print(f"Test2 dataset size: {len(test2_loader.dataset)}")
