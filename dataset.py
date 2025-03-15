import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle


class HAM10kDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        image_id = sample["image"]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = sample["label"]
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.df)


class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None, class_label=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.class_label = class_label

        # Define the batch files for train and test
        if self.train:
            files = [f"data_batch_{i}" for i in range(1, 6)]  # 5 training batches
        else:
            files = ["test_batch"]  # Single test batch

        self.data, self.targets = self.load_data(files)

    def load_data(self, files):
        data, targets = [], []
        for file_name in files:
            file_path = os.path.join(self.root, "cifar-10-batches-py", file_name)
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data.append(batch[b'data'])
                targets.extend(batch[b'labels'])

        # Convert to NumPy array
        data = np.vstack(data).reshape(-1, 3, 32, 32).astype(np.uint8)  # Reshape to (N, C, H, W)
        targets = np.array(targets)
        if self.class_label is not None:
            data = data[targets == self.class_label]
            targets = targets[targets == self.class_label]
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        img = np.transpose(img, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

        if self.transform:
            img = self.transform(img)

        return img, label
