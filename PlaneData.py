import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import pandas as pd
from skimage import io, transform
from holodeck.environments import *


class SinglePoseDataset(Dataset):

    def __init__(self, img_dir, csv_file, img_transform, label="orientation", len=None):
        self.data = pd.read_csv(csv_file)
        self.transform = img_transform
        self.label = label
        self.adjusted_len = len

        self.img_dir = img_dir

        self.images = datasets.ImageFolder(img_dir, self.transform)

    def __len__(self):
        if self.adjusted_len is not None:
            return self.adjusted_len
        else:
            return 1000

    def __getitem__(self, idx):
        orientation = torch.Tensor(self.data.iloc[idx, 1:].values.astype(dtype=np.float32))

        image = self.images[idx]

        if self.label is "orientation":
            sample = (image[0], orientation)
        else:
            sample = (image[0], image[0])

        return sample
