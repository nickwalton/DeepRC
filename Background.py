import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import pandas as pd
from skimage import io, transform
from holodeck.environments import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import holodeck
from holodeck import agents
from holodeck.environments import *
from holodeck.sensors import Sensors
import cv2
import math
import csv
from os import listdir
import random

img_transform = transforms.Compose([
    transforms.CenterCrop((128, 128))#,
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class SinglePoseDataset(Dataset):

    def __init__(self, img_dir, img_transform=None, custom_len=None, euler=False, im_name="/im"):

        if custom_len is None:
            self.custom_length = -1
        else:
            self.custom_length = custom_len

        self.background_img_dir = "data/Background/images"

        self.im_name = im_name

        self.euler = euler
        csv_file = img_dir + "/orientations.csv"
        self.data = pd.read_csv(csv_file)
        if img_transform is None:
            self.transform = transforms.Compose([])
        else:
            self.transform = img_transform

        self.img_dir = img_dir + "/images"
        self.len = len(os.listdir(self.img_dir))

    def is_blue(self,elem):
        if elem[0] < 0.75 and elem[0] > 0.55 and elem[1] > 0.75:
            return True
        else:
            return False

    def random_background(self, orig_img):
        random_img_path = random.choice(os.listdir(self.background_img_dir))
        back_img = Image.open(self.background_img_dir + "/" + random_img_path)
        back_img = self.transform(back_img)
        back_img_np = np.asarray(back_img)
        img = np.asarray(orig_img)
        width, height = img.shape[0], img.shape[1]
        img.setflags(write=1)
        img = img / 255.0
        back_img_np = back_img_np / 255.0

        hsv_img = matplotlib.colors.rgb_to_hsv(img)
        hsv_background = matplotlib.colors.rgb_to_hsv(back_img_np)

        for i in range(width):
            for j in range(height):
                if self.is_blue(hsv_img[i,j]):
                    hsv_img[i,j] = hsv_background[i,j]

        img = matplotlib.colors.hsv_to_rgb(hsv_img)
        return img

    def __len__(self):
        if self.custom_length is not -1:
            return self.custom_length
        else:
            return self.len

    def __getitem__(self, idx):
        angles = self.data.iloc[idx, 0:].values.astype(dtype=np.float32)
        sin = np.sin(angles*math.pi/180.0)
        cos = np.cos(angles*math.pi/180.0)
        sin_cos = np.stack([sin, cos], axis=1)
        sin_cos = torch.Tensor(sin_cos)
        img_path = self.img_dir + self.im_name + str(idx) + ".jpg"
        orig_image = Image.open(img_path)
        orig_image.show(0)
        image = self.transform(orig_image)
        image = self.random_background(image)
        sample = (image, angles[0:9])

        return sample


if __name__ == '__main__':
    dataset = SinglePoseDataset("data/VaporLite", img_transform=img_transform)
    im = dataset[2][0]
