import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import pandas as pd
from skimage import io, transform
from holodeck.environments import *
import numpy as np

import holodeck
from holodeck import agents
from holodeck.environments import *
from holodeck.sensors import Sensors
import cv2
import csv


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


def save_img(state, writer, img_ind):
    image = state[Sensors.VIEWPORT_CAPTURE][:, :, 0:3]
    dir = state[Sensors.ORIENTATION_SENSOR]
    img_name = "data/plane_data2/im" + str(img_ind) + ".jpg"
    cv2.imwrite(img_name, image)
    row = [img_name, str(dir[0][0]), str(dir[0][1]), str(dir[0][2]),
           str(dir[1][0]), str(dir[1][1]), str(dir[1][2]),
           str(dir[2][0]), str(dir[2][1]), str(dir[2][2])]
    writer.writerow(row)


def gather_data():
    """This editor example shows how to interact with holodeck worlds while they are being built
    in the Unreal Engine. Most people that use holodeck will not need this.
    """
    sensors = [Sensors.ORIENTATION_SENSOR, Sensors.LOCATION_SENSOR, Sensors.VIEWPORT_CAPTURE]
    agent = AgentDefinition("uav0", agents.UavAgent, sensors)
    env = HolodeckEnvironment(agent, start_world=False)
    state, _, _, _ = env.reset()

    with open('data/plane_data2/orientations.csv', mode='w') as employee_file:
        orientation_writer = csv.writer(employee_file, delimiter=',')
        prec = 36
        iters = int(360/prec)
        img_ind = 0

        for i in range(iters):
            for j in range(iters):
                for k in range(iters):
                    rot = [i*prec, j*prec, k*prec]
                    loc = state[Sensors.LOCATION_SENSOR]*100
                    env.teleport("uav0", location=loc, rotation=rot)
                    state = env.tick()["uav0"]
                    save_img(state, orientation_writer, img_ind)
                    img_ind +=1


if __name__ == '__main__':
    gather_data()
