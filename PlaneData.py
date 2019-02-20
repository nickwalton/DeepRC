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
import math
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


def save_img(state, writer, img_loc, img_ind):
    image = state[Sensors.VIEWPORT_CAPTURE][:, :, 0:3]
    dir = state[Sensors.ORIENTATION_SENSOR]
    img_name = img_loc + "images/im" + str(img_ind) + ".jpg"
    cv2.imwrite(img_name, image)
    row = [img_name, str(dir[0][0]), str(dir[0][1]), str(dir[0][2]),
           str(dir[1][0]), str(dir[1][1]), str(dir[1][2]),
           str(dir[2][0]), str(dir[2][1]), str(dir[2][2])]
    writer.writerow(row)


def rotate_x(theta):
    return np.array(
        [[1, 0, 0],
         [0, math.cos(theta), -math.sin(theta)],
         [0, math.sin(theta), math.cos(theta)]])


def rotate_y(theta):
    return np.array(
        [ [math.cos(theta), 0,  math.sin(theta)],
          [0, 1, 0],
          [-math.sin(theta), 0,  math.cos(theta)]])


def rotate_z(theta):
    return np.array(
        [[math.cos(theta), - math.sin(theta), 0],
        [ math.sin(theta), math.cos(theta), 0],
         [0, 0, 1]])


def gather_data_simple():
    """This editor example shows how to interact with holodeck worlds while they are being built
    in the Unreal Engine. Most people that use holodeck will not need this.
    """
    sensors = [Sensors.ORIENTATION_SENSOR, Sensors.LOCATION_SENSOR, Sensors.VIEWPORT_CAPTURE]
    agent = AgentDefinition("uav0", agents.UavAgent, sensors)
    env = HolodeckEnvironment(agent, start_world=False)
    state, _, _, _ = env.reset()

    img_loc = 'data/plane2_data1/'

    with open(img_loc + 'orientations.csv', mode='w') as employee_file:
        orientation_writer = csv.writer(employee_file, delimiter=',')
        prec = 40
        iters = int(360/prec)
        img_ind = 0

        for i in range(0, iters):
            print("Percent Rendered ", str(100*i/iters) + "%")
            for j in range(iters):
                for k in range(iters):
                    rot = [i*prec, j*prec, k*prec]
                    loc = state[Sensors.LOCATION_SENSOR]*100
                    env.teleport("uav0", location=loc, rotation=rot)
                    state = env.tick()["uav0"]
                    save_img(state, orientation_writer, img_loc, img_ind)
                    img_ind +=1


def uav_tracker():
    sensors = [Sensors.ORIENTATION_SENSOR, Sensors.LOCATION_SENSOR, Sensors.VIEWPORT_CAPTURE]
    agent = AgentDefinition("uav0", agents.UavAgent, sensors)
    env = HolodeckEnvironment(agent, start_world=False)
    state, _, _, _ = env.reset()
    command = [0, 0, 0, 100000]

    for i in range(10000):

        uav_loc = state[Sensors.LOCATION_SENSOR]
        dist = np.sqrt(np.sum(np.square(uav_loc)))
        theta_y = -math.asin(uav_loc[2] / dist)
        theta_z = math.asin(uav_loc[1] / np.sqrt(np.sum(np.square(uav_loc[0:2]))))

        direction_vec = np.array([1, 0, 0])

        new_direction = np.matmul(np.matmul(rotate_z(theta_z), rotate_y(theta_y)), direction_vec)

        #env.teleport_camera([0, 0, 0], list(new_direction))
        pixel = pixel_loc(theta_z, theta_y, 512, 512)
        print(pixel)
        state, _, _, _ = env.step(command)


def pixel_loc(theta_z, theta_y, width, height):
    #horizontal_fov = 2.11677606
    vertical_fov = 1.5708
    horizontal_fov = vertical_fov

    x_ratio = (theta_z + horizontal_fov/2) / horizontal_fov
    y_ratio = (theta_y + vertical_fov/2) / vertical_fov

    x_pos = int(width*x_ratio)
    y_pos = int(height*y_ratio)

    return x_pos, y_pos


if __name__ == '__main__':
    uav_tracker()
