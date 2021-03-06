import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import pandas as pd
from skimage import io, transform
from holodeck.environments import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import holodeck
from holodeck import agents
from holodeck.environments import *
from holodeck.sensors import Sensors
import cv2
import math
import csv
from os import listdir


def rotationMatrixToEulerAngles(R):

    R = np.reshape(R.cpu().numpy(), (3,3))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


class SinglePoseDataset(Dataset):

    def __init__(self, img_dir, img_transform=None, custom_len=None, euler=False, im_name="/im"):

        if custom_len is None:
            self.custom_length = -1
        else:
            self.custom_length = custom_len

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
        image = self.transform(orig_image)

        if self.euler:
            sample = (image, angles[9:12])
        else:
            sample = (image, angles[0:9])

        return sample


def save_img(state, writer, img_loc, img_ind, rot, pixel_loc=None, added_x=None):
    image = state[Sensors.VIEWPORT_CAPTURE][:, :, 0:3]
    img_name = img_loc + "images/im" + str(img_ind) + ".jpg"
    cv2.imwrite(img_name, image)
    row = list(state[Sensors.ORIENTATION_SENSOR].flatten())
    if pixel_loc is not None:
        row.append(rot[0])
        row.append(rot[1])
        row.append(rot[2])
        row.append(pixel_loc[0])
        row.append(pixel_loc[1])
        row.append(added_x)
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

def gather_single_data(img_loc, n_images=2000):
    """This editor example shows how to interact with holodeck worlds while they are being built
    in the Unreal Engine. Most people that use holodeck will not need this.
    """
    sensors = [Sensors.ORIENTATION_SENSOR, Sensors.IMU_SENSOR, Sensors.LOCATION_SENSOR, Sensors.VIEWPORT_CAPTURE]
    agent = AgentDefinition("uav0", agents.UavAgent, sensors)
    env = HolodeckEnvironment(agent, start_world=False)
    state, _, _, _ = env.reset()
    wait_time = 5
    loc = np.copy(state[Sensors.LOCATION_SENSOR]) * 100

    with open(img_loc + 'orientations.csv', mode='w') as employee_file:
        orientation_writer = csv.writer(employee_file, delimiter=',')
        orientation_writer.writerow(["u1x","u1y","u1z","u2x","u2y","u2z","u3x","u3y","u3z","rotx","roty", "rotz","pixel_x","pixel_y", "added_x"])

        for img_ind in range(0, n_images):

            rot_x = np.random.randint(-180, 180)
            rot_y = np.random.randint(-90, 90)
            rot_z = np.random.randint(-180, 180)

            rot = [rot_x, rot_y, rot_z]

            tel_loc = np.copy(loc)
            added_x = np.random.randint(-50,100)
            tel_loc[0] += added_x
            env.teleport("uav0", location=tel_loc, rotation=rot)
            for _ in range(wait_time):
                state = env.tick()["uav0"]

            uav_loc = state[Sensors.LOCATION_SENSOR]
            dist = np.sqrt(np.sum(np.square(uav_loc)))
            theta_y = -math.asin(uav_loc[2] / dist)
            theta_z = math.asin(uav_loc[1] / np.sqrt(np.sum(np.square(uav_loc[0:2]))))

            # env.teleport_camera([0, 0, 0], list(new_direction))
            pixel = pixel_loc(theta_z, theta_y, 512, 512)
            orientation = state[Sensors.ORIENTATION_SENSOR]

            save_img(state, orientation_writer, img_loc, img_ind, rot, pixel_loc=pixel, added_x=added_x)


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


def gather_single_data_adv():
    """
        1. Set camera position
        2. Set direction and distance of plane
        3. Set orientation of plane
        4. Save image
        5. Record orientation and pixel location of plane
        6. Repeat
    :return:
    """
    img_loc = 'data/NewData/'
    with open(img_loc + 'orientations.csv', mode='w') as employee_file:

        sensors = [Sensors.ORIENTATION_SENSOR, Sensors.LOCATION_SENSOR, Sensors.VIEWPORT_CAPTURE]
        agent = AgentDefinition("uav0", agents.UavAgent, sensors)
        env = HolodeckEnvironment(agent, start_world=False)
        state, _, _, _ = env.reset()

        orientation_writer = csv.writer(employee_file, delimiter=',')
        img_ind = 0

        # 1. Set camera Position
        camera_pos = [0, 0, 0]

        # 2. Set direction and distance of plane from camera
        min_dist = 10
        max_dist = 20
        plane_dist = np.random.randn((1)) * (max_dist - min_dist) + min_dist

        theta_y = np.random.randn((1)) * math.pi / 2
        theta_z = np.random.randn((1)) * 2 * math.pi
        plane_dir = np.matmul(np.matmul(rotate_z(theta_z), rotate_y(-theta_y)), np.array([1, 0, 0]))
        relative_plane_loc = plane_dir * plane_dist

        # 3. Set orientation of plane
        iters = 100

        for i in range(0, iters):
            rot = np.random.randn((3))*2* math.pi
            loc = camera_pos + relative_plane_loc
            env.teleport("uav0", location=loc, rotation=rot)
            new_direction = np.matmul(np.matmul(rotate_z(theta_z), rotate_y(theta_y)), np.array([1,0,0]))
            env.teleport_camera([0, 0, 0], list(new_direction))
            state = env.tick()["uav0"]
            pixel = pixel_loc(theta_z, theta_y, 512, 512)
            save_img(state, orientation_writer, img_loc, img_ind, pixel_loc = pixel)
            img_ind += 1


def create_matching_background(n=500, start_ind=0):
    orientation_path = "data/April18Exp/blank10k/orientations.csv"
    blank_data = pd.read_csv(orientation_path)
    img_loc = "data/April18Exp/background10k/"
    new_orient_path = img_loc + "orientations.csv"
    normal_camera_dist = -215
    # If first
    with open(new_orient_path, mode='a') as employee_file:
        orientation_writer = csv.writer(employee_file, delimiter=',')

        end_ind = start_ind + n

        sensors = [Sensors.ORIENTATION_SENSOR, Sensors.IMU_SENSOR, Sensors.LOCATION_SENSOR, Sensors.VIEWPORT_CAPTURE]
        agent = AgentDefinition("uav0", agents.UavAgent, sensors)
        env = HolodeckEnvironment(agent, start_world=False)
        state, _, _, _ = env.reset()
        wait_time = 5
        loc = np.copy(state[Sensors.LOCATION_SENSOR]) * 100

        for img_ind in range(start_ind, end_ind):

            orients = blank_data.iloc[img_ind, 0:].values.astype(dtype=np.float32)

            rot = orients[9:12]
            added_y = 0
            added_x = orients[14]
            pixel = orients[12:14]
            tel_loc = np.copy(loc)
            tel_loc[0] += added_x
            env.teleport("uav0", location=tel_loc, rotation=rot)
            for _ in range(wait_time):
                state = env.tick()["uav0"]

            save_img(state, orientation_writer, img_loc, img_ind, rot, pixel_loc=pixel, added_x=added_x)

    env.__on_exit__()


if __name__ == '__main__':
    #img_loc = "data/April18Exp/blank10k/"
    #gather_single_data(img_loc, n_images=10000)
    create_matching_background(start_ind=9000, n=1000)
