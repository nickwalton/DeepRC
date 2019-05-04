import torch.nn as nn
import torch
from PlaneData import *
from coordconv import CoordConv1d, CoordConv2d, CoordConv3d
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


img_transform = transforms.Compose([
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def convert_to_degrees(sin_cos):
    return np.arctan2(sin_cos[:,0], sin_cos[:,1])*180.0/math.pi


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


class TestDataset(Dataset):

    def __init__(self, simulated=False, custom_len=None, euler=False, im_name="/im"):

        self.simulated = simulated
        if self.simulated:
            img_dir= "data/April18Exp/background2k"
        else:
            img_dir = "data/April18Exp/test"
        if custom_len is None:
            self.custom_length = -1
        else:
            self.custom_length = custom_len

        self.im_name = im_name

        self.euler = euler
        csv_file = img_dir + "/orientations.csv"
        self.data = pd.read_csv(csv_file)

        if self.simulated:
            self.transform = transforms.Compose([
                transforms.CenterCrop((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.img_dir = img_dir + "/images"
        self.len = len(os.listdir(self.img_dir))

    def __len__(self):
        if self.simulated:
            return self.len
        else:
            return 30

    def __getitem__(self, idx):
        idx = idx
        or_data = self.data.iloc[idx, 0:].values.astype(dtype=np.float32)
        img_path = self.img_dir + self.im_name + str(idx+1) + ".jpg"
        orig_image = Image.open(img_path)

        pixels = or_data[1:3]
        angles = or_data[3:6]
        angles = eulerAnglesToRotationMatrix(angles*math.pi/180).reshape(9).astype(np.float32)
        image = self.transform(orig_image)

        sample = (image, angles)

        return sample


class DownConvLayer(nn.Module):
    def __init__(self, input_depth, output_depth, activation=nn.ReLU, coord=False, use_dropout=False):
        super(DownConvLayer, self).__init__()
        if coord:
            self.conv_layer = CoordConv2d(input_depth, output_depth, 5, stride=2, padding=2, with_r=True)
        else:
            self.conv_layer = nn.Conv2d(input_depth, output_depth, 5, stride=2, padding=2)

        self.activation = activation()
        self.drop = nn.Dropout(0.4)
        self.use_dropout = use_dropout

    def forward(self, x):
        output = self.activation(self.conv_layer(x))

        if self.use_dropout:
            return self.drop(output)
        else:
            return output


class RegressionNet(nn.Module):

    def __init__(self):
        super(RegressionNet, self).__init__()

        self.net = nn.Sequential(
            DownConvLayer(3, 24, coord=False),
            DownConvLayer(24, 48),
            DownConvLayer(48, 24),
            DownConvLayer(24, 8))

        self.fc1 = nn.Linear(8*8*8,128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x1 = self.net(x)
        x2 = self.act1(self.fc1(x1.view(x1.size(0), -1)))
        output = self.fc2(x2)
        return output


def train(model, train_loader, optimizer, loss_func):
    model.train()
    num = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("cuda"), target.to("cuda")
        optimizer.zero_grad()
        output = model(data)
        num += target.shape[0]
        loss = loss_func(output, target)
        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss.item()


def test(model, test_loader, loss_func, results_writer, row, should_print=False):
    model.eval()
    total_loss = 0
    num = 0
    or_errors = np.zeros(9)
    angle_err = np.zeros(3)
    with torch.no_grad():

        for data, target in test_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            total_loss += loss_func(output, target)
            num += data.shape[0]

            or_errors += np.abs(target.cpu().numpy()[0] - output.cpu().numpy()[0])
            pred_angles = rotationMatrixToEulerAngles(output)* 180/math.pi
            target_angles = rotationMatrixToEulerAngles(target) * 180/math.pi

            err = np.abs(pred_angles - target_angles)
            alt_err = np.abs(err - 360)

            for i in range(3):
                if alt_err[i] < err[i]:
                    err[i] = alt_err[i]

            angle_err += err

        print("Avg Angle Error", angle_err / num)

        row.append(total_loss.item()/num)
        row.append(sum(angle_err) / (3*num))
        row.append(sum(or_errors) / (num * 9))
        results_writer.writerow(row)
        if should_print:
            print("Angle err", err)
            #print("Avg Orientation err: ", err / num)
            print("Total avg Err: ", sum(or_errors) / (num * 9))
            print("Total Avg Angle Error: ", sum(angle_err) / (3*num))

        return total_loss.item()/num


def run_regression(simulated_test_data=False):
    batch_size = 48
    lr = 1e-4
    epochs = 5

    image_dir = "data/April18Exp/background10k"
    pose_dataset = SinglePoseDataset(image_dir, img_transform)

    if simulated_test_data:
        image_dir = "data/April18Exp/background2k"
        test_dataset = SinglePoseDataset(image_dir, img_transform)
    else:
        test_dataset = TestDataset(simulated=simulated_test_data)
    shuffle_dataset = True
    test_split = 0.0

    dataset_size = len(pose_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(1)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(indices)

    train_loader = torch.utils.data.DataLoader(pose_dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    model = RegressionNet().cuda()
    #model = torch.load("models/regression-model50").cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    with open("regressionresults.csv", mode='a') as file:
        results_writer = csv.writer(file, delimiter=',')
        results_writer.writerow(["epoch", "train_loss", "test_loss", "Avg Orientation Error", "Avg Angle Error", "Total Avg Angle Error"])

        for epoch in range(1, epochs+1):
            train_loss = train(model, train_loader, optimizer, loss_func)
            row = [epoch, train_loss]
            test_loss = test(model, test_loader, loss_func, results_writer, row)
            print("Train loss: ", train_loss, " Test loss: ",test_loss )


if __name__ == '__main__':
    run_regression(simulated_test_data=False)
    run_regression(simulated_test_data=True)
