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
from PlaneData import *
from torch.utils.data.sampler import SubsetRandomSampler


img_transform = transforms.Compose([
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def convert_to_degrees(sin_cos):
    return np.arctan2(sin_cos[:,0], sin_cos[:,1])*180.0/math.pi


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


class BinnedNet(nn.Module):

    def __init__(self):
        super(BinnedNet, self).__init__()

        self.net = nn.Sequential(
            DownConvLayer(3, 24, coord=False),
            DownConvLayer(24, 48),
            DownConvLayer(48, 24),
            DownConvLayer(24, 16))

        self.fc1 = nn.Linear(16*8*8,200)
        self.act1 = nn.ReLU()

        # TODO How many output nodes?
        # TODO Also we need a softmax for classification right?
        self.fc2 = nn.Linear(200, 180)

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


def test(model, test_loader, loss_func, should_print=False):
    model.eval()
    total_loss = 0
    num = 0
    err = np.zeros(9)
    with torch.no_grad():

        angle_err = np.zeros(3)

        for data, target in test_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            total_loss += loss_func(output, target)
            num += 1

            err += np.abs(target.cpu().numpy()[0] - output.cpu().numpy()[0])
            pred_angles = rotationMatrixToEulerAngles(output)
            target_angles = rotationMatrixToEulerAngles(target)
            angle_err += np.abs(pred_angles * 180 / math.pi - target_angles * 180 / math.pi)

        if should_print:

            #print("Avg Orientation err: ", err / num)
            print("Total avg Err: ", sum(err) / (num * 9))
            print("Avg Angle Error: ", sum(angle_err) / (3*num))

        return total_loss.item()/num


def run_bimodal():
    batch_size = 48
    lr = 1e-4
    epochs = 150

    image_dir = "data/ThousandSet"
    pose_dataset = SinglePoseDataset(image_dir, img_transform)
    shuffle_dataset = True
    test_split = 0.2

    dataset_size = len(pose_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(1)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(pose_dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(pose_dataset, batch_size=1,
                                                    sampler=test_sampler)

    model = BinnedNet().cuda()
    #model = torch.load("models/bimodal-EulerLarge-model").cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    ## This will need to be changed...
    loss_func = nn.MSELoss()

    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, optimizer, loss_func)
        test_loss = test(model, test_loader, loss_func)
        print("Epoch ", epoch, " Train Loss: " + str(train_loss), " Test Loss: " + str(test_loss))

        if epoch % 5 is 0:
            torch.save(model, "models/regression-model")
            test_loss = test(model, test_loader, loss_func, should_print=True)


if __name__ == '__main__':
    run_bimodal()
