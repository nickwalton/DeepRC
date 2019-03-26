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


class BiModalNet(nn.Module):

    def __init__(self):
        super(BiModalNet, self).__init__()

        self.net = nn.Sequential(
            DownConvLayer(3, 8, coord=True),
            DownConvLayer(8, 16),
            DownConvLayer(16, 8),
            DownConvLayer(8, 4))

        self.fc1 = nn.Linear(4*8*8, 64)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)

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


def test(model, test_loader, loss_func, should_print=False):
    model.eval()
    total_loss = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            total_loss += loss_func(output, target)
            num += output.shape[0]

            if should_print:
                print("Targ raw ", target[0].cpu().numpy().flatten())
                print("Pred raw ", output[0].cpu().numpy().flatten())

                #print("Targ deg ", target_deg)
                #print("Pred deg ", pred_deg)

        return total_loss.item()/num


def run_bimodal():
    batch_size = 16
    lr = 3e-4
    epochs = 30

    image_dir = "data/OneAxis"
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
    test_loader = torch.utils.data.DataLoader(pose_dataset, batch_size=batch_size,
                                                    sampler=test_sampler)

    model = BiModalNet().cuda()
    #model = torch.load("models/bimodal-EulerLarge-model").cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    for epoch in range(1, epochs+1):
        train(model, train_loader, optimizer, loss_func)
        train_loss = test(model, train_loader, loss_func)
        test_loss = test(model, test_loader, loss_func)
        print("Epoch ", epoch, " Train Loss: " + str(train_loss), " Test Loss: " + str(test_loss))

        if epoch % 10 is 0:
            torch.save(model, "models/bimodal-EulerLarge-model")
            test_loss = test(model, test_loader, loss_func, should_print=True)


if __name__ == '__main__':
    run_bimodal()
