from DenseNet import DenseNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from holodeck.environments import *
from holodeck import agents
from holodeck.environments import *
from holodeck.sensors import Sensors
from PIL import Image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_transform = transforms.Compose([
                             transforms.Grayscale(),
                             transforms.ToTensor(),
                             normalize,
                         ])

class SinglePoseDataset(Dataset):

    def __init__(self, img_dir, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.img_dir = img_dir

        self.images = datasets.ImageFolder(img_dir, img_transform)


    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        orientation = torch.tensor(self.data.iloc[idx, 1:].values.astype(dtype=np.float32))

        image = self.images[idx]
        sample = (image[0], orientation)

        return sample


def train(model, train_loader, optimizer, loss_func):
    model.train()
    num = 0
    total_loss = 0
    display = True

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("cuda"), target.to("cuda")
        optimizer.zero_grad()
        output = model(data)
        num += target.shape[0]
        loss = loss_func(output, target)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if display and batch_idx is 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')


    print("Training Loss: " + str(total_loss.item()))


def test(model, test_loader, loss_func):
    model.eval()
    total_loss = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            total_loss += loss_func(output, target)
            num += output.shape[0]
        print("Test Loss: " + str(total_loss.item()))


def train_pose():
    batch_size = 48
    lr = 1e-4
    epochs = 5

    csv_path = "plane_data1/orientations.csv"
    image_dir = "plane_data1"
    pose_dataset = SinglePoseDataset(image_dir, csv_path)

    input_size = pose_dataset[0][0].shape[1]
    output_size = 9

    """train_split = 0.7
    train_size = int(train_split * len(pose_dataset))
    test_size = len(pose_dataset) - train_size
    """

    train_loader = DataLoader(pose_dataset, batch_size=batch_size, shuffle=True)
    model = DenseNet(input_size=input_size, output_size=output_size, type="cont").cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    for epoch in range(1, epochs+1):
        train(model, train_loader, optimizer, loss_func)
        test(model, train_loader, loss_func)

        torch.save(model, "models/single-pose-model")

def predict_pose():

    model = torch.load('models/single-pose-model')

    sensors = [Sensors.VIEWPORT_CAPTURE, Sensors.LOCATION_SENSOR, Sensors.VELOCITY_SENSOR]
    agent = AgentDefinition("uav0", agents.UavAgent, sensors)
    env = HolodeckEnvironment(agent, start_world=False)
    env.agents["uav0"].set_control_scheme(1)
    command = [0, 0, 10, 50]

    for i in range(10):
        env.reset()
        for _ in range(1000):
            env.reset()
            state, reward, terminal, _ = env.step(command)
            orig_image = state[Sensors.VIEWPORT_CAPTURE][:, :, 0:3]
            image = Image.fromarray(orig_image)
            image = torch.unsqueeze(img_transform(image),0)
            orientation_pred = model(image.cuda()).cpu().detach().numpy().squeeze(0)
            or_arrow = -orientation_pred[0:3] *3000

            for _ in range(100):
                start_loc = [-180, 4010, 1110]
                end_loc = [-180, 4010, 1110]
                end_loc[0] += or_arrow[0]
                end_loc[1] += or_arrow[1]
                end_loc[2] += or_arrow[2]
                env.draw_arrow(start_loc, end_loc, [0, 0, 255], 10)
                env.tick()



if __name__ == '__main__':
    predict_pose()











