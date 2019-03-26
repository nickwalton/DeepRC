from DenseNet import DenseNet
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from holodeck.environments import *
from holodeck import agents
import cv2
from holodeck.sensors import Sensors
from PIL import Image

from PlaneData import *

img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


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

    csv_path = "data/plane_data1/orientations.csv"
    image_dir = "data/plane_data1"
    pose_dataset = SinglePoseDataset(image_dir, csv_path, img_transform)

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

        torch.save(model, "models/single-axis-model")


# TODO test that this is the same after as the dataset method.
def prep_image(image):
    image = Image.fromarray(image)
    image = torch.unsqueeze(img_transform(image), 0)
    return image

def draw_axes(env, start_loc, orientation):
    end_loc = start_loc + 10 * orientation[0:3]
    env.draw_arrow(start_loc, end_loc, [255, 0, 0], 10)
    end_loc = start_loc + 10 * orientation[3:6]
    env.draw_arrow(start_loc, end_loc, [0, 255, 0], 10)
    end_loc = start_loc + 10 * orientation[6:9]
    env.draw_arrow(start_loc, end_loc, [0, 0, 255], 10)


def predict_pose():

    model = torch.load('models/single-pose-model')

    sensors = [Sensors.VIEWPORT_CAPTURE, Sensors.LOCATION_SENSOR, Sensors.VELOCITY_SENSOR, Sensors.ORIENTATION_SENSOR]
    agent = AgentDefinition("uav0", agents.UavAgent, sensors)
    cam_sensor = [Sensors.RGB_CAMERA]
    cam_agent = AgentDefinition("sphere0", agents.ContinuousSphereAgent, cam_sensor)
    env = HolodeckEnvironment([agent, cam_agent], start_world=False, camera_height=512, camera_width=512)
    env.agents["uav0"].set_control_scheme(1)

    for _ in range(10):
        env.reset()
        states = env.tick()
        uav_state = states["uav0"]
        image = states["sphere0"][Sensors.RGB_CAMERA][:, :, 0:3]
        image = prep_image(image)

        # TODO retrain network with euler angles instead of axes
        pred_orientation = model(image.cuda()).cpu().detach().numpy().squeeze(0)
        true_orientation = uav_state[Sensors.ORIENTATION_SENSOR][:].flatten()

        diff = true_orientation - pred_orientation
        orientation = pred_orientation

        for _ in range(1000):
            start_loc = uav_state[Sensors.LOCATION_SENSOR][:]
            draw_axes(env, start_loc, orientation)
            env.tick()


if __name__ == '__main__':
    train_pose()












