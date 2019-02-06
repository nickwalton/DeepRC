import torch.nn as nn
from PlaneData import *
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import math
import scipy.misc


img_transform = transforms.Compose([
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor()
])


class DownConvLayer(nn.Module):
    def __init__(self, input_depth, output_depth, activation=nn.ReLU):
        super(DownConvLayer, self).__init__()
        self.conv_layer = nn.Conv2d(input_depth, output_depth, kernel_size=5, stride=2, padding=2)
        self.activation = activation()

    def forward(self, x):
        return self.activation(self.conv_layer(x))


class UpConvLayer(nn.Module):
    def __init__(self, input_depth, output_depth, activation=nn.ReLU):
        super(UpConvLayer, self).__init__()
        self.conv_layer = nn.ConvTranspose2d(input_depth, output_depth, kernel_size=5, stride=2,
                                             padding=2, output_padding=1)
        self.activation = activation()

    def forward(self, x):
        result = self.conv_layer(x)
        return self.activation(result)


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            DownConvLayer(3, 128),
            DownConvLayer(128, 256),
            DownConvLayer(256, 256),
            DownConvLayer(256, 512))

        self.decoder = nn.Sequential(
            UpConvLayer(512, 256),
            UpConvLayer(256, 256),
            UpConvLayer(256, 128),
            UpConvLayer(128, 3, activation=nn.Sigmoid)
        )

        self.encode_fc = nn.Linear(512*8*8, 128)
        self.decode_fc = nn.Linear(128, 512*8*8)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        z = self.encode_fc(x)
        return z

    def decode(self, z):
        z = self.decode_fc(z)
        z = self.decoder(z.reshape((z.shape[0], 512, 8, 8)))
        return z


class CodeBook(object):

    def __init__(self, model, dataset):

        self.model = model.cpu()

        self.z_vecs = []
        self.rots = []

        for i in range(len(dataset)):
            sample = dataset[i]
            z = self.model.encode(sample[0].unsqueeze(0))
            self.z_vecs.append(z[0])
            self.rots.append(sample[1])

        print("Codebook Finished")

    def predict(self, image):
        z = self.model.encode(image.unsqueeze(0)).squeeze(0)

        closest_dist = -1
        closest_ind = -1

        for i, vec in enumerate(self.z_vecs):
            dist = nn.functional.cosine_similarity(z, vec, dim=0)
            if dist > closest_dist:
                closest_dist = dist
                closest_ind = i

        return closest_ind, self.rots[closest_ind]


def train(model, train_loader, optimizer, loss_func, epoch):
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

    print("Epoch" + str(epoch) + " Training Loss: " + str(total_loss.item()))


def test(model, test_loader, loss_func, epoch):
    model.eval()
    total_loss = 0
    num = 0
    display = True
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            total_loss += loss_func(output, target)
            num += output.shape[0]

            if display:
                scipy.misc.imsave('test_imgs/test' + str(total_loss.cpu().detach().numpy()) + 'output.jpg',
                                  np.swapaxes(output[0].cpu().detach().numpy(), 0, 2))
                scipy.misc.imsave('test_imgs/test' + str(total_loss.cpu().detach().numpy()) + 'input.jpg',
                                  np.swapaxes(data[0].cpu().detach().numpy(), 0, 2))
                display = False

        print("Epoch" + str(epoch) + " Test Loss:     " + str(total_loss.item()))


def display_img(img):
    plt.imshow(np.swapaxes(img.cpu().detach().numpy(), 0, 2))


def ae_train():
    batch_size = 128
    epochs = 1000
    lr = 1e-3

    csv_path = "data/plane_data1/orientations.csv"
    image_dir = "data/plane_data1"
    pose_dataset = SinglePoseDataset(image_dir, csv_path, img_transform, label="image")

    ex_img = pose_dataset[0][0]
    plt.imshow(np.swapaxes(ex_img.numpy(), 0, 2))

    train_loader = DataLoader(pose_dataset, batch_size=batch_size, shuffle=True)
    model = VAE().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, loss_func, epoch)
        test(model, train_loader, loss_func, epoch)

        if epoch % 25 is 0:
            torch.save(model, "models/AE-model" + str(epoch))


def ae_test():
    csv_path = "data/plane_data1/orientations.csv"
    image_dir = "data/plane_data1"
    pose_dataset = SinglePoseDataset(image_dir, csv_path, img_transform, len=500)

    model = torch.load("models/AE-model300")
    codebook = CodeBook(model, pose_dataset)

    for i in range(501,800):
        sample = pose_dataset[i]
        img = sample[0]
        orientation = sample[1]
        ind, pred = codebook.predict(img)

        pred_pic = pose_dataset[ind][0]

        scipy.misc.imsave('test_imgs/test' + str(i) + 'output.jpg',
                          np.swapaxes(img.numpy(), 0, 2))
        scipy.misc.imsave('test_imgs/test' + str(i) + 'input.jpg',
                          np.swapaxes(pred_pic.numpy(), 0, 2))


        print(orientation)
        print(pred)


if __name__ == '__main__':
    ae_test()


