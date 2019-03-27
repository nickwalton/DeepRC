import torch.nn as nn
from PlaneData import *
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import math
from scipy.misc import imsave
from tqdm import tqdm
import bisect
from torch.utils.data.sampler import SubsetRandomSampler

img_transform = transforms.Compose([
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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


def rotationMatrixToEulerAngles(R):
    R = R.reshape((3,3))

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


def convert_to_degrees(sin_cos):
    return np.arctan2(sin_cos[0,:,0], sin_cos[0,:,1])*180.0/math.pi


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            DownConvLayer(3, 16),
            DownConvLayer(16, 32),
            DownConvLayer(32, 64),
            DownConvLayer(64, 128))

        self.decoder = nn.Sequential(
            UpConvLayer(128, 64),
            UpConvLayer(64, 32),
            UpConvLayer(32, 16),
            UpConvLayer(16, 3, activation=nn.Tanh)
        )

        self.encode_fc = nn.Linear(128*8*8, 128)
        self.decode_fc = nn.Linear(128, 128*8*8)

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
        z = self.decoder(z.reshape((z.shape[0], 128, 8, 8)))
        return z


class CodeBook(object):

    def __init__(self, model, dataloader):

        self.model = model.cpu()

        self.z_vecs = []
        self.images = []
        self.rots = []
        self.indices = []

        for batch_idx, (data, target) in enumerate(dataloader):
            sample = data
            self.images.append(data)
            z = self.model.encode(sample)
            self.z_vecs.append(z[0])
            self.rots.append(target)

        print("Codebook Finished")

    def predict(self, image, n=1, just_top=True, get_img=False):
        z = self.model.encode(image).squeeze(0)

        top_dist_list = [(-1, -math.inf)]*n

        for i, vec in enumerate(self.z_vecs):
            dist = nn.functional.cosine_similarity(z, vec, dim=0).cpu().detach().numpy()

            # insert into proper place in top_dist_list
            for j in range(n):
                if dist > top_dist_list[j][1]:
                    top_dist_list.insert(j, (i, dist))
                    top_dist_list.pop()
                    break

        if get_img:
            return self.images[top_dist_list[0][0]], self.rots[top_dist_list[0][0]]
        else:
            return self.rots[top_dist_list[0][0]]


def train(model, train_loader, optimizer, loss_func):
    model.train()
    num = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("cuda"), target.to("cuda")
        optimizer.zero_grad()
        output = model(data)
        num += target.shape[0]
        loss = loss_func(output, data)
        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss.item()


def display_img(img):
    plt.imshow(np.swapaxes(img.cpu().detach().numpy(), 0, 2))


def test(model, test_loader, loss_func):
    model.eval()
    total_loss = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            total_loss += loss_func(output, data)
            num += output.shape[0]
        return total_loss.item()/num


def run_codebook():
    batch_size = 128
    lr = 6e-4
    epochs = 20
    model = AutoEncoder().cuda()
    #model = torch.load("models/ae-model")
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
    test_loader = torch.utils.data.DataLoader(pose_dataset, batch_size=1, sampler=test_sampler)

    codebook_loader = torch.utils.data.DataLoader(pose_dataset, batch_size=1,
                                                    sampler=train_sampler)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    for epoch in range(1, epochs+1):
        train(model, train_loader, optimizer, loss_func)
        train_loss = test(model, train_loader, loss_func)
        print("Epoch ", epoch, " Train Loss: " + str(train_loss))
        test_loss = test(model, test_loader, loss_func)
        print("Epoch ", epoch, " Test Loss: " + str(test_loss))

        if epoch % 5 is 0:
            torch.save(model, "models/ae-model-8k")

    codebook = CodeBook(model, codebook_loader)

    err = np.zeros(9)
    num = 0
    for batch_idx, (data, target) in enumerate(test_loader):

        closest_img, pred = codebook.predict(data, get_img=True)
        imsave("test_imgs/oneaxis/im" + str(num) + "closest.jpg", np.swapaxes(closest_img[0], 0, 2))
        imsave("test_imgs/oneaxis/im" + str(num) + "original.jpg", np.swapaxes(data[0], 0, 2))
        err += np.abs(target.cpu().numpy()[0] - pred.cpu().numpy()[0])

        #print("Orient Diff " + str(num), " ", target.cpu().numpy()[0] - pred.cpu().numpy()[0])
        pred_angles = rotationMatrixToEulerAngles(pred)
        target_angles = rotationMatrixToEulerAngles(target)
        print("Pred: ", pred_angles*180/math.pi, " Target: ", target_angles*180/math.pi)

        num += 1

    print("Avg err: ", err/num)
    print("Total avg Err: ", sum(err)/(num*9))


if __name__ == '__main__':
    run_codebook()

