import torch.nn as nn
from PlaneData import *
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import math
import scipy.misc
from tqdm import tqdm
import bisect

img_transform = transforms.Compose([
    #transforms.CenterCrop((128, 128)),
    transforms.Resize((128,128)),
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

        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            z = self.model.encode(sample[0].unsqueeze(0))
            self.z_vecs.append(z[0])
            self.rots.append(sample[1])

        print("Codebook Finished")

    def _sort_by_dist(self, val):
        return val[1]

    def predict(self, image, n=1):
        z = self.model.encode(image.unsqueeze(0)).squeeze(0)

        top_dist_list = [(-1, -math.inf)]*n

        for i, vec in enumerate(self.z_vecs):
            dist = nn.functional.cosine_similarity(z, vec, dim=0).cpu().detach().numpy()

            # insert into proper place in top_dist_list
            for j in range(n):
                if dist > top_dist_list[j][1]:
                    top_dist_list.insert(j, (i, dist))
                    top_dist_list.pop()
                    break

        return top_dist_list


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

    return total_loss.item()


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


def ae_train(image_dir):
    batch_size = 256
    epochs = 1000
    lr = 1e-3

    csv_path = image_dir + "/orientations.csv"
    pose_dataset = SinglePoseDataset(image_dir, csv_path, img_transform, label="image")

    ex_img = pose_dataset[0][0]
    plt.imshow(np.swapaxes(ex_img.numpy(), 0, 2))

    train_loader = DataLoader(pose_dataset, batch_size=batch_size, shuffle=True)
    model = VAE().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    for epoch in range(0, epochs + 1):
        res = train(model, train_loader, optimizer, loss_func, epoch)
        #test(model, train_loader, loss_func, epoch)

        if epoch % 10 is 0:
            torch.save(model, "models/LRG_AE-model" + str(res))


def ae_test(codebook_img_dir, sample_img_dir, test_img_folder):

    csv_path = codebook_img_dir + "/orientations.csv"

    codebook_dataset = SinglePoseDataset(codebook_img_dir, csv_path, img_transform)

    sample_csv_path = sample_img_dir + "/orientations.csv"
    sample_dataset = SinglePoseDataset(sample_img_dir, sample_csv_path, img_transform)

    model = torch.load("models/AE-model300")
    print("Building codebook")
    codebook = CodeBook(model, codebook_dataset)

    print("Testing Images")
    for i in tqdm(range(0, 100)):
        sample = sample_dataset[i]
        img = sample[0]
        orientation = sample[1]
        preds = codebook.predict(img, n=4)

        pred_indices = [pred[0] for pred in preds]
        pred_dists = [pred[1] for pred in preds]

        #scipy.misc.imsave(test_img_folder + str(i) + 'input.jpg',
        #                  np.swapaxes(img.numpy(), 0, 2))

        #for j in range(4):
        #    pred_pic = codebook_dataset[pred_indices[j]][0]

        #    scipy.misc.imsave(test_img_folder + str(i) + str("-") + str(j) + "dist" + str(pred_dists[j]) + 'output.jpg',
        #                      np.swapaxes(pred_pic.numpy(), 0, 2))


if __name__ == '__main__':
    #codebook_img_dir = "data/large_uniform_without_sky"
    #sample_img_dir = "data/uniform_with_sky"
    #test_img_folder = 'test_imgs/largecodebooktest/'
    #ae_test(codebook_img_dir, sample_img_dir, test_img_folder)

    # TODO change data gathering framework so that the relative camera position and plane position is the same every time. 

    image_dir = "data/large_uniform_without_sky"
    ae_train(image_dir)

