import torch.nn as nn
from PlaneData import *
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import math
from scipy.misc import imsave
from tqdm import tqdm
from PlaneData import rotationMatrixToEulerAngles
import os
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


class MixedDataset(Dataset):

    def __init__(self, img_transform=None, test=False, ten_k=True):
        if ten_k:
            self.blank_img_dir = "data/April18Exp/blank10k/images/"
            self.back_img_dir = "data/April18Exp/background10k/images/"
            csv_blank = "data/April18Exp/blank10k/orientations.csv"
        else:
            self.blank_img_dir = "data/April18Exp/blank2k/images/"
            self.back_img_dir = "data/April18Exp/background2k/images/"
            csv_blank = "data/April18Exp/blank2k/orientations.csv"

        self.data = pd.read_csv(csv_blank)
        if img_transform is None:
            self.transform = transforms.Compose([])
        else:
            self.transform = img_transform

        self.len = len(os.listdir(self.blank_img_dir))

    def __len__(self):
        return self.len-1

    def __getitem__(self, idx):
        angles = self.data.iloc[idx, 0:].values.astype(dtype=np.float32)

        blank_img_path = self.blank_img_dir + "im" + str(idx) + ".jpg"
        back_img_path = self.back_img_dir + "im" + str(idx) + ".jpg"

        blank_orig_image = Image.open(blank_img_path)
        blank_image = self.transform(blank_orig_image)

        back_orig_image = Image.open(back_img_path)
        back_image = self.transform(back_orig_image)

        sample = (back_image, blank_image, angles[9:12])

        return sample


class RealTestDataset(Dataset):

    def __init__(self, custom_len=None, euler=False, im_name="/im"):

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

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

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
        image = self.transform(orig_image)

        sample = (image, angles)

        return sample


def convert_to_degrees(sin_cos):
    return np.arctan2(sin_cos[0,:,0], sin_cos[0,:,1])*180.0/math.pi


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.z_length = 64
        self.largest_filter = 128

        self.encoder = nn.Sequential(
            DownConvLayer(3, 16),
            DownConvLayer(16, 32),
            DownConvLayer(32, 48),
            DownConvLayer(48, self.largest_filter))

        self.decoder = nn.Sequential(
            UpConvLayer(self.largest_filter, 48),
            UpConvLayer(48, 32),
            UpConvLayer(32, 16),
            UpConvLayer(16, 3, activation=nn.Tanh)
        )

        self.encode_fc = nn.Linear(self.largest_filter*8*8, self.z_length)
        self.decode_fc = nn.Linear(self.z_length, self.largest_filter*8*8)

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
        z = self.decoder(z.reshape((z.shape[0], self.largest_filter, 8, 8)))
        return z


class CodeBook(object):

    def __init__(self, model, dataloader):

        self.model = model.cpu()

        self.z_vecs = []
        self.images = []
        self.rots = []
        self.indices = []

        for batch_idx, (_, data, target) in enumerate(dataloader):
            sample = data
            self.images.append(data)
            z = self.model.encode(sample)
            self.z_vecs.append(z[0])
            self.rots.append(target)

        print("Codebook Finished")

    def predict(self, image, n=5, just_top=False, get_img=False):
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

        if just_top:
            if get_img:
                return self.images[top_dist_list[0][0]], self.rots[top_dist_list[0][0]]
            else:
                return self.rots[top_dist_list[0][0]]
        else:
            return [self.images[top_dist_list[i][0]] for i in range(n)], [self.rots[top_dist_list[i][0]] for i in range(n)]


def train(model, train_loader, optimizer, loss_func):
    model.train()
    num = 0
    total_loss = 0

    for batch_idx, (data, target, angles) in enumerate(train_loader):
        data, target, angles = data.to("cuda"), target.to("cuda"), angles.to("cuda")
        optimizer.zero_grad()
        output = model(data)
        num += angles.shape[0]
        loss = loss_func(output, target)
        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss.item()/num


def display_img(img):
    plt.imshow(np.swapaxes(img.cpu().detach().numpy(), 0, 2))


def test(model, test_loader, loss_func):
    model.eval()
    total_loss = 0
    num = 0
    with torch.no_grad():
        for data, target, angles in test_loader:
            data, target, angles = data.to("cuda"), target.to("cuda"), angles.to("cuda")
            output = model(data)
            total_loss += loss_func(output, target)
            num += angles.shape[0]
        return total_loss.item()/num


def run_codebook():

    if not os.path.exists("test_imgs/codebook_test"):
        os.makedirs("test_imgs/codebook_test")

    # Train AutoEncoder
    batch_size = 64
    lr = 2e-3
    epochs = 5
    model = AutoEncoder().cuda()
    #model = torch.load("models/codebook-model30")
    pose_dataset = MixedDataset(img_transform=img_transform)
    test_dataset = MixedDataset(img_transform=img_transform, ten_k=False)

    shuffle_dataset = True
    test_split = 0.0
    dataset_size = len(pose_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(1)
        np.random.shuffle(indices)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(indices)

    train_loader = torch.utils.data.DataLoader(pose_dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    codebook_loader = torch.utils.data.DataLoader(pose_dataset, batch_size=1,
                                                    sampler=train_sampler)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    with open("codebookresults2.csv", mode='w') as file:
        results_writer = csv.writer(file, delimiter=',')
        results_writer.writerow(["epoch", "train_loss", "test_loss"])

        for epoch in range(1, epochs+1):
            train(model, train_loader, optimizer, loss_func)
            train_loss = test(model, train_loader, loss_func)
            test_loss = test(model, test_loader, loss_func)
            print("Epoch ", epoch, " Train Loss: " + "%.8f" % train_loss,"Epoch ", epoch, " Test  Loss: " + "%.8f" % test_loss)

            results_writer.writerow([epoch, train_loss, test_loss])

            if epoch % 5 is 0:
                torch.save(model, "models/codebook-model" + str(epoch))


def test_codebook(simulated=False):

    if simulated:
        im_save_folder = "codebook_test"
    else:
        im_save_folder = "codebook_real"

    if not os.path.exists("test_imgs/" + im_save_folder):
        os.makedirs("test_imgs/" + im_save_folder)

    model = torch.load("models/codebook-model5")
    pose_dataset = MixedDataset(img_transform=img_transform)
    if simulated:
        test_dataset = MixedDataset(img_transform=img_transform, ten_k=False)
    else:
        test_dataset = RealTestDataset()

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    codebook_loader = torch.utils.data.DataLoader(pose_dataset, batch_size=1)

    with open("codebookresults.csv", mode='a') as file:
        results_writer = csv.writer(file, delimiter=',')
        results_writer.writerow(["epoch", "train_loss", "test_loss"])

        # Set up codebook and test
        codebook = CodeBook(model, codebook_loader)

        angle_err = np.zeros(3)
        num = 0
        for batch_idx, (data, target) in enumerate(test_loader):

            closest_imgs, preds = codebook.predict(data, get_img=True)

            closest_img = closest_imgs[0]
            pred = preds[0]

            if num < 50:
                for i in range(5):
                    imsave("test_imgs/" + im_save_folder + "/im" + str(num) + "closest" + str(i) + ".jpg",
                           np.swapaxes(closest_imgs[i][0], 0, 2))
                    imsave("test_imgs/" + im_save_folder + "/im" + str(num) + "original.jpg", np.swapaxes(data[0], 0, 2))

            pred_angles = pred.squeeze(0).cpu().numpy()
            target_angles = target.squeeze(0).cpu().numpy()
            err = np.abs(pred_angles - target_angles)
            alt_err = np.abs(err - 360)

            for i in range(3):
                if alt_err[i] < err[i]:
                    err[i] = alt_err[i]

            angle_err += err

            print("Pred: ", pred_angles, " Target: ", target_angles)

            num += 1

        print("Avg Angle Error: ", angle_err / num)


if __name__ == '__main__':
    #run_codebook()
    #test_codebook(simulated=False)
    test_codebook(simulated=True)
