# Uses this repo https://github.com/amdegroot/ssd.pytorch


import os
import sys
module_path = os.path.abspath(os.path.join('singleshotdetection'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PlaneData import *
from torchvision import transforms
from data import VOC_CLASSES as labels
from ssd import build_ssd



def detect():
    img_transform = transforms.Compose([
        transforms.CenterCrop((1000,1000)),
        transforms.Resize((300,300))
        ])

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')


    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_weights('singleshotdetection/weights/ssd300_mAP_77.43_v2.pth')

    image_dir = "./data/April18Exp/test"

    #pose_dataset = SinglePoseDataset(image_dir, img_transform=img_transform, im_name="/images")
    pose_dataset = SinglePoseDataset(image_dir, img_transform=img_transform, im_name="/")


    for i in range(5, 100):
        img, label = pose_dataset[i]
        original_img = np.array(img.copy())

        img = np.array(img, dtype=np.float32)
        img = img - (104.0, 117.0, 123.0)
        img = img.astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)

        xx = Variable(img.unsqueeze(0))     # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = net(xx)


        rgb_image = original_img

        top_k=10

        plt.figure(figsize=(10,10))
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.imshow(rgb_image)  # plot the image for matplotlib
        currentAxis = plt.gca()

        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.1:
                score = detections[0,i,j,0]
                label_name = labels[i-1]
                display_txt = '%s: %.2f'%(label_name, score)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                color = colors[i]
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
                j+=1

        plt.show()

def train_sse():
    pass


if __name__ == '__main__':
    detect()
