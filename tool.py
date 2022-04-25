from tqdm import tqdm
import torch
import json
from time import time
import numpy as np
import torch.nn.functional as F
import argparse
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FaceDataset
from resnet import ResNet
from model import Model
import matplotlib.pyplot as plt
import os


def cos_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def match(feature, labels):
    authentic = []
    impostor = []
    n = len(feature)
    for i in range(n):
        for j in range(i+1, n):
            curr = cos_similarity(feature[i, :], feature[j, :])
            if labels[i] == labels[j]:
                authentic.append(curr)
            else:
                impostor.append(curr)

    return authentic, impostor


def tool(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #std=[0.229, 0.224, 0.225])
    ])

    subject2class = json.load(open(args.subject2class, 'r', encoding='utf8'))

    # Load validating data
    validate_ds = FaceDataset(args.data, subject2class=subject2class, transform=transform)
    validate_dl = DataLoader(validate_ds, batch_size=16, num_workers=0, shuffle=False, drop_last=False)

    # Model 
    model = torch.load(args.checkpoint)
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device).eval()


    feature = None
    labels = None
    with torch.no_grad():
        for batch_num, (img, gt) in enumerate(validate_dl):
            
            img = img.to(device)
            #print(img)
            y = model.predict(img)
            y = y.view((y.shape[0], -1))
            y = y.cpu().detach().numpy()
            
            if feature is None:
                feature = y
            else:
                feature = np.concatenate((feature, y), axis=0)

            if labels is None:
                labels = gt
            else:
                labels = np.concatenate((labels, gt), axis=0)
    
    authentic, impostor = match(feature, labels)

    plt.hist(
        authentic,
        bins="auto",
        histtype="step",
        density=True,
        label="Authentic",
        color="b",
        linewidth=1.5,
    )
    plt.hist(
        impostor,
        bins="auto",
        histtype="step",
        density=True,
        label="Impostor",
        color="r",
        linestyle="dashed",
        linewidth=1.5,
    )

    plt.ylabel("Relative Frequency")
    plt.xlabel("Match Scores")

    plt.savefig(args.output, dpi=150)
         

parser = argparse.ArgumentParser()
parser.add_argument('-data', default='./split/validate.txt')
parser.add_argument('-subject2class', default='./split/subject2class.json')
parser.add_argument('-checkpoint', default='./weights/best.pth')
parser.add_argument('-output', default='./validate.png')
args = parser.parse_args()
tool(args)
