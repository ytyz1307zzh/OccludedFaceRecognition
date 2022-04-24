from tqdm import tqdm
import torch
from time import time
import numpy as np
import torch.nn.functional as F
import argparse
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import Vggface
from resnet import ResNet
from model import Baseline
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

def tool(img_list_path="validate.txt"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #std=[0.229, 0.224, 0.225])
    ])

    # Load validating data
    validate_ds = Vggface(img_list_path, transform=transform)
    validate_dl = DataLoader(validate_ds, batch_size=16, num_workers=2, shuffle=False, drop_last=False)

    # Model 
    model = torch.load("weights/9.pth")
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

    plot_path = os.path.join(os.path.split(img_list_path)[1].split(".")[0]+".png")
    plt.savefig(plot_path, dpi=150)
         

if __name__ == '__main__':
    import sys
    tool(sys.argv[1])