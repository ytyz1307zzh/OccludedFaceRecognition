from tqdm import tqdm
import torch
import os
from time import time
import torch.nn.functional as F
import argparse
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FaceDataset
import json


def validate():

    parser = argparse.ArgumentParser()
    parser.add_argument('-test_data', default="./split/test_mask.txt")
    parser.add_argument('-subject2class', defualt='./split/subject2class.json')
    parser.add_argument('-checkpoint', default="./weights/best.pth", help="Which checkpoint to load")
    parser.add_argument('-batch', default=16, type=int, help='batch size')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #std=[0.229, 0.224, 0.225])
    ])

    subject2class = json.load(open(args.subject2class, 'r', encoding='utf8'))

    # Load validating data
    validate_ds = FaceDataset(args.test_data, subject2class=subject2class, transform=transform)
    validate_dl = DataLoader(validate_ds, batch_size=args.batch, num_workers=0, shuffle=False, drop_last=False)

    # Model 
    model = torch.load(args.checkpoint)
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device).eval()

    correct = 0
    with torch.no_grad():
        for batch_num, (img, gt) in enumerate(validate_dl):
            
            img = img.to(device)
            gt = gt.to(device)
            #print(img)
            y = model(img)
            pred = torch.argmax(y, dim=1)
            correct += torch.sum(pred == gt).item()

    val_acc = correct / len(validate_ds)
    print(f"{val_acc * 100:.2f}%")
         

if __name__ == '__main__':
    validate()
