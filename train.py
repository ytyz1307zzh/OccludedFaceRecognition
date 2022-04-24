from getopt import gnu_getopt
import itertools
from tqdm import tqdm
import torch
import numpy as np
import os
from time import time
import torch.nn.functional as F
import argparse
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import Vggface
from resnet import ResNet
from model import Baseline


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', default='./split')
    parser.add_argument('-save_dir', default='./weights')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #std=[0.229, 0.224, 0.225])
    ])

    # Load training data 
    train_ds = Vggface(os.path.join(args.data_dir, "train.txt"), transform=transform)
    train_dl = DataLoader(train_ds, batch_size=16, num_workers=2, shuffle=True, drop_last=False)

    # Load validating data
    validate_ds = Vggface(os.path.join(args.data_dir, "validate.txt"), transform=transform)
    validate_dl = DataLoader(validate_ds, batch_size=16, num_workers=2, shuffle=False, drop_last=False)

    # Set loss function
    criterion = torch.nn.CrossEntropyLoss()

    subject2class = train_ds.subject2class
    num_classes = len(subject2class)
    # Model 
    model = Baseline(num_classes=num_classes, strict=False, verbose=True)
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #    print(f"Model will use {torch.cuda.device_count()} GPUs!")
    #    model = torch.nn.DataParallel(model)

    model.to(device).train()

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    os.makedirs(args.save_dir, exist_ok=True)

    running_error = []
    val_acc = None
    for epoch in range(10):
        model.train()
        
        for batch_num, (img, gt) in enumerate(train_dl):
            optim.zero_grad()
            img = img.to(device)
            gt = gt.to(device)
            #print(img)
            y = model(img)

            loss = criterion(y, gt)
            loss.backward()
            
            optim.step()
            
            running_error.append(float(loss.cpu().data.numpy()))
            running_error_display = np.mean(running_error[-100:])
            desc = 'Epoch: {} Batch: {} Training Loss: {} Validation Accuracy: {}'.format(
                epoch, batch_num, running_error_display, val_acc
            )
            print(desc)

        # Save model
        save_path = os.path.join(args.save_dir, '{}.pth'.format(epoch))
        torch.save(model, save_path)

        model.eval()
        valid_losses = []
        correct = 0
        with torch.no_grad():
            for batch_num, (img, gt) in enumerate(validate_dl):
                img = img.to(device)
                gt = gt.to(device)
                y = model(img)
                loss = criterion(y, gt)
                valid_losses.append(float(loss.cpu().data.numpy()))

                pred = torch.argmax(y, dim=1)
                correct += torch.sum(torch.tensor(pred) == torch.tensor(gt))

        val_acc = correct / len(validate_ds)

        desc = 'Epoch: {} Validation Loss: {} Validation Accuracy: {}'.format(epoch, np.mean(valid_losses), val_acc)
        print(desc)
         

if __name__ == '__main__':
    train()
