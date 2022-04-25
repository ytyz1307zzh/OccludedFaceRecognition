import pdb
from getopt import gnu_getopt
import itertools

import torchvision.models
import pickle
from tqdm import tqdm
import torch
import numpy as np
import json
import os
from time import time
import torch.nn.functional as F
import argparse
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FaceDataset
from resnet import ResNet
from model import Model


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data', default='./split/train.txt')
    parser.add_argument('-valid_data', default='./split/validate.txt')
    parser.add_argument('-subject2class', default='./split/subject2class.json',
                        help="the mapping from subject names to class labels")
    parser.add_argument('-train_from_scratch', default=False, action='store_true',
                        help='if `True`, then train from scratch and ignore the pretrained resnet model')
    parser.add_argument('-resnet', default='./resnet18_weights.pkl', help='pretrained resnet18 model')
    parser.add_argument('-save_dir', default='./weights')
    parser.add_argument('-lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('-epochs', default=10, type=int, help='training epochs')
    parser.add_argument('-batch', default=16, type=int, help="batch size")
    parser.add_argument('-wait_steps', default=5, type=int, help="early stopping wait steps")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #std=[0.229, 0.224, 0.225])
    ])

    subject2class = json.load(open(args.subject2class, 'r', encoding='utf8'))

    # Load training data 
    train_ds = FaceDataset(args.train_data, subject2class=subject2class, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=args.batch, num_workers=0, shuffle=True, drop_last=False)

    # Load validating data
    validate_ds = FaceDataset(args.valid_data, subject2class=subject2class, transform=transform)
    validate_dl = DataLoader(validate_ds, batch_size=args.batch, num_workers=0, shuffle=False, drop_last=False)

    # Set loss function
    criterion = torch.nn.CrossEntropyLoss()

    subject2class = train_ds.subject2class
    num_classes = len(subject2class)
    # Model 
    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    if not args.train_from_scratch:
        state_dict = pickle.load(open(args.resnet, 'rb'))
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        for key, value in state_dict.items():
            state_dict[key] = torch.tensor(value)
        model.load_state_dict(state_dict, strict=False)
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #    print(f"Model will use {torch.cuda.device_count()} GPUs!")
    #    model = torch.nn.DataParallel(model)

    model.to(device).train()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    running_error = []
    first_batch = True
    best_acc = -1
    wait_steps = 0
    best_epoch = None

    for epoch in tqdm(range(args.epochs), desc="Training"):
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
            # running_error_display = np.mean(running_error[-100:])
            if first_batch:
                desc = f'Epoch: 0 Training Loss: {loss.item()}'
                print(desc)
                first_batch = False

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
                correct += torch.sum(pred == gt)

        val_acc = correct / len(validate_ds)

        desc = f'Epoch: {epoch + 1} Train Loss: {np.mean(running_error)} Validation Loss: {np.mean(valid_losses)} ' \
               f'Validation Accuracy: {val_acc * 100:.2f}%'
        print(desc)
        running_error.clear()

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            print(f'New best accuracy: {best_acc * 100:.2f}%')
            # Save the best model checkpoint
            save_path = os.path.join(args.save_dir, 'best.pth')
            torch.save(model, save_path)
            wait_steps = 0
        else:
            wait_steps += 1
            print(f'Did not beat best accuracy! Current: {val_acc * 100:.2f}% Best: {best_acc * 100:.2f}%')
            if wait_steps == args.wait_steps:
                print(f'Early stopping! Best accuracy: {best_acc * 100:.2f}% at epoch {best_epoch}')
                break
         

if __name__ == '__main__':
    train()
