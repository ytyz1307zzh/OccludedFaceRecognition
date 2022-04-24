from tqdm import tqdm
import torch
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


def validate(img_list_path="validate.txt"):
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


    correct = 0
    with torch.no_grad():
        for batch_num, (img, gt) in enumerate(validate_dl):
            
            img = img.to(device)
            gt = gt.to(device)
            #print(img)
            y = model(img)
            pred = torch.argmax(y, dim=1)
            correct += torch.sum(torch.tensor(pred) == torch.tensor(gt)).item()

    val_acc = correct / len(validate_ds)
    print(val_acc)
         

if __name__ == '__main__':
    import sys
    validate(sys.argv[1])
