from tqdm import tqdm
import torch
import cv2

def get_subject(img_path):
    return img_path.split("/")[-2]


class Vggface:
    def __init__(self, img_list_path='./image_dir', transform=None):
        self.img_list = []
        with open(img_list_path) as f:
            for line in f:
                self.img_list.append(line.strip())

        self.subject2class = {}
        for img_path in self.img_list:
            subject = get_subject(img_path)
            if subject not in self.subject2class:
                self.subject2class[subject] = len(self.subject2class)

        self.transform = transform


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        img_path = self.img_list[i]
        subject = get_subject(img_path)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        if self.transform:
            img = self.transform(img)

        return img, self.subject2class[subject]