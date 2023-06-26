import os, torch, cv2, random
import numpy as np
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from scipy.ndimage.morphology import binary_erosion
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import filters
import numpy as np
import imageio
import dataloader.transforms as trans
import json, numbers
from glob import glob
import pickle

class BUDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        #print(self.size)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                #trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                #trans.adjust_light(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])


    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label


    def __len__(self):
        return self.size


class APTOSDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        #print(self.size)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                #trans.adjust_light(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        #self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label

    def __len__(self):
        return self.size



class ISICDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)

        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                #trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                #trans.adjust_light(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        #self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label


    def __len__(self):
        return self.size
