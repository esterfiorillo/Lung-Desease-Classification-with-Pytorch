import torch
import torchvision
#import torchxrayvision as xrv
import torchvision, torchvision.transforms
import skimage
from skimage.io import imread, imsave
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import numpy as np
import skimage.transform as scikit_transform
from sklearn.utils import shuffle
from skimage import color
from skimage import exposure
from skimage import transform
import random


def get_dataset_info (dataset_path, masks_path):
    list_images = []
    list_labels = []
    list_masks = []
    for i in os.listdir(os.path.join(dataset_path, "pneumonia")):   
        list_images.append(os.path.join(dataset_path, "pneumonia", i))
        list_labels.append(0)
        list_masks.append((os.path.join(masks_path, "pneumonia", i)))
    
    for i in os.listdir(os.path.join(dataset_path, "covid")):
        list_images.append(os.path.join(dataset_path, "covid", i))
        list_labels.append(1)
        list_masks.append((os.path.join(masks_path, "covid", i)))

    for i in os.listdir(os.path.join(dataset_path, "normal")):
        list_images.append(os.path.join(dataset_path, "normal", i))
        list_labels.append(2)
        list_masks.append((os.path.join(masks_path, "normal", i)))


    list_images, list_labels, list_masks = shuffle(list_images, list_labels, list_masks)

    return list_images, list_labels, list_masks

class COVID19_Dataset(Dataset):

    def __init__ (self, list_images, list_labels, list_masks, mode, transform=None):

        self.list_images = list_images
        self.list_labels = list_labels
        self.list_masks = list_masks
        self.len = len(self.list_images)
        self.mean =  0.5401238348652285
        self.std = 0.2623742925726315
        self.mode = mode

        # Create dict of classes
        self.classes = ['pneumonia', 'covid', 'normal']
        self.num_classes = len(self.classes)
       
        
        self.weight_class = 1. / np.unique(np.array(self.list_labels), return_counts=True)[1]
        #self.weight_class = np.array((2.422134, 27.084856, 1.817458))
        self.samples_weights = self.weight_class[self.list_labels]
        self.transform = transform

    
    def trim(self, img, prb):

        tolerance = 0.05 * float(img.max())

        # Mask of non-black pixels (assuming image has a single channel).
        bin = img > tolerance

        # Coordinates of non-black pixels.
        coords = np.argwhere(bin)

        # Bounding box of non-black pixels.
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

        # Get the contents of the bounding box. 
        img_crop = img[x0:x1, y0:y1]
        prb_crop = prb[x0:x1, y0:y1]

        return img_crop, prb_crop


    def __len__(self):
        return self.len

    def weight(self):
        return self.weight_class

    def __getitem__(self, index):
        img_path = self.list_images[index]
        mask_path = self.list_masks[index]
        lbl = self.list_labels[index]

        img = imread(img_path, 1)
        mask = imread(mask_path, 1)

        #img = img.astype(np.float32)
        #mask = mask.astype(np.float32)

        #mask = mask/255
        #img = img/255

        #img, mask = self.trim(img, mask)

        if self.mode == 'train':
            aug_choice = np.random.randint(3)
            if aug_choice == 0:
                #Flip an array horizontally.
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
                
            elif aug_choice == 1:
                #Flip an array horizontally.
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            
            #elif aug_choice ==2:
            else:
                angle = (np.random.rand(1) - 0.5) * 20
                img = transform.rotate(img, angle)
                mask = transform.rotate(mask, angle)
                
            

        img = (img-self.mean)/self.std
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)))

        if img.shape[0] != 224 or img.shape[1] != 224:
            img = scikit_transform.resize(img, (224,224)).astype(img.dtype)
        
        if mask.shape[0] != 224 or mask.shape[1] != 224:
            mask = scikit_transform.resize(mask, (224,224)).astype(mask.dtype)
              

        img = img[:, :, None]
        mask = mask[:, :, None]

        imagem = np.concatenate((img, img, mask), axis=2)

        # Apply transform 
        if self.transform:
            imagem = self.transform(imagem).float()

        
        return imagem, lbl, img_path