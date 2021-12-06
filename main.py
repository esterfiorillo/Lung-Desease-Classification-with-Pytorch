import numpy as np
import torch
import torchvision
import torchvision, torchvision.transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import networks
import dataset
import metrics
import train
import skimage
from skimage.io import imread
import argparse
import yaml

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/demo_covidx.yaml', help='Path to the config file.')
opts = parser.parse_args()
config = get_config(opts.config)


batch_size = config['batch_size']
num_epochs = config['num_epochs']
learning_rate = config['learning_rate']
channels = config['channels']
method = config['method']
feature_extract = config['feature_extract']


path_images_train = config['path_images_train']
path_images_val = config['path_images_val']
masks_train = config['masks_train']
masks_val = config['masks_val']

list_img_train, lbl_train, list_masks_train = dataset.get_dataset_info(path_images_train, masks_train)
list_img_val, lbl_val, list_masks_val = dataset.get_dataset_info(path_images_val, masks_val)


mean = [0.5401238348652285]
std = [0.2623742925726315]


transform_data = torchvision.transforms.Compose([transforms.ToTensor()
                                                ])



tmp_dataset_train = dataset.COVID19_Dataset(list_img_train, lbl_train, list_masks_train, mode = 'train', transform = transform_data)
tmp_dataset_val = dataset.COVID19_Dataset(list_img_val, lbl_val, list_masks_val, mode = 'val', transform = transform_data)


# Create a sampler by samples weights 
sampler = torch.utils.data.sampler.WeightedRandomSampler(
    weights=tmp_dataset_train.samples_weights,
    num_samples=tmp_dataset_train.len)


dataloaders_dict = {}


dataloaders_dict['train'] = torch.utils.data.DataLoader(tmp_dataset_train, 
                                                    batch_size=batch_size, 
                                                    sampler=sampler,
                                                    #shuffle = True,
                                                    num_workers=5)

dataloaders_dict['val'] = torch.utils.data.DataLoader(tmp_dataset_val, 
                                                    batch_size=batch_size, 
                                                    num_workers=5, 
                                                    shuffle=False)


num_classes = dataloaders_dict['train'].dataset.num_classes


model_ft, _ = networks.model_factory(method, num_classes, channels, feature_extract)
model_ft = model_ft.to(device)


params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)


optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=learning_rate, weight_decay=1e-5, amsgrad=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

criterion = nn.CrossEntropyLoss()
'''
weights = [2.422134, 27.084856, 1.817458]
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss().to(device)
'''

model_ft, res = train.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_classes, scheduler, num_epochs=num_epochs, is_inception=(method=="inception"))

#saving best models

out = 'outputs/model24.pt'
torch.save(model_ft, out)
