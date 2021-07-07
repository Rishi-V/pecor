import torch
from torch import nn as nn
from torchvision import models

import matplotlib.pyplot as plt
import os
import numpy as np
import pdb

from util import pytorch_utils as ptu

# High level comments:
# Inspiration from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html


# Input: model (a model), requires_grad (bool)
# Output: None
# Side affects: Sets model paramters .requires_grad attribute to requires_grad
def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


class SubSetAlexNetModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        model = models.alexnet(pretrained=True)  # You may need to download the model once initially (this is handled)
        set_parameter_requires_grad(model, False)
        self.sequential = model.features[:num_layers]
        self.sequential =  nn.Sequential(model.features[0],model.features[3])
        self.mask_sequential = [nn.AvgPool2d(1, stride=2),nn.AvgPool2d(1, stride=2)]
        
        
        # Make sure that the last conv layer has stride=1
        for i in range(1, len(self.sequential)+1):
            if self.sequential[-i].__class__.__name__ == "Conv2d":
                self.sequential[-i].stride = 1
                self.mask_sequential[-i].stride = 1
                self.mask_sequential[-i].kernel_size =  self.sequential[-i].kernel_size
                self.mask_sequential[-i].padding =  self.sequential[-i].padding
        self.mask_sequential = nn.Sequential(*self.mask_sequential)
        
        print(f"Model subset: \n{self.sequential}")
        print(f"Mask Model subset: \n{self.mask_sequential}")
        
    # Input: images (B,C,D,D)
    # Output: (B,C,?,?)
    def forward(self, images):
        return self.sequential(images)

    def forward_mask(self, mask):
        return self.mask_sequential(mask)
    

def create_dataset(data_folder="data/mini_example/"):
    directory = os.fsencode(data_folder)
    images = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            anImage = plt.imread(data_folder+filename)
            images.append(np.array(anImage))
    return images

    


if __name__ == "__main__":
    # images = create_dataset()
    model = SubSetAlexNetModel(4)

    # for anImage in images:
    #     anImage = ptu.np_img_to_tensor(anImage) # (W,H,C) np -> (1,C,W,H)
    #     tmp = model(anImage)

    # img_size = (1,3,64,64)
    # rand_img = np.random.rand(*img_size)
    # pdb.set_trace()

    # tmp = model(rand_img)
    # initialize_model()
