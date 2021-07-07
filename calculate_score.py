import torch
from torch import nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import spatial
import pdb
import pandas as pd
class calculateScore:
    def __init__(self):
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def score(self, output,gt,mask,mask2,percent_param,rarity_score): 
        # raw score, cosine similarity in rendered region
        self.pixel_score = torch.abs(self.cos(output,gt))
        self.pixel_score_raw = self.cos(output,gt)
        mask = torch.where(mask<0.5, 0, 1)
        #mask only zero out background pixels, more intuitively, it should punish parts that rendered on background 
        self.mask = torch.where(mask2<0.5, 0, 1)*mask
        self.rarity_score = rarity_score*self.mask
        self.pixel_score = self.pixel_score*rarity_score*self.mask
        
        # indices in rendered region
        indices = np.where(mask.cpu() != 0)
        indices_num = int(len(indices[0])* percent_param)
        flattened_score = torch.flatten(self.pixel_score)
        # top K highest scores and their indices
        topk = torch.topk(flattened_score,indices_num)
        self.topk_num = topk[0]
        topk_ind = topk[1]
        score = (torch.sum(self.topk_num).cpu())/indices_num

        # reconstruct the cost image for visualization purpose
        w = self.pixel_score.shape[1]
        h = self.pixel_score.shape[2]
        num_output = w*h
        self.vis_score = torch.zeros(num_output)
        self.vis_score[topk_ind] = flattened_score[topk_ind].cpu()
        self.vis_score = torch.reshape(self.vis_score,(w,h))

        return score.cpu().numpy()

#visualize scores and mask as needed
def visualize_score(self,idx,percent_param,scene,file_name):
        self.vis_score[0][0] = 1
        plt.imshow(self.vis_score, cmap='hot')
        # self.rarity_score[0][0] = 1
        # plt.imshow(self.rarity_score.cpu().numpy().squeeze(), cmap='hot')
        # self.pixel_score1[0][0] = 1
        # plt.imshow(self.pixel_score1.cpu().numpy().squeeze())
        # plt.colorbar()
        # self.pixel_score[0][0] = 1
        # plt.imshow(self.pixel_score.cpu().numpy().squeeze(), cmap='hot')
        plt.show()
