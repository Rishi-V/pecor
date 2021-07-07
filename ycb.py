import dataloader
import encoding_model
import calculate_score
from sklearn import mixture
import matplotlib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pdb
import numpy as np
import pickle
import evaluation
import shutil
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from pycocotools.coco import COCO
import pdb

from sklearn.cluster import KMeans

#main function for cost calculation
def run(output_f, error_f, search_object, data_folder, pose_file,model_dir,IND, gt_pose):

    #define alpha here
    alpha = 0.2
    #load data
    dl = dataloader.outputData
    data = dl.loadedData(data_folder)
    #load pose for each rendered image
    pose_dict = {}
    with open(pose_file, 'r') as f1:
        data_pose = f1.readlines()
    for line in data_pose:
        if line:
            pose = line.rstrip().split(",")
            pose_dict[int(pose[0])] = [float(pose[1]),float(pose[2]),float(pose[3]),float(pose[4]),float(pose[5]),float(pose[6]),float(pose[7])]
    #load model
    model = encoding_model.SubSetAlexNetModel(1)
    model = model.to(DEVICE)
    #score 
    score_function = calculate_score.calculateScore()
    gt = None
    best_score = 0
    best_pic = None
    rarity_score = None

    for inputs,labels in tqdm(data):

        idx = labels[0].numpy().flatten()[0]
        inputs = inputs.to(DEVICE)
        output = model(inputs)
        
        #construct mask for rendered region, mask is simple average pooling with same stride, kernal size and padding with the model used
        mask = torch.sum(inputs,1)
        #denoise
        mask[mask < 0.1] = 0
        mask[mask >= 0.1] = 1
        mask = mask.to(DEVICE)
        mask = model.forward_mask(mask)
        # 
        if gt is None:
            gt = output
            mask2 = mask
            indices = np.where(mask.cpu() != 0)[1:]
            save_data = gt.squeeze().permute(1,2,0).cpu()
            save_data_flat = save_data[indices]
            indices = np.where(mask.cpu().squeeze().flatten() != 0)[0]
            # k mean for rarity (only calculated once for each input image)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=20,random_state = 333)
            kmeans.fit(save_data_flat)
            predict = kmeans.predict(save_data_flat)
            frequency = np.bincount(predict)
            sum_cluster = np.sum(frequency)
            frequency = frequency/sum_cluster
            #here is the width and height after model
            w = 474
            h = 634
            num_output = w*h
            vis_score = np.zeros(num_output)
            for idx in range(len(indices)):
                vis_score[indices[idx]] = 1- frequency[predict[idx]] 
            rarity_score = torch.tensor(np.reshape(vis_score,(w,h))).float().to(DEVICE)      
        else:
            #mask is for rendered image, mask2 is for input image
            score = score_function.score(output,gt,mask,mask2,alpha,rarity_score)
            if score > best_score:
                best_score = score
                best_pic = idx
                
    res_pose = pose_dict[best_pic]
    evaluation.calculate_ADD(gt_pose, res_pose,model_dir,search_object,f = error_f, IND = IND,ycb = True)
    output_f.write(str(IND)+":"+str(best_pic)+":"+str(res_pose[0])+","+str(res_pose[1])+","+str(res_pose[2])+","+str(res_pose[3])+","+str(res_pose[4])+","+str(res_pose[5])+","+str(res_pose[6])+"\n")
    return res_pose




        

       