import numpy as np
import pandas as pd
import cv2
import os
from copy import deepcopy
import shutil
import json



mask_folder = r'D:\mircroscopy-1\16bitimages\slide1-manual-masks\accepted_annotated'
img_folder = r'D:\mircroscopy-1\16bitimages\slide1-manual-masks\accepted'
slide=r'Neo_cx'
mask_folder,img_folder = os.path.join(mask_folder,slide),os.path.join(img_folder,slide)

masks_dir = os.listdir(mask_folder)

dice_coeff_arr = []
IOU_arr = []
for stack in masks_dir:
    mask_centroids_arr = []
    width=0
    height=0

    slices = os.listdir(os.path.join(mask_folder,stack))
    dice_coeff_stack=[]  
    IOU_stack = []      
    for sliceNo,sliceName in enumerate(slices):

        img_path = os.path.join(img_folder,stack,sliceName)  
        image = cv2.imread(img_path, -1)#
        (width,height,dims) = image.shape
        # images.append(image)
        # images1.append(input_img)
        #finding the centroids of the mask
        mask_img = cv2.imread(os.path.join(mask_folder,stack,sliceName))[:,:,2]
        (numLabels1, labels1, stats1, mask_centroids) = cv2.connectedComponentsWithStats(mask_img,8,cv2.CV_32S)   
        # print(np.unique(numLabels1))
        # cv2.imshow("labels1",labels1)
        # mask_labels.append(labels1)
        mask_centroids = np.array(mask_centroids[1:]) 


          
        for i,j in enumerate(np.int64(mask_centroids)):

            mask_contours,_=cv2.findContours(mask_img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for k,l in enumerate(mask_contours):
                if(cv2.pointPolygonTest(l,(int(j[0]),int(j[1])),True) >= 0):
                    emp_img = np.zeros((width,height))
                    cv2.drawContours(emp_img,mask_contours,k,(255,255,255),-1)
            tres_input = deepcopy(image)

            ret,tres_image = cv2.threshold(np.uint8(tres_input[:,:,2]),30,255,cv2.THRESH_BINARY)
            tres_contours,_=cv2.findContours(tres_image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for k,l in enumerate(tres_contours):
                if(cv2.pointPolygonTest(l,(int(j[0]),int(j[1])),True) >= 0):
                    emp_img1 = np.zeros((width,height))
                    cv2.drawContours(emp_img1,tres_contours,k,(255,255,255),-1)
            
            dice_coeff = 2*np.logical_and(emp_img,emp_img1).sum()/(np.count_nonzero(emp_img.astype(int)) + np.count_nonzero(emp_img1.astype(int)))
            IOU = np.logical_and(emp_img,emp_img1).sum()/(np.logical_or(emp_img,emp_img1).sum())
            # print("dice-coeff",dice_coeff)   
            dice_coeff_stack.append(round(dice_coeff,2))
            IOU_stack.append(round(IOU,2))

    dice_coeff_arr.append(dice_coeff_stack)
    IOU_arr.append(IOU_stack)
dice_coeff_df = pd.DataFrame(dice_coeff_arr)
dice_coeff_df.to_csv('./dice_coeff_tresholding.csv')

dice_coeff_df = pd.DataFrame(IOU_arr)
dice_coeff_df.to_csv('./IOU_tresholding.csv')





        # for j in mask_centroids[1:]:
        #     mask_centroids_arr.append(np.insert(j,0,int(sliceNo)))
