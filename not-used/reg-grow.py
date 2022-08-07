import numpy as np
import pandas as pd
import cv2
import os
from copy import deepcopy
import shutil
import json
from region_growing_code import region_growing



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
        mask_img = cv2.resize(cv2.imread(os.path.join(mask_folder,stack,sliceName))[:,:,2],(width,height),cv2.INTER_CUBIC)
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
                    # print('------------------------------------')
                    # print(emp_img.shape)
                    # cv2.imshow("emp img",emp_img)
                    # cv2.waitKey()
            # tres_input = deepcopy(image)
            out_img=region_growing.apply_region_growing(np.uint8(image[:,:,2]),(j[0],j[1]))
            # print(out_img.shape)
            # print('------------------------------------')
            reg_grow_contours,hv2 = cv2.findContours(np.uint8(out_img),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print("first")
            # reg_grow_input_img = deepcopy(input_img)
            emp_img1 = np.zeros((width,height)) 
            cv2.drawContours(emp_img1,contours=reg_grow_contours,contourIdx=-1,color=(255,255,255),thickness=-1)
            # cv2.imshow("reg gow output",out_img)
            # cv2.waitKey()
            
            # cv2.imshow("out img",out_img)
            # cv2.waitKey()
            # ret,tres_image = cv2.threshold(tres_input,30,255,cv2.THRESH_BINARY)
            # tres_contours,_=cv2.findContours(np.uint8(image[:,:,2]),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # reg_grow_contours,hv2 = cv2.findContours(np.uint8(out_img),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # for k,l in enumerate(tres_contours):
            #     if(cv2.pointPolygonTest(l,(int(j[0]),int(j[1])),True) >= 0):
            #         emp_img1 = np.zeros((width,height))
            #         cv2.drawContours(emp_img1,tres_contours,k,(255,255,255),-1)
            
          
            print('------------------------------------')
            print(np.logical_and(emp_img,emp_img1).sum())
            print(np.count_nonzero(emp_img.astype(int)) + np.count_nonzero(emp_img1.astype(int)))
           
            print('------------------------------------')
            dice_coeff = 2*np.logical_and(emp_img,emp_img1).sum()/(np.count_nonzero(emp_img.astype(int)) + np.count_nonzero(emp_img1.astype(int)))
            IOU = np.logical_and(emp_img,emp_img1).sum()/(np.logical_or(emp_img,emp_img1).sum())
            # print("dice-coeff",dice_coeff)   
            dice_coeff_stack.append(round(dice_coeff,2))
            IOU_stack.append(round(IOU,2))

    dice_coeff_arr.append(dice_coeff_stack)
    IOU_arr.append(IOU_stack)
dice_coeff_df = pd.DataFrame(dice_coeff_arr)
dice_coeff_df.to_csv('./dice_coeff_reg_grow.csv')

dice_coeff_df = pd.DataFrame(IOU_arr)
dice_coeff_df.to_csv('./IOU_reg_grow.csv')





        # for j in mask_centroids[1:]:
        #     mask_centroids_arr.append(np.insert(j,0,int(sliceNo)))
