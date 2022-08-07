import cv2
from glob import glob
from sklearn.datasets import load_files
import numpy as np
import os
import tifffile
import math

def crop_images(path):
    images = os.listdir(path)
    W=123
    H=123
    size=10
    start_index = math.ceil((len(images) - size)/2)
    middle_images = images[start_index:start_index+10]
    for img_path in middle_images:
        img = tifffile.imread(path+img_path)
        copy = img.copy()
        imageHeight,imageWidth = img.shape[0],img.shape[1]
        x2=1
        y2=1
        for y in range(0,imageHeight,H):
            for x in range(0,imageWidth,W):
                # if((imageHeight - y) < H or  (imageHeight - x) < W):
                #     break
                cropped = copy[x:x+W,y:y+H]
                if(not os.path.exists(path+str(x2)+'_'+str(y2))):
                    os.mkdir(path+str(x2)+'_'+str(y2))
                cv2.imwrite(path+str(x2)+'_'+str(y2)+'/'+img_path,cropped)
                cv2.rectangle(img,(x,y),(x+W,y+H),(0,0,255),2)
                x2=x2+1
            y2=y2+1
            x2=1
        # if(not os.path.exists(path+'patched/')):
        #     os.mkdir(path+'patched/')
        # cv2.imwrite(path+'patched/'+img_path,img)


crop_images('./16bitimages/Slide2/NeoCx/')

# print(glob('images/Neo cx/*'))

# img = cv2.imread('Neo cx/Series001_z00_ch01.png')
# cv2.imshow('uncropped',img)