import cv2
from glob import glob
from sklearn.datasets import load_files
import numpy as np
import os

def crop_images(path):
    data = load_files(path)
    images = np.array(data['filenames'])
    W=490
    H=490
    for img_path in images:
        img = cv2.imread(img_path)
        copy = img.copy()
        os.mkdir(img_path[:-4])
        imageHeight,imageWidth = img.shape[0],img.shape[1]
        x2=1
        y2=1
        for y in range(0,imageHeight,H):
            for x in range(0,imageWidth,W):
                if((imageHeight - y) < H or  (imageHeight - x) < W):
                    break
                cropped = copy[x:x+W,y:y+H]
                cv2.imwrite(img_path[:-4]+'/'+str(x2)+'_'+str(y2)+'.png',cropped)
                cv2.rectangle(img,(x,y),(x+W,y+H),(0,255,0),2)
                x2=x2+1
            y2=y2+1
            x2=1
        cv2.imwrite(img_path[:-4]+'/original.png',img)


crop_images('neoCx_lng_tiff')

# print(glob('images/Neo cx/*'))

# img = cv2.imread('Neo cx/Series001_z00_ch01.png')
# cv2.imshow('uncropped',img)