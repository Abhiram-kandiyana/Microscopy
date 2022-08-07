import cv2
from glob import glob
from sklearn.datasets import load_files
import numpy as np
import os
import tifffile
import math
save_to_folder_path = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\Slide2-64X64"
def crop_images(path):
    images = os.listdir(path)
    W=64
    H=64
    size=10
    start_index = math.ceil((len(images) - size)/2)
    middle_images = images[start_index:start_index+10]
    for img_path in middle_images:
        img = tifffile.imread(os.path.join(path,img_path))
        # img = img/255
        # img=np.reshape(img,(img.shape[0],img.shape[1],1))
        # ch2 = np.zeros(img.shape)
        # img  =np.concatenate((ch2,ch2,img),axis=2)
        # img = np.uint8(img)
        copy = img.copy()
        imageHeight,imageWidth = img.shape[0],img.shape[1]
        x2=1
        y2=1
        for y in range(0,imageHeight,H):
            for x in range(0,imageWidth,W):
                # if((imageHeight - y) < H or  (imageHeight - x) < W):
                #     break
                cropped = copy[x:x+W,y:y+H]
                image_dir_path=os.path.join(save_to_folder_path,r'Neo_cx',str(x2)+'_'+str(y2))
                if(not os.path.exists(image_dir_path)):
                    os.mkdir(image_dir_path)
                cv2.imwrite(os.path.join(image_dir_path,img_path),cropped)
                cv2.rectangle(img,(x,y),(x+W,y+H),(0,0,255),2)
                x2=x2+1
            y2=y2+1
            x2=1
        if(not os.path.exists(os.path.join(path,'patched/'))):
            os.mkdir(os.path.join(path,'patched/'))
        cv2.imwrite(os.path.join(path,'patched/',img_path),img)


crop_images(r'C:\Users\KAVYA\Abhiram\microscopy\16bitimages\Slide2\NeoCx')

# print(glob('images/Neo cx/*'))

# img = cv2.imread('Neo cx/Series001_z00_ch01.png')
# cv2.imshow('uncropped',img)