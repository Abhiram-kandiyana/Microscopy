import os
import cv2
import sys
import numpy as np
sys.path.append(r'E:\MedicalImageAnalysis\source_code\Utils')
from disp_img import disp_img

path_to_img = r'C:\Users\KAVYA\Abhiram\microscopy\mean-shift\marked_images_latest\test\seed_tres_10_data_tres_14\10_2'
img_list=os.listdir(path_to_img)

for i in img_list:
    img = cv2.imread(os.path.join(path_to_img,i), -1)
    # img = cv2.resize(cv2.imread(os.path.join(path_to_img,i)),(123,123),cv2.INTER_NEAREST)
    # img = img/255
    # img=np.reshape(img,(img.shape[0],img.shape[1],1))
    # ch2 = np.zeros(img.shape)
    # img  =np.concatenate((ch2,ch2,img),axis=2)
    # img = np.uint8(img)
    # img = cv2.GaussianBlur(img,(3,3),0)
    disp_img('Image - '+i, img)#(img[:,:,2]>200)*100)
    # cv2.waitKey(0)
 