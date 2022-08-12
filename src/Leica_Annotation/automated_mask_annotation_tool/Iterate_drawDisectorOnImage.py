# Draw disector on all the images in the given folder

import cv2
import os
import sys
sys.path.append(r'E:\MedicalImageAnalysis\source_code\Utils')
from PutDisector_OnImage import *

#INPUTS: Start
path_to_src_dir = r'E:\MedicalImageAnalysis\Data\PI3-21_images\EDF'  # Path to the folder nof images on which to draw disector
save_to_dir = r'E:\MedicalImageAnalysis\Data\PI3-21_images\EDF_with_disector'  # path to the folder where to save the images with disector
ext = '.png'  # extension of the input files
#INPUTS: End

if not os.path.exists(save_to_dir):
    os.makedirs(save_to_dir)

images = os.listdir(path_to_src_dir)
print('Processing {} images'.format(len(images)))
for imgName in images:
    if os.path.splitext(imgName)[1] != ext:
        continue
    img = cv2.imread(os.path.join(path_to_src_dir, imgName))
    disector_img, _, _, _ = PutDisectorOnImage(img)
    #cv2.imshow('disector_img', disector_img)
    #cv2.waitKey()
    cv2.imwrite(os.path.join(save_to_dir, imgName), disector_img)

print('Done')