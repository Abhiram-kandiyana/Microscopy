# script to get min and max blob size in ST so that it can be used as a threshold in postprocessing
import cv2
import numpy as np
import os
from statistics import mean
from statistics import median
import matplotlib.pyplot as plt
import importlib

loader = importlib.machinery.SourceFileLoader( 'mc_constants', r'/Users/abhiramkandiyana/Microscopy/constants/constants.py')
spec = importlib.util.spec_from_loader( 'mc_constants', loader )
mc_constants = importlib.util.module_from_spec( spec )
loader.exec_module( mc_constants)

stackNameNotStartsWith = "."

##INPUTS: START
path_to_ST = mc_constants.slide1_64X64_masks_train  #r'C:\Users\palakdave\MedicalImagingProject\Data\Leica_images\64x64\experiments\66\Slide1\train\masks'
##INPUTS: END
stack_names = os.listdir(path_to_ST)
# image_names = os.listdir(path_to_ST)
# image_names = [i for i in image_names if os.path.isfile(os.path.join(path_to_ST,i))]
min_area=1024*1024
max_area=0

all_areas = []
all_widths = []
all_heights = []
for stack_name in stack_names:
    if(stack_name.startswith(stackNameNotStartsWith)):
        continue

    image_names = os.listdir(os.path.join(path_to_ST,stack_name))
    image_names = [i for i in image_names if os.path.isfile(os.path.join(path_to_ST,stack_name,i))]
    for image_name in image_names:
        image = np.uint8(cv2.imread(os.path.join(path_to_ST,stack_name,image_name),-1))
        retval,labels,stats,_ = cv2.connectedComponentsWithStats(image)

        for idx in range(1,retval-1):
            if stats[idx,cv2.CC_STAT_AREA]==1:
                # cv2.imshow("img",image)
                print(image_name)
                # cv2.waitKey()

        areas =[stats[i,cv2.CC_STAT_AREA] for i in range(1, retval)]  # discarding zero bcz that is background
        bb_widths = [stats[i,cv2.CC_STAT_WIDTH] for i in range(1, retval)]
        bb_heights = [stats[i,cv2.CC_STAT_HEIGHT] for i in range(1, retval)]
        all_areas.extend(areas)  # to compute average area in all images
        all_widths.extend(bb_widths) # to compute average bb size width in all images
        all_heights.extend(bb_heights)  # to compute average bb size height in all images
        try:
            if min(areas) < min_area:
                min_area = min(areas)
            if max(areas) > max_area:
                max_area = max(areas)
        except:
            pass
        #print('min area: {}, max area: {}'.format(min(areas), max(areas)))
        #cv2.imshow('image', np.uint8(image>0)*255)
        #cv2.waitKey()
print('min area: {}, max area: {}, mean area: {}, median area: {}, mean bb width: {}, mean bb height: {}, min bb width: {}, min bb height: {}, max bb width: {}, max bb height: {}'.
      format(min_area, max_area, mean(all_areas), median(all_areas), mean(all_widths), mean(all_heights), min(all_widths), min(all_heights), max(all_widths), max(all_heights)))


uniq, cnts = np.unique(all_areas, return_counts=True)
#plt.hist(all_areas, density=False, bins=45)  # density=False would make counts
plt.bar(uniq, cnts, color='g')
plt.ylabel('No. of Particles', fontsize=18)
plt.xlabel('Area (pixels)', fontsize=18)
plt.xticks(range(0, max(uniq), 5)) #
plt.xticks(fontsize= 12)
plt.show()


uniq, cnts = np.unique(all_widths, return_counts=True)
#plt.hist(all_areas, density=False, bins=45)  # density=False would make counts
plt.bar(uniq, cnts, color='g')
plt.ylabel('No. of Particles', fontsize=18)
plt.xlabel('Width (pixels)', fontsize=18)
plt.xticks(range(0, max(uniq), 5)) #
plt.xticks(fontsize= 12)
plt.show()


uniq, cnts = np.unique(all_heights, return_counts=True)
#plt.hist(all_areas, density=False, bins=45)  # density=False would make counts
plt.bar(uniq, cnts, color='g')
plt.ylabel('No. of Particles', fontsize=18)
plt.xlabel('Height (pixels)', fontsize=18)
plt.xticks(range(0, max(uniq), 5)) #
plt.xticks(fontsize= 12)
plt.show()

