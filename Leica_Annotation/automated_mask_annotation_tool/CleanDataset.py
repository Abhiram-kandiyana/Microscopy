# To go over annotated stacks and remove those have noise or stacks empty of images.



import os
import sys
import shutil
import cv2
from tqdm import tqdm


path2Mouse = '/media/saeed3/Seagate_Backup_Plus_Drive/BrainImagesGrant/IBA1_dataset_dualStain/Backup/IBA1_dualStain/PI3-35_annotated'

if not path2Mouse.endswith('_annotated'):
    sys.exit()

listOfSections = os.listdir(path2Mouse)

for Section in tqdm(listOfSections):
    if not os.path.isdir(os.path.join(path2Mouse,Section)):
        continue
    else:
        listOfStacks = os.listdir(os.path.join(path2Mouse,Section))
        for Stack in listOfStacks:
            if not os.path.isdir(os.path.join(path2Mouse,Section,Stack)):
                continue
            else:
                listOfImages = os.listdir(os.path.join(path2Mouse,Section,Stack))
                if len(listOfImages) == 0:
                    shutil.rmtree(os.path.join(path2Mouse,Section,Stack))
                    continue
                for Image in listOfImages:
                    cv2.namedWindow('Image')
                    cv2.moveWindow('Image',0,0)
                    if os.path.isdir(os.path.join(path2Mouse,Section,Stack,Image)):
                        continue
                    else:
                        img = cv2.imread(os.path.join(path2Mouse,Section,Stack,Image),-1)
                        cv2.resizeWindow('Image',img.shape[0],img.shape[1])
                        cv2.imshow('Image',img)
                        cv2.waitKey(100)
                userInput = input('Remove?')
                if userInput == 'y':
                    print('removing {} {}'.format(Section,Stack))
                    shutil.rmtree(os.path.join(path2Mouse,Section,Stack))
                else:
                    continue
