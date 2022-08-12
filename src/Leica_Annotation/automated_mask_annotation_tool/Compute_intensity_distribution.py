import os
import numpy as np
import h5py
import cv2
import random
import statistics


#  sort the slice names in a stack
def slice_number(sliceName, stackName, aug_type):
    t = sliceName.replace(stackName + '_', '')
    return int(t.replace('_' + aug_type, ''))


def compute_intensity_distribution(path_images, path_masks):
    # check OpenCV version
    cv2_ver_major = cv2.__version__.split('.')[0]
    if path_masks is None:
        file_names = os.listdir(path_images)
    else:
        file_names = os.listdir(path_masks)
    stackNames = []
    aug_types = []
    for name in file_names:
        splits = name.split('_')
        try:
            stackNames.append(splits[0] + '_' + splits[1] + '_' + splits[2] + '_' + splits[3] + '_' + splits[4])
            # if (section_name == splits[1]):
            #    stackNames.append(splits[0]+'_'+splits[1]+'_'+splits[2])
            if name.count("_") == 5:
                aug_type = ''
            else:
                aug_type = name.split('_', 6)[6]
            aug_types.append(os.path.splitext(aug_type)[0])
        except:
            pass
    unique_stack_names = set(stackNames)
    stackNames = sorted(unique_stack_names)

    uniques_aug_types = set(aug_types)
    aug_types = sorted(uniques_aug_types)

    total_stacks = len(stackNames) * len(aug_types)

    print('total stacks:', total_stacks)
    print("Per particle intensities:")
    avg_intensity = []  # an entry per stack
    for stackName in stackNames:
        # stackName = 'PI3-22_Section2_Stack1'
        sliceNames_same_stack = [os.path.splitext(name)[0] for name in file_names if
                                 name.startswith(stackName + '_')]  # take images of only a stack
        # print('Slices per stack:', len(sliceNames_same_stack))

        for aug_type in aug_types:
            sliceNames = []
            for name in sliceNames_same_stack:
                if name.count("_") == 5:
                    aug = ''
                else:
                    aug = name.split('_', 6)[6]
                if aug == aug_type:
                    sliceNames.append(os.path.splitext(name)[0])
            sliceNames_sorted = sorted(sliceNames, key=lambda sliceName: slice_number(sliceName, stackName, aug_type))
            sliceNames = sliceNames_sorted
            del sliceNames_sorted
            # print(sliceNames)

            # print(sliceNames)

            # read images of a stack
            for sliceName in sliceNames:
                image = cv2.imread(os.path.join(path_images, sliceName + '.tif'), -1) #ext
                image = image//255
                print(image[0][:10])
                image = np.uint8(image)
                print(image[0][:10])
                exit()
                mask = cv2.imread(os.path.join(path_masks, sliceName + '.bmp'), cv2.IMREAD_GRAYSCALE)  # sliceName
                #cv2.imshow("image before resizing", image)
                #cv2.imshow("mask", mask)
                if image.shape != mask.shape:
                    image = cv2.resize(image, mask.shape, interpolation=cv2.INTER_NEAREST)
                #cv2.imshow("image after resizing", image)
                #cv2.waitKey()
                # find contours
                if cv2_ver_major == '3':
                    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                else:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # iterate through each contour and save avg intensity
                for contour in contours:
                    if CENTROID_INTENSITY:
                        M = cv2.moments(contour)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        avg_intensity.append(image[cy, cx])
                        print(image[cy, cx])
                    else:
                        cntr_mask = np.zeros_like(mask)
                        cv2.drawContours(cntr_mask, [contour], -1, 255, -1)
                        #_, temp = cv2.threshold(image, 25700, 255, cv2.THRESH_BINARY)  # 100
                        #temp_mask = cv2.bitwise_and(cntr_mask, np.uint8(temp))
                        # temp_mask = cv2.erode(temp_mask, np.ones((3,3), np.uint8), iterations=2)
                        temp_mask = cntr_mask

                        non_zeros = image[temp_mask > 0]
                        if len(non_zeros) > 0:
                            rep_intensity = int(cv2.mean(image, temp_mask)[0])  # mean intensity
                            #rep_intensity = int(np.median(non_zeros))  # median intensity
                        else:  # for debugging
                            cv2.imshow("image", image)
                            cv2.imshow("cntr mask", cntr_mask)
                            #cv2.imshow("th image", temp)
                            cv2.imshow("temp_mask", temp_mask)
                            cv2.waitKey()
                        if not np.isnan(rep_intensity) and rep_intensity:
                            avg_intensity.append(rep_intensity)
                        print(rep_intensity)

    print("Overall avg: ", statistics.mean(avg_intensity))
    print("Overall std: ", statistics.stdev(avg_intensity))
    # print(avg_intensity)


if __name__ == "__main__":
    #  INPUTS: START
    CENTROID_INTENSITY = False  # use intensity at centroid for each particle or avg intensity of the particle
    path_images = r'D:\mircroscopy-1\16bitimages\data_split_16bit\images2\train\images'  # os.path.join(path_to_data_folds, fold, phase,'images') # path to cropped images
    path_masks = r'D:\mircroscopy-1\16bitimages\data_split_16bit\images2\train\masks'  # os.path.join(path_to_data_folds, fold, phase, 'masks')  # path to cropped masks.
    #  INPUTS: END

    compute_intensity_distribution(path_images, path_masks)
    print('DONE')
