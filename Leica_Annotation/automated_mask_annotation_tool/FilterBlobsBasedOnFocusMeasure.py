# Take all predicted slices of a stack and make a pixel white where best confidence in z dir
# output: Saved post processed masks in input predictedMasksDir/PostProcessed


# TO DO: max indices are zeros for background and some cell can also have max indice zero. take care of this

import cv2
import numpy as np
import os, sys
import json

sys.path.append(r'E:\MedicalImageAnalysis\source_code\Utils')
from disp_img import *
from matplotlib import pyplot as plt
# from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def save_masks(saveTo_folder, imgName_list, imgs_list):
    for img, imgName in zip(imgs_list, imgName_list):
        cv2.imwrite(os.path.join(saveTo_folder, imgName), img)


def compute_sobel(img_gray):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(img_gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(img_gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad




# INPUTS: START
ENABLE_IMG_BLUR = False  # wheather to blur each slice before further processing
IMG_BLUR_KERNEL_SIZE = (7, 7)  # (3, 3)  # (7, 7)  # Kernel size of the Gaussian blur of the image
ENABLE_Z_BLUR = False  # wheather to blur Z dir of stack
Z_BLUR_KERNEL_SIZE = 0.5  # sigma of 1D kernel
PRED_CONFIDENCE_THRESHOLD = 15  # 10 #15  # minimum confidence threshold in predicted slices
ENABLE_AREA_FILTERING = True  # Enable area based filtering (discard cells with area<CELL_PIXEL_AREA_TH)
CELL_PIXEL_AREA_TH = 200  # 300  # min area of a cell in pixels
BASE_TH_LOWER_BY = 5  # while just thresholding the predicted mask, threshold is less than PRED_CONFIDENCE_THRESHOLD
# by this amount
ENABLE_CLOSING = False  # perform closing operation
CLOSING_KERNEL_SIZE = (3, 3)  # size of closing operation kernel
ENABLE_COMBINE_OVERLAPPING = False  # combine two overlapping blobs from two consecutive slices as one cell
MIN_OVERLAP_PIXELS = 40  # min overlapping pixels between two blobs from two consecutive slices to be combined as
ENABLE_COMBINE_HIGH_OVERLAP = True  # combine two blobs with overlap > MAX_OVERLAP_PERCENT in any two slices (not
# only consecutive)
MAX_OVERLAP_PERCENT = 75  # 80  # max allowed overlap between any two blobs in any slices
# single cell
ENABLE_DRAW_FILLED_BLOBS = True  # Draw blobs in the mask after filling holes (by drawcontours)
ENABLE_WATERSHED = False  # apply watershed on each slice, each blob
'''
stackNames = ['PI3-20_Section1_Stack3', 'PI3-20_Section1_Stack4', 'PI3-20_Section1_Stack5', 'PI3-20_Section1_Stack6', 'PI3-20_Section1_Stack7', 'PI3-20_Section1_Stack8', 'PI3-20_Section1_Stack9', 'PI3-20_Section2_Stack1', 'PI3-20_Section2_Stack2', 'PI3-20_Section2_Stack3', 'PI3-20_Section2_Stack4', 'PI3-20_Section2_Stack5', 'PI3-20_Section2_Stack6']
stackName = 'PI3-20_Section2_Stack2'  # name of the stack to process
'''
predictedMasksDir = r'E:\MedicalImageAnalysis\Data\temp\fold_PI20_PI20_PI20_test_predMasks'  # r'E
# :\MedicalImageAnalysis\Data\PI3-20_images\masks'  # Dir with all the predicted masks
postProcessedBinaryMaskDir =r'E:\MedicalImageAnalysis\Data\temp\fold_PI20_PI20_PI20_test_predMasks\PostProcessed_no_watershed'
saveTo_folder_pp = os.path.join(postProcessedBinaryMaskDir,'FocusFiltered')  # save post-processed predicted masks in this dir
# INPUTS: END

if not os.path.exists(saveTo_folder_pp):
    os.makedirs(saveTo_folder_pp)


predictedMasks = os.listdir(predictedMasksDir)
stackNames = []
for name in predictedMasks:
    if os.path.isdir(os.path.join(predictedMasksDir, name)):
        continue
    splits = name.split('_')
    try:
        stackNames.append(splits[0] + '_' + splits[1] + '_' + splits[2])
    except:
        pass
unique_stack_names = set(stackNames)
stackNames = sorted(unique_stack_names)

counts_dict = {}
for stackName in stackNames:
    stackName = 'PI3-20_Section3_Stack7'
    sliceNames = [name for name in predictedMasks if name.startswith(stackName)]  # take images of only a stack
    print(sliceNames)
    slices = []
    MASKS = []

    # read images of a stack
    for sliceName in sliceNames:
        temp = cv2.imread(os.path.join(predictedMasksDir, sliceName), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(postProcessedBinaryMaskDir, sliceName), cv2.IMREAD_GRAYSCALE)
        slices.append(temp)
        MASKS.append(mask)

    slicesArray = np.array(slices)  # convert list of 2D numpy arrays to ndarray
    MASKS = np.array(MASKS)

    # Filter cells in MASKS based on area like property
    if ENABLE_AREA_FILTERING:
        for idx, mask in enumerate(MASKS):
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask)  # 0th label is for background here
            # delete contour if area less then area threshold
            #grad_img = compute_sobel(slicesArray[idx])
            # grad_img = cv2.Laplacian(slicesArray[idx], cv2.CV_64F)
            min_img = slicesArray.min(axis=0)
            #grad_imgs = np.gradient(slicesArray, edge_order=2, axis=0)
            #grad_img = grad_imgs.var(axis = 0)
            #grad_img = cv2.convertScaleAbs(grad_img)
            #cv2.imshow('grad_img', grad_img)
            #cv2.waitKey()
            for blobNo in range(1, retval):  # 0th index is for background here
                #x1 = stats[blobNo][cv2.CC_STAT_LEFT]
                #x2 = x1 + stats[blobNo][cv2.CC_STAT_WIDTH]
                #y1 = stats[blobNo][cv2.CC_STAT_TOP]
                #y2 = y1 + stats[blobNo][cv2.CC_STAT_HEIGHT]
                #blob_roi_img = slicesArray[idx][y1:y2, x1:x2]
                # focus_measure = cv2.Laplacian(blob_roi_img, cv2.CV_64F).var()
                # focus_measure = cv2.Laplacian(blob_roi_img, cv2.CV_64F)
                #grad_img = compute_sobel(blob_roi_img)
                #grad_img = grad_imgs[idx]
                grad_img = slicesArray[idx] - min_img
                focus_measure = sum(grad_img[labels == blobNo]) / stats[blobNo][cv2.CC_STAT_AREA]
                #focus_measure = grad_img[labels == blobNo].var()
                blob_img = np.zeros_like(grad_img)
                blob_img[labels == blobNo] = 255
                cv2.destroyAllWindows()
                cv2.imshow('blob_img', blob_img)
                cv2.imshow('grad_img', np.int8(grad_img))
                print('focus_measure', focus_measure)
                cv2.waitKey()

    if ENABLE_DRAW_FILLED_BLOBS:
        for idx, mask in enumerate(MASKS):
            _, temp_contours, _ = cv2.findContours(MASKS[idx], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(MASKS[idx], temp_contours, -1, 255, -1)

    if ENABLE_WATERSHED:
        MASKS = watershed_stack(MASKS, slicesArray)

    # save masks and automatic counts
    save_masks(saveTo_folder_pp, sliceNames, MASKS)  # save post-processed masks
    cell_count = 0
    for mask in MASKS:
        retval, _ = cv2.connectedComponents(mask)
        cell_count = cell_count + (retval - 1)
    counts_dict[stackName] = cell_count
with open(os.path.join(saveTo_folder_pp, "automatic_counts.json"), 'w') as fp:
    json.dump(counts_dict, fp)


