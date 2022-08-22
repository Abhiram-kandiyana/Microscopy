from ast import excepthandler
from cgi import test
from copy import deepcopy
import enum
import shutil
from warnings import catch_warnings
import numpy as np
import cv2
import os
import json
import pandas as pd
import math
import importlib
import random

global count

# Import constants module
loader = importlib.machinery.SourceFileLoader('mc_constants',r'/Users/abhiramkandiyana/Microscopy/constants/constants.py')
spec = importlib.util.spec_from_loader('mc_constants', loader)
mc_constants = importlib.util.module_from_spec(spec)
loader.exec_module(mc_constants)
# img = cv2.imread('./data/merged.png')

slideName = mc_constants.Slide2

stack_length = mc_constants.stack_length

image_dir_name = mc_constants.image_dir

image_path = os.path.join(mc_constants.testing_dir,slideName,image_dir_name)

random.seed(44)

image_dir = os.listdir(image_path)
maskno = 0

disector_percentage = 75

result_json_list = []

tres = 10
tres1 = 14
cluster_radius = 3
invalid_stack_starts_with  = '.'

annotation_path = os.path.join(mc_constants.testing_dir,slideName,mc_constants.count_annotation_const,mc_constants.count_annotation_json)



def PutDisectorOnImage(I, percentage=25):
    # if I not 3 channel then make it 3 channel
    if len(I.shape) != 3:
        I_new = np.empty((I.shape[0], I.shape[1], 3), dtype=I.dtype)
        I_new[:, :, 0] = I
        I_new[:, :, 1] = I
        I_new[:, :, 2] = I
        I = I_new
    percentage = 100 * math.sqrt(percentage / 100)
    disectorWidth = int(math.ceil(percentage * min(I.shape[0], I.shape[1]) / 100))
    x = int(math.ceil((I.shape[0] - disectorWidth) / 2))
    y = int(math.ceil((I.shape[1] - disectorWidth) / 2))

    # Inclusion
    I[x, range(y, y + disectorWidth)] = [0, 255, 0]
    I[range(x, x + disectorWidth), y + disectorWidth] = [0, 255, 0]

    # Exclusion
    I[range(x, x + disectorWidth), y] = [0, 255, 0]
    I[x + disectorWidth, range(y, y + disectorWidth)] = [0, 255, 0]
    return I, disectorWidth, x, y

total_count = 0

ManualAnnotation = []
for stackNo, stackName in enumerate(image_dir):
    print(stackName)
    if(not stackName.startswith(invalid_stack_starts_with)):
        stackNameFull = slideName + '_' + image_dir_name + '_' + stackName
        stack_path = os.path.join(image_path, stackName)
        sliceNames = os.listdir(stack_path)
        sliceNames = sorted(sliceNames)
        images = []
        images1 = []
        centroids_arr = []
        width = 0
        height = 0
        for sliceNo, sliceName in enumerate(sliceNames):
            image = cv2.imread(os.path.join(image_path, stackName, sliceName), -1)  #
            image = image / 256
            # cv2.imshow("image",image)
            image = np.uint8(image)
            # cv2.imshow("input image",image)
            # cv2.waitKey()
            (width, height) = image.shape
            ret, image1 = cv2.threshold(image, tres, 255, cv2.THRESH_TOZERO)
            # cv2.imshow("seed thresh image",image1)
            # cv2.waitKey()
            output = cv2.connectedComponentsWithStats(image1, 8, cv2.CV_32S)
            (numLabels, ccws_labels, stats, centroids) = output
            # cv2.imshow("image1",image)
            # cv2.waitKey()
            # image = cv2.GaussianBlur(image,k,0)
            images.append(image)
            images1.append(image1)
            # finding the centroids of the mask
            # cv2.imshow('mask_img',cv2.imread(os.path.join(mask_path,stackName,sliceName[:-4]+'.png'))[:,:,2])
            # cv2.waitKey()

            for i in centroids[1:]:
                centroids_arr.append(np.append(i, int(sliceNo)))
        images_array = np.array(images)
        images_array1 = np.array(images1)
        centroids_arr = np.array(centroids_arr)
        data = []
        data1 = []
        for z in range(stack_length):
            for x in range(width):
                for y in range(height):
                    # data.append([z,x,y,images_array[z,x,y]])
                    if (images_array[z, x, y] > tres1):
                        data.append([z, x, y, images_array[z, x, y]])
                    # if(images_array[z,x,y] > tres):
                    #     data1.append([z,x,y,images_array[z,x,y]])
                    # if(centroids_arr[z][x][0] == data[z   b n])
        for i in centroids_arr:
            data1.append([i[2], i[1], i[0], images_array1[int(i[2]), int(i[1]), int(i[0])]])
        data, data1 = np.array(data), np.array(data1)
        data, data1 = np.float32(data), np.float32(data1)


        def weightedEuclideanDist(a, b, w=[0.2, 0.4, 0.4, 0]):
            q = a - b
            return np.sqrt((w * q * q).sum())


        count = 0


        class Mean_Shift:
            def __init__(self, radius=cluster_radius, max_iter=800):
                self.radius = radius
                self.max_iter = max_iter
                self.iter = 0

            def fit(self, data):
                centroids = {}
                global copy
                copy = []
                for i in range(len(data1)):
                    centroids[i] = data1[i]
                for i in data:
                    copy.append(np.append(i, 100))
                prev_in_bandwidth = {}
                while True:
                    self.iter = self.iter + 1
                    new_centroids = []
                    # if(self.iter == 7 and len(copy) > 0):
                    #     break
                    # if(self.iter == 5 and len(copy) > 0):
                    # #     exit()
                    # copy = []
                    # for i in data:
                    #     copy.append(np.append(i,100))
                    in_bandwidth = {}
                    for k in range(len(data)):
                        featureset = data[k]
                        distances = [weightedEuclideanDist(featureset, centroids[centroid], [0.2, 0.4, 0.4, 0])
                                     for centroid in centroids]
                        classification = (distances.index(min(distances)))

                        centroid = centroids[classification]
                        # if int(centroid[0]) == 6:
                        #     print(centroid)
                        #     print(weightedEuclideanDist(featureset,centroid,[0.4,0.2,0.2,0.2]))
                        #     if((featureset[2] >= 46 and featureset[2] <= 49) and (featureset[1] >= 51 and featureset[1] <= 54) ):
                        #         print(featureset)
                        #     print(int(abs(featureset[0] - centroid[0])) < 4)
                        # if(self.iter == 1):
                        #     if weightedEuclideanDist(featureset,centroid,[0.2,0.4,0.4,0]) < self.radius:#custom function
                        #         try:
                        #             in_bandwidth[classification].append(featureset)
                        #         except:
                        #             in_bandwidth[classification] = []
                        #             in_bandwidth[classification].append(featureset)

                        #         copy[k][4] = classification

                        #     else:
                        #         copy[k][4] = 100
                        if (int(abs(featureset[0] - centroid[0])) < 2):  # need to change
                            # print(weightedEuclideanDist(featureset,centroid,[0.2,0.4,0.4,0]))
                            if int(weightedEuclideanDist(featureset, centroid, [0.2, 0.4, 0.4,
                                                                                0])) < self.radius:  # custom function
                                try:
                                    in_bandwidth[classification].append(featureset)
                                except:
                                    in_bandwidth[classification] = []
                                    in_bandwidth[classification].append(featureset)

                                copy[k][4] = classification
                            else:
                                copy[k][4] = 100

                        # else:
                        #     in_bandwidth_imgs = []
                        #     for i in prev_in_bandwidth:

                        #         prev_in_bandwidth_arr = [arr.tolist() for arr in prev_in_bandwidth[i]]
                        #         if list(featureset) in prev_in_bandwidth_arr:
                        #             in_bandwidth_arr = np.array(prev_in_bandwidth[i])
                        #             in_bandwidth_imgs = in_bandwidth_arr[:,0]
                        #             break
                        #     if(len(in_bandwidth_imgs) > 0):
                        #         if((max(in_bandwidth_imgs) - min(in_bandwidth_imgs)) < 4.0): #need to change
                        #         # print(weightedEuclideanDist(featureset,centroid,[0.2,0.4,0.4,0]))
                        #             if weightedEuclideanDist(featureset,centroid,[0.2,0.4,0.4,0]) < self.radius:#custom function
                        #                 try:
                        #                     in_bandwidth[classification].append(featureset)
                        #                 except:
                        #                     in_bandwidth[classification] = []
                        #                     in_bandwidth[classification].append(featureset)

                        #                 copy[k][4] = classification

                        #             else:
                        #             # if(self.iter == 6 and len(copy) > 0):
                        #             #     print("featureset {} and centroid {}".format(featureset,centroid))
                        #                 copy[k][4] = 100
                        #             # else set to bg

                        #         else:
                        #             centroids_len = np.max(list(centroids.keys()))
                        #             centroids[centroids_len] = featureset

                        # required_copy = copy[300:]

                        # print(required_copy)

                    # print("=======================")
                    # print(len(centroids))
                    # print(len(in_bandwidth))
                    # print("=======================")

                    # if(self.iter == 2):
                    #     print(centroids)
                    #     break
                    prev_in_bandwidth = in_bandwidth
                    for i in in_bandwidth:
                        if (len(in_bandwidth[i]) > 0):
                            new_centroid = np.average(in_bandwidth[i], axis=0)
                            new_centroids.append(new_centroid.tolist())

                    new_centroids_dict = {}
                    for i, j in enumerate(new_centroids):
                        new_centroids_dict[i] = j
                    # print(new_centroids_dict)
                    new_centroids1 = []
                    # # new_centroids_copy = deepcopy(new_centroids)
                    visited = []
                    visited_count = 0

                    for l, i in enumerate(new_centroids):
                        #     print(type(i))
                        if (not (i in visited)):
                            # print("======================")
                            # print(l)
                            # print(i)
                            # print("=======================")
                            arr = []
                            arr.append(i)
                            for k, j in enumerate(new_centroids):
                                # print("distance",int(weightedEuclideanDist(np.array(i),np.array(j))))
                                if (weightedEuclideanDist(np.array(i), np.array(j)) < self.radius):
                                    visited.append(j)
                                    arr.append(j)
                                    visited_count = visited_count + 1;
                            # print("=============================")
                            # if(l is 27):
                            #     print("centroids_arr",arr)
                            #     exit()
                            new_centroids1.append(tuple(np.average(arr, axis=0)))

                    # if(len(new_centroids) == len(new_centroids1)):
                    #     break

                    # print(len(new_centroids))
                    # print(len(new_centroids1))

                    # new_centroids1 = []
                    # # new_centroids_copy = deepcopy(new_centroids)
                    # visited = np.ndarray(shape=(len(new_centroids),4))
                    # visited_count = 0
                    # for i in new_centroids:
                    #     print(type(i))
                    #     if(not(i in visited)):
                    #         arr = []
                    #         arr.append(i)

                    #         for k,j in enumerate(new_centroids):
                    #             if(weightedEuclideanDist(i,j) < self.radius):
                    #                 visited[visited_count] = j
                    #                 arr.append(j)
                    #                 visited_count = visited_count+1;
                    #         new_centroids1.append(tuple(np.average(arr,axis=0)))
                    # exit()

                    uniques = list(set(new_centroids1))
                    # print(new_centroids)
                    # print("uniques")
                    # print(uniques)

                    prev_centroids = dict(centroids)

                    centroids = {}
                    for i in range(len(uniques)):
                        centroids[i] = np.array(uniques[i])

                    optimized = True

                    for i in centroids:  # add tolerance
                        if not np.array_equal(centroids[i], prev_centroids[i]):
                            optimized = False
                        if not optimized:
                            break

                    if optimized:
                        break
                    # print("after second for loop")
                self.centroids = centroids
                self.classifications = {}

                # for i in range(len(self.centroids)):
                #     self.classifications[i] = []

                # for featureset in data:
                #     #compare distance to either centroid
                #     distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                #     # print(distances.index(min(distances)))
                #     classification = (distances.index(min(distances)))

                #     # featureset that belongs to that cluster
                #     self.classifications[classification].append(np.uint8(featureset))


        clf = Mean_Shift()
        clf.fit(data)
        centroids = clf.centroids
        print(clf.iter)

        labels = np.array([100] * width * height * stack_length)
        labels = np.reshape(labels, (stack_length, width, height))


        copy = np.array(copy)
        copy = copy.astype(int)
        for i in copy:
            labels[int(i[0]), int(i[1]), int(i[2])] = i[4]
        labels = np.uint8(labels)
        labels_uniques = np.unique(labels)

        # eliminating cluster which are outside the disector box
        in_disector_array = []
        labels1 = deepcopy(labels)
        for index, img in enumerate(labels1):
            disector_percentage = 75
            disector_percentage = 100 * math.sqrt(disector_percentage / 100)
            disectorWidth = int(math.ceil(disector_percentage * min(img.shape[0], img.shape[1]) / 100))
            x = int(math.ceil((img.shape[0] - disectorWidth) / 2))
            y = int(math.ceil((img.shape[1] - disectorWidth) / 2))
            x1 = x + disectorWidth
            y1 = y
            x2 = x
            y2 = y + disectorWidth
            x3 = x1
            y3 = y2
            # blank_img = np.zeros((img.shape[0],img.shape[1],3))
            # cv2.rectangle(blank_img,(x,y),(x3,y3),color=(100,100,100),thickness = -1)
            # blank_img = np.uint8(blank_img[:,:,2])
            # print(type(blank_img[0,0]))
            # print(type(img[0,0]))
            # and_img = cv2.bitwise_and(blank_img,img)
            # cv2.imshow("and img",and_img)
            # cv2.waitKey()
            # cv2.imshow("original img",img)
            # cv2.waitKey()
            # if(index == 6):
            #     print(np.unique(img))
            cropped_img = img[x - 1:x1, y - 1:y2]
            # if(index == 6):
            #     print(np.unique(cropped_img))
            #     cv2.imshow("cropped img",cropped_img)
            #     cv2.waitKey()
            #     print("hello")
            # cv2.imshow("cropped img",cropped_img)
            # cv2.waitKey()
            in_disector_array = np.append(in_disector_array, np.unique(cropped_img))

        # print(in_disector_array)
        in_disector_array = np.uint16(in_disector_array)
        labels_set = set(np.unique(labels))
        in_disector_set = set(np.unique(in_disector_array))
        # # labels[not labels in in_disector_array] = 100
        out_disector_array = list(labels_set - in_disector_set)
        # print(out_disector_array)
        for i in out_disector_array:
            labels[labels == i] = 100

        # labels_path1 = os.path.join('./labels1',stackName)
        # if not os.path.exists(labels_path1):
        #     os.makedirs(labels_path1)
        # colors = []
        # for i in range(0,100):
        #     b = random.randint(10,255)
        #     g = random.randint(10,255)
        #     r = random.randint(10,255)
        #     colors.append([b,g,r])

        # result_dir = stackName
        # # curr_res_path = os.path.join(result_path,result_dir)
        # predicted_cluster_img_dir_path = os.path.join(mc_constants.predicted_clusters_images_path,mode,treshold_dir_name,stackName)
        # black_color = [0,0,0]
        # # print("clusters in 8th image",np.unique(labels[8]))
        # # print("clusters in 9th image",np.unique(labels[9]))
        # if not os.path.exists(predicted_cluster_img_dir_path):
        #     os.makedirs(predicted_cluster_img_dir_path)
        # uniques = np.unique(labels)
        # for i in range(stack_length):
        #     labeled_color = cv2.cvtColor(labels[i],cv2.COLOR_GRAY2BGR)
        #     # print(labeled_color.shape)
        #     for k in range(width):
        #         for l in range(height):
        #             if  labeled_color[k][l][0] == 100:
        #                 labeled_color[k][l] = black_color

        #             else:
        #                 index=np.where(uniques == labeled_color[k][l][0])[0][0]
        #                 labeled_color[k][l][0], labeled_color[k][l][1], labeled_color[k][l][2] = colors[index][0],colors[index][1],colors[index][2]

        #     # labeled_3 = cv2.merge([labeled_color[i],labeled_color[i],labeled_img[i]])
        #     (labeled_color,_,_,_) = PutDisectorOnImage(labeled_color,75)
        #     # cv2.imshow("labeled color",labeled_color)
        #     # cv2.waitKey()
        #     cv2.imwrite(os.path.join(labels_path1,sliceNames[i][:-4]+'.png'),labeled_color)

        # GROUPING CONSECUTIVE IMAGES FOR A CLUSTER

        labels_uniques = list(set(np.unique(labels)) - set([100]))
        max_label = np.max(labels_uniques)
        for k in labels_uniques:
            # print("=========================")
            # print(k)
            label_img_index = []
            consecutive_img_array = []
            for index, img in enumerate(labels):
                if k in img:
                    # print(index)
                    label_img_index.append(index)
                    consecutive_img_array = mc_constants.group_consecutives(label_img_index)
                    if len(consecutive_img_array) > 1:
                        label_index_array = np.argwhere(labels[index] == k)
                        for i in label_index_array:
                            labels[index, i[0], i[1]] = max_label + 1
                        max_label = max_label + 1
                        consecutive_img_array.pop(1)

            # print(k)
            # print(consecutive_img_array)

            # for i in consecutive_img_array[1:]:

        labels_uniques = np.unique(labels)
        # adjusting the centroid of the cluster comparing the intensity of the centroid of the blob in each image
        for k in labels_uniques:
            intensity_per_img = {}
            max_intensity = 0.0
            index_array = np.array([]);
            for index, img in enumerate(labels):

                if k in img and k != 100:
                    index_array = np.argwhere(img == k)
                    # print(index_array)
                    blank_img = np.zeros(img.shape)
                    # TALK TO PALAK
                    for i in index_array:
                        blank_img[i[0], i[1]] = 255
                    blank_img = np.uint8(blank_img)
                    _, _, _, cluster_centroid = cv2.connectedComponentsWithStats(blank_img, 8, cv2.CV_32S)
                    cluster_centroid = np.uint8(cluster_centroid[1])
                    input_img = images_array1[index];
                    centroid_intensity = input_img[cluster_centroid[1], cluster_centroid[0]]
                    if centroid_intensity > max_intensity:
                        max_intensity = centroid_intensity;
                        max_intensity_index = index
            index_array = np.argwhere(labels == k)
            if (index_array.size > 0):
                already_marked = False
                for i in index_array:
                    if (i[0] == max_intensity_index):
                        if (not already_marked):
                            already_marked = True
                    else:
                        labels[i[0], i[1], i[2]] = 100

        # labels_path2 = os.path.join('./labels2',stackName)
        # if not os.path.exists(labels_path2):
        #     os.makedirs(labels_path2)
        # colors = []
        # for i in range(0,100):
        #     b = random.randint(10,255)
        #     g = random.randint(10,255)
        #     r = random.randint(10,255)
        #     colors.append([b,g,r])

        # result_dir = stackName
        # # curr_res_path = os.path.join(result_path,result_dir)
        # predicted_cluster_img_dir_path = os.path.join(mc_constants.predicted_clusters_images_path,mode,treshold_dir_name,stackName)
        # black_color = [0,0,0]
        # # print("clusters in 8th image",np.unique(labels[8]))
        # # print("clusters in 9th image",np.unique(labels[9]))
        # if not os.path.exists(predicted_cluster_img_dir_path):
        #     os.makedirs(predicted_cluster_img_dir_path)
        # uniques = np.unique(labels)
        # for i in range(stack_length):
        #     labeled_color = cv2.cvtColor(labels[i],cv2.COLOR_GRAY2BGR)
        #     # print(labeled_color.shape)
        #     for k in range(width):
        #         for l in range(height):
        #             if  labeled_color[k][l][0] == 100:
        #                 labeled_color[k][l] = black_color

        #             else:
        #                 index=np.where(uniques == labeled_color[k][l][0])[0][0]
        #                 labeled_color[k][l][0], labeled_color[k][l][1], labeled_color[k][l][2] = colors[index][0],colors[index][1],colors[index][2]

        #     # labeled_3 = cv2.merge([labeled_color[i],labeled_color[i],labeled_img[i]])
        #     (labeled_color,_,_,_) = PutDisectorOnImage(labeled_color,75)
        #     # cv2.imshow("labeled color",labeled_color)
        #     # cv2.waitKey()
        #     cv2.imwrite(os.path.join(labels_path2,sliceNames[i][:-4]+'.png'),labeled_color)

        # for k in np.unique(labels):
        #     count = 0
        #     for index,img in enumerate(labels):
        #         if k in img and k != 100:
        #             count = count + 1

        # labels[not labels in list(labels_set - in_disector_set)] = 100
        # cv2.imshow("labels",labels[7])
        # cv2.waitKey()
        # print(np.unique(labels[7]))

        # for index in centroids:
        #     x = int(centroids[index][2])
        #     y = int(centroids[index][1])
        #     img_index = int(centroids[index][0])
        #     img = cv2.imread(os.path.join(marked_img_dir_path,sliceNames[img_index]))
        #     img = cv2.circle(img, (x,y), radius=0, color=(255, 0, 0), thickness=-1)
        #     cv2.imwrite(os.path.join(marked_img_dir_path,sliceNames[img_index]),img)


        for k in np.unique(labels):
            if np.count_nonzero(labels == k) < 4:
                labels[labels == k] = 100

        cells = []
        # adding the predicted centroids to json.
        labels_uniques = np.unique(labels)
        for k in labels_uniques:
            index_array = np.array([]);
            if k != 100:
                for index, img in enumerate(labels):
                    if k in img:
                        index_array = np.argwhere(img == k)
                        # print(index_array)
                        blank_img = np.zeros(img.shape)
                        for i in index_array:
                            blank_img[i[0], i[1]] = 255
                        blank_img = np.uint8(blank_img)

                        _, _, _, cluster_centroids = cv2.connectedComponentsWithStats(blank_img, 8, cv2.CV_32S)

                        for centroid in cluster_centroids[1:]:
                            cluster_centroid = np.uint8(centroid)
                            cells.append({"centroid": cluster_centroid.tolist(), "lineIds": [-1, -1], "sliceNo": index})

        cell_count = len(cells)
        total_count += cell_count

        ManualAnnotation.append({"StackName": stackNameFull, "cells": cells, "count": cell_count, "height": height, "width": width})

annotation_dir = {image_dir_name:ManualAnnotation,"all_stacks":len(ManualAnnotation),"total_count":total_count}


try:
    with open(annotation_path,'w') as fp:
        json.dump(annotation_dir, fp, sort_keys=True, indent=2)
except Exception as e:
    print("error while writing the json file:  {}".format(e))





