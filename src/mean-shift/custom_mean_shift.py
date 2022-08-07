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
loader = importlib.machinery.SourceFileLoader( 'mc_constants', r'C:\Users\KAVYA\Abhiram\microscopy\constants\constants.py' )
spec = importlib.util.spec_from_loader( 'mc_constants', loader )
mc_constants = importlib.util.module_from_spec( spec )
loader.exec_module( mc_constants)
# img = cv2.imread('./data/merged.png')

mode = mc_constants.algo_test
stack_length = mc_constants.stack_length

if(mode == mc_constants.train):
    image_path= mc_constants.slide1_64X64_train
    mask_path = mc_constants.slide1_64X64_masks_train

elif(mode == mc_constants.test):
    image_path= mc_constants.slide2_test
    mask_path = mc_constants.slide2_masks_test

elif(mode == mc_constants.valid):
    image_path= mc_constants.slide1_64X64_val
    mask_path = mc_constants.slide1_64X64_masks_val

elif(mode == mc_constants.algo_test):
    image_path = mc_constants.testing_img_dir
    mask_path =  mc_constants.testing_masks_dir


result_path = "./custom_ms_results/"
image_dir = mc_constants.image_dir
mask_img_path = mask_path

marked_img_path = mc_constants.marked_images_path
csv_results_path = mc_constants.csv_results_path


random.seed(44)

def PutDisectorOnImage(I,percentage=25):
    # if I not 3 channel then make it 3 channel
    if len(I.shape) != 3:
        I_new = np.empty((I.shape[0],I.shape[1],3),dtype=I.dtype)
        I_new[:,:,0] = I
        I_new[:,:,1] = I
        I_new[:,:,2] = I
        I = I_new
    percentage = 100 * math.sqrt(percentage / 100)
    disectorWidth = int(math.ceil(percentage * min(I.shape[0], I.shape[1]) / 100))
    x = int(math.ceil((I.shape[0] - disectorWidth) / 2))
    y = int(math.ceil((I.shape[1] - disectorWidth) / 2))



    #Inclusion
    I[x, range(y, y + disectorWidth)] = [0,255,0]
    I[range(x,x+disectorWidth),y+disectorWidth] = [0,255,0]



    #Exclusion
    I[range(x,x+disectorWidth),y] = [0,255,0]
    I[x+disectorWidth,range(y,y+disectorWidth)] = [0,255,0]
    return I,disectorWidth,x,y

# filter to reduce noise

# flatten the image
# flat_image = img.reshape((-1,3))
# flat_image = np.float32(flat_image)
image_dir = os.listdir(image_path)
masks = os.listdir(mask_img_path)
maskno=0

disector_percentage=75

result_json_list = []

# print(len(image))
min_error=100
best_accuracy = 0
best_treshold=0
max_recall = 0
# seed_treshold_arr = [10,14,21,23,25,27,29]
# data_treshold_arr = [10,12,14,16,18,20,22]
seed_treshold_arr = [14]
data_treshold_arr = [14]

# treshold_arr = [10]
# treshold_arr1 = [10]
testing = True

# rejected_path = os.path.join(path,'rejected')
# accepted_path = os.path.join(path,'accepted')

# if(testing):
    
#     try:
#         shutil.rmtree(rejected_path)
#     except OSError as e:
#         print("Error: %s : %s" % (rejected_path, e.strerror))
#     try:
#         shutil.rmtree(accepted_path)
#     except OSError as e:
#         print("Error: %s : %s" % (accepted_path, e.strerror))

# if(not os.path.exists(rejected_path)):
#     os.makedirs(rejected_path)

# if(not os.path.exists(accepted_path)):
#     os.makedirs(accepted_path)




for tres in seed_treshold_arr:
    for tres1 in data_treshold_arr:
        treshold_dir_name = "seed_threshold_"+str(tres)+"_data_threshold_"+str(tres1)
        total_tp=0
        total_fp=0
        total_fn=0
        total_gt=0
        total_cents=0
        result_csv_list= []
        centroids_json_arr = []
        for stackNo,stackName in enumerate(masks):
            print(stackName)
            stack_path = os.path.join(image_path,stackName)
            sliceNames  = os.listdir(stack_path)
            images = []  
            images1 = [] 
            centroids_arr = [] 
            mask_centroids_arr = []
            # print(sliceNames[0])
            # print(stackName)
            width=0
            height=0
            for sliceNo,sliceName in enumerate(sliceNames):   
                image = cv2.imread(os.path.join(image_path,stackName,sliceName), -1)#
                image = image/256
                image = np.uint8(image)
                # cv2.imshow("input image",image)
                # cv2.waitKey()       
                (width,height) = image.shape
                ret,image1 = cv2.threshold(image,tres,255,cv2.THRESH_TOZERO)
                # cv2.imshow("seed thresh image",image1)
                # cv2.waitKey()
                output = cv2.connectedComponentsWithStats(image1,8, cv2.CV_32S)
                (numLabels, ccws_labels, stats, centroids) = output
                # cv2.imshow("image1",image)
                # cv2.waitKey()
                # image = cv2.GaussianBlur(image,k,0)
                images.append(image)
                images1.append(image1)
                #finding the centroids of the mask
                # cv2.imshow('mask_img',cv2.imread(os.path.join(mask_path,stackName,sliceName[:-4]+'.png'))[:,:,2])
                # cv2.waitKey()
                (numLabels1,  ccws_labels1, stats1, mask_centroids) = cv2.connectedComponentsWithStats(cv2.imread(os.path.join(mask_img_path,stackName,sliceName[:-4]+'.png'))[:,:,2],8,cv2.CV_32S)

                for i in centroids[1:]:
                    centroids_arr.append(np.append(i,int(sliceNo)))
                # print(centroids_arr)
                for j in mask_centroids[1:]:
                    mask_centroids_arr.append(np.insert(j,0,int(sliceNo)))
            images_array= np.array(images) 
            images_array1 = np.array(images1)
            centroids_arr = np.array(centroids_arr) 
            mask_centroids_arr = np.array(mask_centroids_arr)
            # print(centroids_arr[0])
            # print(mask_centroids_arr[0])
            # print(images_array[0,0,1])
            # print(centroids_arr)
            data=[]
            data1=[]
            for z in range(stack_length):
                for x in range(width):
                    for y in range(height):
                        # data.append([z,x,y,images_array[z,x,y]])  
                        if(images_array[z,x,y] > tres1):
                            data.append([z,x,y,images_array[z,x,y]])
                        # if(images_array[z,x,y] > tres):
                        #     data1.append([z,x,y,images_array[z,x,y]])
                        # if(centroids_arr[z][x][0] == data[z   b n])
            for i in centroids_arr:
                data1.append([i[2],i[1],i[0],images_array1[int(i[2]),int(i[1]),int(i[0])]])
            data,data1=np.array(data),np.array(data1)
            data,data1 = np.float32(data),np.float32(data1)
            def  weightedEuclideanDist(a,b,w):
                q = a-b
                # print(np.sqrt((w*q*q).sum()))
                return np.sqrt((w*q*q).sum())
            count=0
            class Mean_Shift:
                def __init__(self, radius=4,max_iter=800):
                    self.radius = radius
                    self.max_iter = max_iter
                    self.iter = 0

                def fit(self, data):
                    centroids = {}
                    global copy
                    for i in range(len(data1)):
                        centroids[i] = data1[i]
                    while True:
                        self.iter=self.iter+1
                        new_centroids = []
                        if(self.iter == 2):
                            print(len(centroids))
                            exit()
                        copy = []
                        for i in data:
                            copy.append(np.append(i,100))
                        for j,i in enumerate(centroids):
                            in_bandwidth = []
                            centroid = centroids[i]
                            # if int(centroid[0]) == 6:
                            #     print(centroid)
                            for k in range(len(data)):
                                featureset = data[k]
                                # print(weightedEuclideanDist(featureset,centroid,[0.4,0.2,0.2,0.2]))
                                # if((featureset[2] >= 46 and featureset[2] <= 49) and (featureset[1] >= 51 and featureset[1] <= 54) ):
                                #     print(featureset)
                                # print(int(abs(featureset[0] - centroid[0])) < 4)
                                if(int(abs(featureset[0] - centroid[0])) < 4): #need to change
                                    # print(weightedEuclideanDist(featureset,centroid,[0.2,0.4,0.4,0]))
                                    if int(weightedEuclideanDist(featureset,centroid,[0.2,0.4,0.4,0])) < self.radius:#custom function     
                                        in_bandwidth.append(featureset)
                                        copy[k][4] = j
                                    #else set to bg
                                
                                #else create a new centroid
                                       
                            # required_copy = copy[300:]

                            # print(required_copy)
                            if(len(in_bandwidth) > 0):
                                new_centroid = np.average(in_bandwidth,axis=0)
                                print("new centroid",new_centroid)
                                exit()
                                new_centroids.append(tuple(new_centroid))
                        print(len(new_centroids))

                        uniques = sorted(list(set(new_centroids)))
                        print(len(uniques))
                        # print(new_centroids)
                        # print("uniques")
                        # print(uniques)

                        prev_centroids = dict(centroids)

                        centroids = {}
                        for i in range(len(uniques)):
                            centroids[i] = np.array(uniques[i])

                        optimized = True

                        for i in centroids:
                            if not np.array_equal(centroids[i], prev_centroids[i]):
                                optimized = False
                            if not optimized:
                                break
                            
                        if optimized:
                            break
                        # print("after second for loop")
                    self.centroids = centroids
                    self.classifications = {}

                    for i in range(len(self.centroids)):
                        self.classifications[i] = []
                        
                    for featureset in data:
                        #compare distance to either centroid
                        distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                        # print(distances.index(min(distances)))
                        classification = (distances.index(min(distances)))

                        # featureset that belongs to that cluster
                        self.classifications[classification].append(np.uint8(featureset))





            clf = Mean_Shift()
            clf.fit(data)
            centroids = clf.centroids
            print(clf.iter)
            # print(centroids)
            classifications = clf.classifications

            required_classifications = classifications[1]
            # print(required_classifications)

            labels = np.array([100]*width*height*stack_length)
            labels = np.reshape(labels,(stack_length,width,height))
            # print(classifications.keys())
            
            # for i in classifications.items():
            #     for j in i[1:]:
            #         for k in j:
            #             labels[k[0],k[1],k[2]] = i[0]
            # labels = np.uint8(labels)

            # print(images_array[0:5])
            centroids_json_arr.append({stackName: centroids})


            # for i in classifications:
            #     for j in classifications[i]:
            #         labels[j[0],j[1],j[2]]=i
            # print("copy:")
            # print(copy)

            # print(labels.shape)

            copy = np.array(copy)
            copy = copy.astype(int)
            for i in copy:
                    labels[int(i[0]),int(i[1]),int(i[2])] = i[4]
            labels = np.uint8(labels)


            if mode == mc_constants.algo_test:
                random.seed(221)
                labels_path = os.path.join('./labels',stackName)
                if not os.path.exists(labels_path):
                    os.makedirs(labels_path)

                colors = []
                for i in range(0,100):
                    b = random.randint(10,255)
                    g = random.randint(10,255)
                    r = random.randint(10,255)
                    colors.append([b,g,r])

                result_dir = stackName
                # curr_res_path = os.path.join(result_path,result_dir)
                predicted_cluster_img_dir_path = os.path.join(mc_constants.predicted_clusters_images_path,mode,treshold_dir_name,stackName)
                black_color = [0,0,0]
                # print("clusters in 8th image",np.unique(labels[8]))
                # print("clusters in 9th image",np.unique(labels[9]))
                if not os.path.exists(predicted_cluster_img_dir_path):
                    os.makedirs(predicted_cluster_img_dir_path)
                uniques = np.unique(labels)
                for i in range(stack_length):
                    labeled_color = cv2.cvtColor(labels[i],cv2.COLOR_GRAY2BGR)
                    # print(labeled_color.shape)
                    for k in range(width):
                        for l in range(height):
                            if  labeled_color[k][l][0] == 100:
                                labeled_color[k][l] = black_color
                            
                            else:
                                index=np.where(uniques == labeled_color[k][l][0])[0][0]
                                labeled_color[k][l][0], labeled_color[k][l][1], labeled_color[k][l][2] = colors[index][0],colors[index][1],colors[index][2]

                    # labeled_3 = cv2.merge([labeled_color[i],labeled_color[i],labeled_img[i]])
                    (labeled_color,_,_,_) = PutDisectorOnImage(labeled_color,75)
                    # cv2.imshow("labeled color",labeled_color)
                    # cv2.waitKey()
                    cv2.imwrite(os.path.join(labels_path,sliceNames[i][:-4]+'.png'),labeled_color)
            
            exit()
             
            # for i in np.unique(labels):
            #     count=0
            #     for j in labels: 

            # uniques = np.bincount(labels)
            # small_clusters={}
            # print(np.unique(labels))
        

          
            # marked_img_dir_path = os.path.join('./marked_images+'+str(tres)+"_15",'Neo_cx',stackName)
            # if(not os.path.exists(marked_img_dir_path)):
            #     os.makedirs(marked_img_dir_path)
            marked_img_dir_path1 = os.path.join(os.path.join(marked_img_path,mode,'seed_tres_'+str(tres)+"_data_tres_"+str(tres1)),stackName)
            if(not os.path.exists(marked_img_dir_path1)):
                os.makedirs(marked_img_dir_path1)
            color_images = [];
            labels_uniques = np.unique(labels)
            # np.delete(labels_uniques,labels_uniques==100)
            
            for index,img in enumerate(images_array1):
                img=np.reshape(img,(width,height,1))
                ch2 = np.zeros(img.shape)
                img  =np.concatenate((ch2,ch2,img),axis=2)
                img = np.uint8(img)
                color_images.append(img)
                (img,_,_,_) = PutDisectorOnImage(img,75)
                cv2.imwrite(os.path.join(marked_img_dir_path1,sliceNames[index][:-4]+'.png'),img)
            # for k in np.unique(labels):
            #     count_per_img = {}
            #     max_count = 0
            #     max_index = []
            #     index_array = np.array([]);
            #     for index,img in enumerate(labels):

            #         if k in img: 
            #             count_per_img = np.count_nonzero(img == k)
            #             if(count_per_img > max_count):
            #                 max_count = count_per_img
            #                 max_index  = index

            #     index_array = np.argwhere(labels[max_index] == k) 
                
            #     max_img = color_images[max_index]    
            #     print((index_array[0][0],index_array[0][1]))
            #     max_img = cv2.circle(max_img, (index_array[0][1],index_array[0][0]), radius=0, color=(255, 0, 0), thickness=-1)
            #     print(sliceNames[max_index])
            #     # cv2.namedWindow('marked_image',cv2.WINDOW_NORMAL)
            #     # cv2.resizeWindow('marked_image', 600,600)
            #     # cv2.imshow("marked_image",max_img)
            #     # cv2.waitKey()
            #     cv2.imwrite(os.path.join(marked_img_dir_path,sliceNames[max_index]),max_img) 

            

            #adjusting the centroid of cluster comparing the 95 percentile intensity of cluster in each stack
            # labels2 = deepcopy(labels)
            # for k in labels_uniques:
            #     intensity_per_img = {}
            #     max_intensity = 0.0
            #     index_array = np.array([]);
            #     if k != 100:
            #         for index,img in enumerate(labels):

            #             if k in img:
            #                 # index_array = np.argwhere(img == k)
            #                 zero_img = np.zeros(img.shape)
            #                 # for i in index_array:
            #                 #     zero_img[i[0],i[1]] = 255
            #                 zero_img[img == k] = 255
            #                 pixel_count = np.count_nonzero(zero_img == 255)
            #                 input_img  = images_array1[index];
                        
            #                 median_intensity = np.percentile(np.int8(input_img[zero_img == 255]),q=95)
            #                 # avg_intensity = intensity/pixel_count
            #                 # print("intensity",intensity)
            #                 if median_intensity > max_intensity:
            #                     max_intensity = median_intensity;
            #                     max_intensity_index = index
            #         index_array = np.argwhere(labels == k) 
            #         if(index_array.size > 0):
            #             already_marked = False
            #             for i in index_array:
                            
            #                 if(i[0] == max_intensity_index):
            #                     {k:max_intensity_index}
            #                     if(not already_marked):
            #                         max_img = color_images[max_intensity_index]    
            #                         max_img = cv2.circle(max_img, (i[2],i[1]), radius=0, color=(255, 0, 0), thickness=-1)
            #                         cv2.imwrite(os.path.join(marked_img_dir_path1,sliceNames[max_intensity_index][:-4]+'.png'),max_img)
            #                         already_marked = True 
            #                 else:
            #                     labels[i[0],i[1],i[2]] = 100
            #eliminating cluster which are outside the disector box
            in_disector_array = []
            labels1 = deepcopy(labels)
            for index,img in enumerate(labels1):
                disector_percentage = 75
                disector_percentage = 100 * math.sqrt(disector_percentage/ 100)
                disectorWidth = int(math.ceil(disector_percentage * min(img.shape[0],img.shape[1]) / 100))
                x = int(math.ceil((img.shape[0] - disectorWidth) / 2)) 
                y = int(math.ceil((img.shape[1] - disectorWidth) / 2))
                x1 = x+disectorWidth
                y1 = y
                x2 = x
                y2 = y+ disectorWidth
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
                cropped_img  = img[x - 1:x1, y-1:y2]
                # if(index == 6):
                #     print(np.unique(cropped_img))
                #     cv2.imshow("cropped img",cropped_img)
                #     cv2.waitKey()
                #     print("hello")
                # cv2.imshow("cropped img",cropped_img)
                # cv2.waitKey()
                in_disector_array =  np.append(in_disector_array,np.unique(cropped_img))


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


            #GROUPING CONSECUTIVE IMAGES FOR A CLUSTER
            
            labels_uniques = list(set(np.unique(labels)) - set([100]))
            max_label = np.max(labels_uniques)
            for k in labels_uniques:
                # print("=========================")
                # print(k)
                label_img_index = []
                consecutive_img_array = []
                for index,img in enumerate(labels):
                    if k in img:
                        # print(index)
                        label_img_index.append(index)
                        consecutive_img_array = mc_constants.group_consecutives(label_img_index)
                        if len(consecutive_img_array) > 1:
                            label_index_array = np.argwhere(labels[index] == k)
                            for i in label_index_array:
                                labels[index,i[0],i[1]] = max_label + 1
                            max_label= max_label + 1
                            consecutive_img_array.pop(1)



            

                           
                            

                # print(k)
                # print(consecutive_img_array)

                
                # for i in consecutive_img_array[1:]:








            labels_uniques = np.unique(labels)
            #adjusting the centroid of the cluster comparing the intensity of the centroid of the blob in each image
            for k in labels_uniques:
                intensity_per_img = {}
                max_intensity = 0.0
                index_array = np.array([]);
                for index,img in enumerate(labels):

                    if k in img and k != 100:
                        index_array = np.argwhere(img == k)
                        # print(index_array)
                        blank_img = np.zeros(img.shape)
                        #TALK TO PALAK
                        for i in index_array:
                            blank_img[i[0],i[1]] = 255
                        blank_img = np.uint8(blank_img)
                        _,_,_,cluster_centroid = cv2.connectedComponentsWithStats(blank_img,8,cv2.CV_32S)
                        cluster_centroid = np.uint8(cluster_centroid[1])
                        input_img  = images_array1[index];
                        centroid_intensity = input_img[cluster_centroid[1],cluster_centroid[0]]
                        if centroid_intensity > max_intensity:
                            max_intensity = centroid_intensity;
                            max_intensity_index = index
                index_array = np.argwhere(labels == k) 
                if(index_array.size > 0):
                    already_marked = False
                    for i in index_array:
                        if(i[0] == max_intensity_index):
                            if(not already_marked):
                                max_img = color_images[max_intensity_index]    
                                max_img = cv2.circle(max_img, (i[2],i[1]), radius=0, color=(255, 0, 0), thickness=-1)
                                cv2.imwrite(os.path.join(marked_img_dir_path1,sliceNames[max_intensity_index][:-4]+'.png'),max_img)
                                already_marked = True 
                        else:
                            labels[i[0],i[1],i[2]] = 100

            
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

           
            for index,value in enumerate(mask_centroids_arr):
                x = int(value[1])
                y = int(value[2])
                img_index = int(value[0])
                img = cv2.imread(os.path.join(marked_img_dir_path1,sliceNames[img_index][:-4]+'.png'))
                img = cv2.circle(img, (x,y), radius=0, color=(0, 255, 0), thickness=-1)
                cv2.imwrite(os.path.join(marked_img_dir_path1,sliceNames[img_index][:-4]+'.png'),img)
                
            for k in np.unique(labels):
                if np.count_nonzero(labels == k) < 4:
                    print("in if")
                    labels[labels == k] = 100

            



            # colors = [[0,0,128],[40,110,170],[0,128,128],[128,128,0],[128,0,0],[75,25,230],[48,130,245],[25,225,255],[60,245,210],[75,180,60],[240,240,70],[200,130,0],[180,30,145],[230,50,240],[128,128,128],[212,190,250],[180,215,255],[200,250,255],[195,255,170],[255,190,220],[255,255,255]]#add few more colors
            colors = []
            for i in range(0,100):
                b = random.randint(10,255)
                g = random.randint(10,255)
                r = random.randint(10,255)
                colors.append([b,g,r])

            result_dir = stackName
            # curr_res_path = os.path.join(result_path,result_dir)
            predicted_cluster_img_dir_path = os.path.join(mc_constants.predicted_clusters_images_path,mode,treshold_dir_name,stackName)
            black_color = [0,0,0]
            # print("clusters in 8th image",np.unique(labels[8]))
            # print("clusters in 9th image",np.unique(labels[9]))
            if not os.path.exists(predicted_cluster_img_dir_path):
                os.makedirs(predicted_cluster_img_dir_path)
            uniques = np.unique(labels)
            for i in range(stack_length):
                labeled_color = cv2.cvtColor(labels[i],cv2.COLOR_GRAY2BGR)
                # print(labeled_color.shape)
                for k in range(width):
                    for l in range(height):
                        if  labeled_color[k][l][0] == 100:
                            labeled_color[k][l] = black_color
                        
                        else:
                            index=np.where(uniques == labeled_color[k][l][0])[0][0]
                            labeled_color[k][l][0], labeled_color[k][l][1], labeled_color[k][l][2] = colors[index][0],colors[index][1],colors[index][2]

                # labeled_3 = cv2.merge([labeled_color[i],labeled_color[i],labeled_img[i]])
                (labeled_color,_,_,_) = PutDisectorOnImage(labeled_color,75)
                # cv2.imshow("labeled color",labeled_color)
                # cv2.waitKey()
                cv2.imwrite(os.path.join(predicted_cluster_img_dir_path,sliceNames[i][:-4]+'.png'),labeled_color)
            tp=0
            fp=0
            fn=0
            gt=len(mask_centroids_arr)
            cents = len(np.unique(labels)) - 1
            # print(mask_centroids_arr[0])
            # print(np.unique(labels))
            # print("width",width)
            # print("height",height)
            # print(mask_centroids_arr)
            # print(labels.shape)
            for i,j in enumerate(np.int64(mask_centroids_arr)):
            
                cl = labels[j[0],j[2],j[1]]
                if labels[j[0],j[2],j[1]] != 100:
                    labels[labels == cl] = 100
                    tp+=1
                else:
                    fn+=1
            fp = len(np.unique(labels))-1



            print("True Positives: ",tp)
            print("False Positives: ",fp)
            print("False Negatives:",fn)
            print("groundtruth: ",gt)
            print("predicted clusters in the stack",cents)
            total_tp+=tp
            total_fp+=fp
            total_fn+=fn
            total_gt+=gt
            total_cents+=cents
            if(tp > 0):
                Stack_accuracy = round(tp/(tp+fn+fp),2)
                Stack_recall = round(tp/(tp+fn),2)
                Stack_precision = round(tp/(tp+fp),2)
                Stack_f1_score = round(2*Stack_precision*Stack_recall/(Stack_precision+Stack_recall),2)
                Stack_error = round(abs(tp+fp-gt)/gt,2)

            else:
                Stack_accuracy = 0
                Stack_recall = 0
                Stack_precision = 0
                Stack_f1_score = 0
                Stack_error = 100
                print("0 tp's")

            # if(Stack_f1_score < 0.70):
            #     shutil.move(stack_path,os.path.join(rejected_path,stackName))

            # elif(Stack_f1_score >=0.70):
            #     shutil.move(stack_path,os.path.join(accepted_path,stackName))


            result_csv_list.append({"Stack Name":stackName,"TP":tp,"FP":fp,"FN":fn,"GT":gt,"accuracy":Stack_accuracy,"f1":Stack_f1_score,"recall":Stack_recall,"precision":Stack_precision,"error":Stack_error})
        
        # with open("./ms_predicted_centroids_"+str(tres)+".json", 'w') as fp:
        #     json.dump(centroids_json_arr, fp, sort_keys=True, indent=2)
        print("tres1",tres)
        print("tres2",tres1)
        if(not os.path.exists(os.path.join(csv_results_path,mode))):
            os.makedirs(os.path.join(csv_results_path,mode))

        result_csv_df = pd.DataFrame(result_csv_list)
        try:
            result_csv_df.to_csv(os.path.join(csv_results_path,mode,'seed_tres_'+str(tres)+'_data_tres_'+str(tres1)+'.csv'))
        except:
            print("Error while storing the results for each stack")

        print("Total true positives: ",total_tp)
        print("Total false positives: ",total_fp)
        print("Total false negatives:", total_fn)
        print("Total Groundtruth: ",total_gt)
        print("Total predicted clusters in the stack",total_cents)
        if(total_tp > 0):
            accuracy = round(total_tp/(total_tp+total_fn+total_fp),2)
            recall = round(total_tp/(total_tp+total_fn),2)
            precision = round(total_tp/(total_tp+total_fp),2)
            f1_score = round(2*precision*recall/(precision+recall),2)
            error = round(abs(total_tp+total_fp-total_gt)/total_gt,2)
        else:
            accuracy = 0
            recall = 0
            precision = 0
            fl_score=  0
            error = 100
            print("0 true positives")


        print("threshold",tres)
        print("error rate",error)
        print("Accuracy: ",accuracy)
        print("Recall: ",recall)
        print("precision: ", precision)
        print("f1_score: ",f1_score)

        result_json_list.append({"seed threshold":tres,"data treshold":tres1, "error rate":error,"Accuracy":accuracy,"Recall":recall,"precision": precision,"f1_score":f1_score,"Total true positives":total_tp,"Total false positives":total_fp,"Total false negatives": total_fn,"Total Groundtruth":total_gt})
        try:
            with open(os.path.join(csv_results_path,mode,'total_results.json'),'w') as fp:
                json.dump(result_json_list,fp,sort_keys=True,indent = 2)
        except:
            print("Error while saving the total results as json")
        if(recall>max_recall):
            min_error=error
            best_treshold=tres
            best_accuracy = accuracy
            best_precision = precision
            best_f1_score = f1_score
            best_recall = recall

print("---------------------------------")
print("Best Treshold ",best_treshold)
print("min error rate ",min_error)
print( "best_accuracy ",best_accuracy) 
print("best recall ",best_recall)
print("best_precision ",best_precision)
print("best_f1_score ",best_f1_score)


# with open("./ResultList.json", 'a') as fp:
#     json.dump(result_json_list, fp, sort_keys=True, indent=2)  

if(not os.path.exists(os.path.join(csv_results_path,mode))):
    os.makedirs(os.path.join(csv_results_path,mode))
result_json_df = pd.DataFrame(result_json_list)
try:
    result_json_df.to_csv(os.path.join(csv_results_path,mode,'ms_total_results_tc2_1_tres_latest.csv'))
except:
    print("Error while saving the total results as csv")


