# from mask_annotation_tool import annotate
import enum
import numpy as np
import pandas as pd
import cv2
import os
from copy import deepcopy
import shutil
import json
from scipy.interpolate import splprep, splev
import scipy.ndimage as ndimage
# import tippy.segmentations as se
# import tippy.basic_operations as bo
# import tippy.display_operations as do
from region_growing_code import region_growing
# img = cv2.imread('./data/merged.png')
path=r'D:\mircroscopy-1\16bitimages\Slide1'
annotation_path = os.path.join(path,'count_annotaion')
result_path = "./contour_images/"

# filter to reduce noise

# flatten the image
# flat_image = img.reshape((-1,3))
# flat_image = np.float32(flat_image)
image_folder = 'NeoCx'
image_path = os.path.join(path,'NeoCx')
slide_name='Slide1_NeoCx'
image_dir = os.listdir(image_path)
count_annotations = os.listdir(path+r'\count_annotaion')[0]  
try:
    with open(os.path.join(annotation_path,count_annotations), 'r') as ref_fp:
        annotation_dict = json.load(ref_fp)
except:
    print('Reference annotation json not available.')
annotation_dict = annotation_dict['Neo_cx']
centroid_dict = {}
for stackNo,stack in enumerate(annotation_dict):
    stackName = stack['StackName'].split('_')
    centroid_dict[stackName[-2]+'_'+stackName[-1]] = []
    for cell in stack['cells']:
        centroid_arr = cell['centroid']
        centroid_arr.insert(0,cell['sliceNo'])
        centroid_dict[stackName[-2]+'_'+stackName[-1]].append(centroid_arr)

maskno=0

# print(len(image))
max_recall=0
best_treshold=0
treshold_arr = [39]
labels_arr  = []
for tres in treshold_arr:
    total_tp=0
    total_fp=0
    total_fn=0
    total_gt=0
    total_cents=0
    stack=0 
    dice_coeff_arr = []
    for stack in image_dir:
        images = []  
        images1 = [] 
        mask_labels = []
        centroids_arr = [] 
        mask_centroids_arr = []
        k=(7,7)
        print(stack)
        # print(sliceNames[0])
        # print(masks[stack])
        width=0
        height=0
        slices = os.listdir(os.path.join(path,image_folder,stack))
        for sliceNo,sliceName in enumerate(slices): 
            img_path = os.path.join(path,image_folder,stack,sliceName)  
            image = cv2.imread(img_path, -1)#
            image = image/255
            input_img = image
            image = np.uint8(input_img)
            (width,height) = image.shape
            ret,image1 = cv2.threshold(image,tres,255,cv2.THRESH_TOZERO)
            output = cv2.connectedComponentsWithStats(image1,8, cv2.CV_32S)
            (numLabels, labels, stats, centroids) = output
            # cv2.imshow("image1",image)
            # cv2.waitKey()
            # image = cv2.GaussianBlur(image,k,0)
            images.append(image)
            images1.append(input_img)
            #finding the centroids of the mask
            # mask_img = cv2.resize(cv2.imread(os.path.join(path+'\masks',masks[sliceNo+stack]))[:,:,2],(width,height),cv2.INTER_CUBIC)
            # (numLabels1, labels1, stats1, mask_centroids) = cv2.connectedComponentsWithStats(mask_img,8,cv2.CV_32S)   
            # print(np.unique(numLabels1))
            # cv2.imshow("labels1",labels1)
            # mask_labels.append(labels1)
            for i in centroids[1:]:
                centroids_arr.append(np.append(i,int(sliceNo)))
            # for j in mask_centroids[1:]:
            #     mask_centroids_arr.append(np.insert(j,0,int(sliceNo)))
        images_array= np.array(images) 
        images_array1 = np.array(images1)
        mask_labels = np.array(mask_labels)
        # centroids_arr = np.array(centroids_arr) 
        try:
            mask_centroids_arr = np.array(centroid_dict[stack])
        except KeyError:
            print("Count annotation not found")
            continue

        # print(images_array[0,0,1])
        # print(centroids_arr)
        data=[]
        data1=[]
        print(images_array.shape)
        for z in range(10):
            for x in range(width):
                for y in range(height):
                    # data.append([z,x,y,images_array[z,x,y]])  
                    if(images_array[z,x,y] > 20):
                        data.append([z,x,y,images_array[z,x,y]])
                    # if(centroids_arr[z][x][0] == data[z   b n])
        for i in centroids_arr:
            data1.append([i[2],i[1],i[0],images_array[int(i[2]),int(i[1]),int(i[0])]])
        data,data1=np.array(data),np.array(data1)
        data,data1 = np.float32(data),np.float32(data1)
        copy=[]
        for i in data:
            copy.append(np.append(i,-1))
        def  weightedEuclideanDist(a,b,w):
            q = a-b
            return np.sqrt((w*q*q).sum())
        class Mean_Shift:
            def __init__(self, radius=4,max_iter=800):
                self.radius = radius
                self.max_iter = max_iter

            def fit(self, data):
                centroids = {}

                for i in range(len(data1)):
                    centroids[i] = data1[i]
                count=0
                while True:
                    new_centroids = []
                    for j,i in enumerate(centroids):
                        in_bandwidth = []
                        centroid = centroids[i]
                        for k in range(len(data)):
                            featureset = data[k]
                            # print(weightedEuclideanDist(featureset,centroid,[0.4,0.2,0.2,0.2]))
                            if(abs(featureset[0] - centroid[0]) < 4):
                                if weightedEuclideanDist(featureset,centroid,[0.2,0.4,0.4,0]) < self.radius:#custom function     
                                    in_bandwidth.append(featureset)
                                    copy[k][4] = j
                        if(len(in_bandwidth) > 0):
                            new_centroid = np.average(in_bandwidth,axis=0)
                        # print(new_centroid)
                            new_centroids.append(tuple(new_centroid))
                        # print("after third for loop")
                    uniques = sorted(list(set(new_centroids)))
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
                    count+=1
                    # print("after second for loop")
                self.centroids = centroids
                self.classifications = {}

                for i in range(len(self.centroids)):
                    self.classifications[i] = []
                    
                for featureset in data:
                    #compare distance to either centroid
                    distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                    #print(distances)
                    classification = (distances.index(min(distances)))

                    # featureset that belongs to that cluster
                    self.classifications[classification].append(np.uint8(featureset))




        clf = Mean_Shift()
        clf.fit(data)
        centroids = clf.centroids
        classifications = clf.classifications
        # print("classifications:",classifications)
        labels = np.array([100]*width*height*10)
        labels = np.reshape(labels,(10,width,height))
        # for i in classifications:
        #     for j in classifications[i]:
        #         labels[j[0],j[1],j[2]]=i
        # print("copy:")
        # print(copy)

        # print(labels.shape)

        for i in copy:
            if i[4] == -1 or i[4] > len(centroids):
                labels[int(i[0]),int(i[1]),int(i[2])] = 100
            else: 
                labels[int(i[0]),int(i[1]),int(i[2])] = i[4]
        labels = np.uint8(labels)
        # for i in np.unique(labels):
        #     count=0
        #     for j in labels:

        # uniques = np.bincount(labels)
        # small_clusters={}
        # print(np.unique(labels))
        # for k in np.unique(labels):
        #     if np.count_nonzero(labels == k) < 20:
        #         labels[labels == k] = 100


        # colors = [[0,0,128],[40,110,170],[0,128,128],[128,128,0],[128,0,0],[75,25,230],[48,130,245],[25,225,255],[60,245,210],[75,180,60],[240,240,70],[200,130,0],[180,30,145],[230,50,240],[128,128,128],[212,190,250],[180,215,255],[200,250,255],[195,255,170],[255,190,220],[255,255,255],[0,0,0]]#add few more colors
        result_dir = stack
        # curr_res_path = result_path+result_dir
        # if not os.path.exists(curr_res_path):
        #     os.mkdir(curr_res_path)
        # if not os.path.exists(path+r'\accepted\Neo_cx'):
        #             # print("in not os.path")
        #     os.makedirs(path+r'\accepted\Neo_cx')
        # if not os.path.exists(path+image_folder+'_annotated'+r'\Neo_cx'):
        #     os.makedirs(path+image_folder+'_annotated'+r'\Neo_cx')  
        # if not os.path.exists(path+r'\rejected'):
        #             # print("in not os.path")
        #     os.mkdir(path+r'\rejected')
        result_dir_split = result_dir[0].split('_')
        # print(result_dir_split)
        accepted_path = os.path.join(path+r'\accepted\Neo_cx',stack)
        annotation_path=os.path.join(path+r'\accepted'+'_annotated'+r'\Neo_cx',stack)
        if not os.path.exists(accepted_path):
            os.makedirs(accepted_path)
        if not os.path.exists(annotation_path):
            os.makedirs(annotation_path)
        for x,y in enumerate(images_array1):
            input_img1=np.reshape(y,(width,height,1))
            ch2 = np.zeros(input_img1.shape)
            input_img1  =np.concatenate((ch2,ch2,input_img1),axis=2)
            input_img1 = np.uint8(input_img1)
            ann_img = np.zeros((width,height))
            cv2.imwrite(os.path.join(annotation_path,str(x)+'.png'),ann_img)
            cv2.imwrite(os.path.join(accepted_path,str(x)+'.png'),input_img1)
            # shutil.copyfile()
        # print(accepted_path)
        rejected_path = os.path.join(path+r'\rejected')
        # if not os.path.exists(curr_res_path):
        #     os.mkdir(curr_res_path)
        # uniques = np.unique(labels)
        # for i in range(10):

        #     labeled_color = cv2.cvtColor(labels[i],cv2.COLOR_GRAY2BGR)
        #     # print(labeled_color.shape)
        #     for k in range(width):
        #         for l in range(height):
        #             if  labeled_color[k][l][0] == 100:
        #                 labeled_color[k][l][0] = colors[21][0]
        #                 labeled_color[k][l][1] = colors[21][1]
        #                 labeled_color[k][l][2] = colors[21][2]
        #             else:
        #                 index=np.where(uniques == labeled_color[k][l][0])[0][0]
        #                 labeled_color[k][l][0], labeled_color[k][l][1], labeled_color[k][l][2] = colors[index][0],colors[index][1],colors[index][2]



        #     # labeled_3 = cv2.merge([labeled_color[i],labeled_color[i],labeled_img[i]])
        #     cv2.imwrite(curr_res_path+'/'+str(i)+'.png',labeled_color)
        tp=0
        fp=0
        fn=0
        gt=len(mask_centroids_arr)
        cents = len(np.unique(labels)) - 1
        # print(mask_centroids_arr[0])
        # print(np.unique(labels))
        
        # dice_match=[]
        # for i,j in enumerate(mask_images):
        #     dice_m=0
        #     particles = j[j>0]
        #     # print(particles.shape)
        #     clusters = labels[i][labels[i] != 100]
        #     # c_count=np.count(clusters)
        #     print(clusters.shape)
        #     # print(c_count)
        #     for x,k in enumerate(j):
        #         for y,l in enumerate(k):
        #             if l!=0 and labels[i,x,y] != 100:
        #                 dice_m+=1
        #     dice_match.append([i,dice_m]) 
        
        # print(dice_match)
        # exit()






             
        dice_coeff = []
        flag2=False
        flag3=False

        for i,j in enumerate(np.int64(mask_centroids_arr)):
            flag=False
            cl_arr= np.array([])
            input_img = deepcopy(images_array1[j[0]])
            if labels[j[0],j[2],j[1]] != 100:

                flag2=True
                tp+=1
                cl = labels[j[0],j[2],j[1]]
                # print("image",j[0])
                # print("label",cl)
                cl_arr = deepcopy(labels[j[0]])
                # cl_arr[cl_arr < cl] = 100
                # cv2.imshow("cl",np.uint8(cl_arr)*50)
                # cv2.waitKey()
                print(j)
                print(cl)
                annotated_img1 = cv2.imread(os.path.join(annotation_path,str(j[0])+".png"),-1)
                print("first")
                # cv2.imshow("annotated_img1",annotated_img1)
                # cv2.waitKey()
                # cv2.imshow("0th step",annotated_img1)
                # cv2.waitKey()
                # for k,x in enumerate(cl_arr):
                #     for l,y in enumerate(x):
                #         if y == cl:
                #             annotated_img1[k][l]=255
                
                # cv2.imshow("label_image",labels[0])
                # cv2.imshow("1st step",annotated_img1)
                # cv2.waitKey()
                
                # print(cl)
                cl_arr[cl_arr == cl] = 255
                cl_arr[cl_arr != 255] = 100
                cl_arr[cl_arr == 100] = 0
                cl_contours,h2 = cv2.findContours(cl_arr,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                smoothened=[]
                for l,y in enumerate(cl_contours):
                    if(cv2.pointPolygonTest(y,(int(j[1]),int(j[2])),True) >= 0):
                        emp_img = np.zeros((width,height))
                        #boundary_smoothing
                        ##spline -- start
                        # x,y = y.T
                         # Convert from numpy arrays to normal arrays
                        # x = x.tolist()[0]
                        # y = y.tolist()[0]
                         # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
                        # tck, u = splprep([x,y], u=None, s=1.0, per=1)
                        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
                        # u_new = np.linspace(u.min(), u.max(), 25)
                        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                        # x_new, y_new = splev(u_new, tck, der=0)
                        # Convert it back to numpy format for opencv to be able to display it
                        # res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
                        # smoothened.append(np.asarray(res_array, dtype=np.int32))
                        ##spline--end
                        ##approxPolyDp--start
                        # epsilon = 0.1*cv2.arcLength(y,True)
                        # approx_cl_contours=cv2.approxPolyDP(cl_contours[l],epsilon,True)
                        cv2.drawContours(emp_img,cl_contours,l,(255,255,255),-1)
                        ##approxPolyDp--End
                        #GaussBlur--start
                        # gauss_clusters = [ndimage.gaussian_filter(y, sigma=2, order=0)]
                        #GaussBlur--end
                        # cv2.drawContours(emp_img,gauss_clusters,0,(255,255,255),-1)
                        new_img = (np.logical_or(annotated_img1,emp_img).astype(int))*255
                        print("second")
                        # cv2.imshow("new_image",np.uint8(new_img))
                        # cv2.waitKey()
                        cv2.imwrite(os.path.join(annotation_path,str(j[0])+".png"),new_img)



                # mask[image<intensity]=0
                # cv2.imshow("cl1",np.uint8(cl_arr)*50)
                # cv2.waitKey()
                # print(mask_labels)
                # ml=mask_labels[j[0],j[2],j[1]]
                # ml_array = mask_labels[j[0]]
                # ml_array[ml_array != ml] = 0
                
                
                # m_contours,h1 = cv2.findContours(np.uint8(ml_array)*255,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                # cl_contours,h2 = cv2.findContours(cl_arr,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                dice_match= 0 
                m_size = 0
                c_size = 0
                # for k,l in enumerate(ml_array):
                #     for m,n in enumerate(l):
                #         if n == ml:
                #             m_size +=1
                #             # m_contour.append([[int(k),int(m)]])
                #         if cl_arr[k,m] == cl:
                #             c_size+=1
                #             # c_contour.append([k,m])
                #         if n == ml and cl_arr[k,m] == cl:
                #             dice_match+=1  #use
                # dice_match = np.logical_and(ml_array == ml,cl_arr == cl)
                # dice_denom = np.logical_or(ml_array == ml,cl_arr == cl)
                # print(m_contours)
                input_img = images_array1[j[0]]
                input_img=np.reshape(input_img,(width,height,1))
                ch2 = np.zeros(input_img.shape)
                input_img  =np.concatenate((ch2,ch2,input_img),axis=2)
                input_img = np.uint8(input_img)
                cv2.imwrite(os.path.join(accepted_path,str(j[0])+".png"),input_img)
                # contoured_img = cv2.drawContours(input_img,contours=m_contours,contourIdx=0,color=(0,255,0),thickness=1)
               
                # cv2.line(input_img,(x-1,y+1),(x+1,y-1),(255,0,0),1)
                # cv2.line(input_img,(x-1,y-1),(x+1,y+1),(255,0,0),1)
                # if os.path.exists(os.path.join(accepted_path,str(j[0])+'.png')):
                #     input_img = cv2.imread(os.path.join(accepted_path,str(j[0])+'.png'),-1)
                # contoured_img = cv2.drawContours(input_img,contours=cl_contours,contourIdx=0,color=(255,0,0),thickness=1)
                # contours, hierarchy = cv2.findContours(images_array[j[0]], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # print(contours[0])

                # cv2.imwrite("./contour"+str(i)+".png",contoured_img)
               
                # cv2.imwrite(os.path.join(annotation_path,str(j[0])+".png"),annotated_img1)
                # dice_c = round(2*dice_match/(m_size+c_size),2)
                # print("-------")
                # print(dice_match)
                # print(m_size)
                # print(c_size)
                # print("-------")
                # dice_coeff.append([dice_c,m_size,c_size])
                labels[labels == cl] = 100
            else:
                flag=True
                fn+=1
                if flag3 == True:
                    if not os.path.exists(rejected_path):
                        os.makedirs(rejected_path)
                    # print(rejected_path)
                    rejected_stack_path = os.path.join(rejected_path,stack)
                    if os.path.exists(rejected_stack_path):
                        shutil.rmtree(rejected_stack_path)
                    shutil.move(accepted_path,rejected_path)
                    break
                tres_results_path = os.path.join('./tresholding_result/',str(stack))
                regionGrowing_results_path = os.path.join('./region_growing_result/',str(stack))
                otsu_tres_results_path = os.path.join('./otsu_tresholding_result/',str(stack))
                if not os.path.exists(regionGrowing_results_path):
                        os.makedirs(regionGrowing_results_path)
                if not os.path.exists(tres_results_path):
                        os.makedirs(tres_results_path)
                if not os.path.exists(otsu_tres_results_path):
                        os.makedirs(otsu_tres_results_path)
                input_img=np.uint8(input_img)
                print(j)
                out_img=region_growing.apply_region_growing(input_img,(j[1],j[2]))
                # cv2.imshow("region growing output",out_img)
                # cv2.waitKey()
                # out_img = se.simple_region_growing(input_img,(j[1],j[2]), 85)
                annotated_img1 = cv2.imread(os.path.join(annotation_path,str(j[0])+".png"),-1)
                reg_grow_contours,hv2 = cv2.findContours(out_img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                tres_input_img = deepcopy(input_img)
                otsu_tres_in_img = deepcopy(input_img)
                ret,img1 = cv2.threshold(tres_input_img,30,255,cv2.THRESH_BINARY)
                ret,img2 = cv2.threshold(otsu_tres_in_img,30,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                input_img=np.reshape(input_img,(width,height,1))
                ch2 = np.zeros(input_img.shape)
                input_img  =np.concatenate((ch2,ch2,input_img),axis=2)
                input_img = np.uint8(input_img)
                tres_out_img = deepcopy(input_img)
                otsu_tres_out_img = deepcopy(input_img)
                reg_grow_input_img = deepcopy(input_img)


                tres_contours,hv3 = cv2.findContours(img1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                otsu_tres_contours,hv3 = cv2.findContours(img2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for l,y in enumerate(tres_contours):
                    if(cv2.pointPolygonTest(y,(int(j[1]),int(j[2])),True) >= 0):
                        # emp_img = np.zeros((width,height))
                        # cv2.drawContours(emp_img,cl_contours,l,(255,255,255),-1)
                        # new_img = (np.logical_or(annotated_img1,emp_img).astype(int))*255
                        # print("second")
                        # cv2.imshow("new_image",np.uint8(new_img))
                        # cv2.waitKey()
                        cv2.drawContours(tres_out_img,tres_contours,l,(255,0,0),1)
                        cv2.imwrite(os.path.join(tres_results_path,str(j[0])+".png"),tres_out_img)
                for l,y in enumerate(otsu_tres_contours):
                    if(cv2.pointPolygonTest(y,(int(j[1]),int(j[2])),True) >= 0):
                        # emp_img = np.zeros((width,height))
                        # cv2.drawContours(emp_img,cl_contours,l,(255,255,255),-1)
                        # new_img = (np.logical_or(annotated_img1,emp_img).astype(int))*255
                        # print("second")
                        # cv2.imshow("new_image",np.uint8(new_img))
                        # cv2.waitKey()
                        cv2.drawContours(otsu_tres_out_img,tres_contours,l,(255,0,0),1)
                        cv2.imwrite(os.path.join(otsu_tres_results_path,str(j[0])+".png"),tres_out_img)


                emp_img = np.zeros((width,height))
                cv2.drawContours(emp_img,contours=reg_grow_contours,contourIdx=-1,color=(255,255,255),thickness=-1)
                new_img = (np.logical_or(annotated_img1,emp_img).astype(int))*255
                cv2.imwrite(os.path.join(annotation_path,str(j[0])+".png"),new_img)
                cv2.drawContours(reg_grow_input_img,reg_grow_contours,-1,(255,0,0),1)
                cv2.imwrite(os.path.join(regionGrowing_results_path,str(j[0])+".png"),reg_grow_input_img)
                
              
                # if not os.path.exists('./region_growing_result/'+str(stack)):
                #             os.makedirs('./region_growing_result/'+str(stack))
                # cv2.drawContours(input_img,contours=contours,contourIdx=-1,color=(255,0,0),thickness=1)
                # cv2.imwrite('./region_growing_result/'+str(stack)+'/'+str(j[0])+'.png',input_img)
                
                # for l,y in enumerate(contours):
                #     if(cv2.pointPolygonTest(y,(int(j[1]),int(j[2])),True) >= 0):
                #         input_img=np.reshape(input_img,(width,height,1))
                #         ch2 = np.zeros(input_img.shape)
                #         input_img  =np.concatenate((ch2,ch2,input_img),axis=2)
                #         input_img = np.uint8(input_img)
                #         annotated_img1 = (cv2.imread(os.path.join(annotation_path,str(j[0])+".png"),-1))//255

                #         # cv2.imshow("before",annotated_img1)
                #         # cv2.waitKey()
                #         emp_img = np.zeros((width,height))
                #         cv2.drawContours(emp_img,contours=contours,contourIdx=l,color=(255,255,255),thickness=-1)
                #         # cv2.imshow("emp_img",emp_img)
                #         # cv2.waitKey()
                #         new_img = (np.logical_or(annotated_img1,emp_img).astype(int))*255
                #         # print(np.count_nonzero(new_img == 1))
                #         # exit()
                #         # cv2.imshow("after",new_img)
                #         # cv2.waitKey()

                #         cv2.imwrite(os.path.join(annotation_path,str(j[0])+".png"),new_img)

                #         print("found contour")
                #         # if not os.path.exists('./region_growing_result/'+str(stack)):
                #         #     os.makedirs('./region_growing_result/'+str(stack))
                #         # cv2.drawContours(input_img,contours=contours,contourIdx=l,color=(255,0,0),thickness=1)
                #         # cv2.imwrite('./region_growing_result/'+str(stack)+'/'+str(j[0])+'.png',input_img)
                #         break
                # do.display_single_image(out_img, "Region Growing result")
                print("in false negative")
                flag3=True
        # print("dice coeff",dice_coeff)     

        # dice_coeff_arr.append(dice_coeff)
       
        fp = len(np.unique(labels))-1



        print("True Positives: ",tp)
        print("False Positives: ",fp)
        print("False Negatives:",fn)
        print("groundtruth: ",gt)
        # print("predicted clusters in the stack",cents)
        total_tp+=tp
        total_fp+=fp
        total_fn+=fn
        total_gt+=gt
        total_cents+=cents
            
    
    # dice_coeff_df = pd.DataFrame(dice_coeff_arr)
    # dice_coeff_df.to_csv('./treshold-'+str(tres)+' dice_coeff.csv')

    print("Total true positives: ",total_tp)
    print("Total false positives: ",total_fp)
    print("Total false negatives:", total_fn)
    print("Total Groundtruth: ",total_gt)
    print("Total predicted clusters ",total_cents)
    accuracy = round(total_tp/(total_tp+total_fn+total_fp),2)
    recall = round(total_tp/(total_tp+total_fn),2)
    precision = round(total_tp/(total_tp+total_fp),2)
    f1_score = round(2*precision*recall/(precision+recall),2)
    print("Accuracy: ",accuracy)
    print("Recall: ",recall)
    print("precision: ", precision)
    print("f1_score: ",f1_score)
    if(recall>max_recall):
        max_recall=recall
        best_treshold=tres
    
    

print("Best Treshold",best_treshold)
print("max recall",max_recall) 

# annotate(path)