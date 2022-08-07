import numpy as np
import cv2
import os
from copy import deepcopy
import shutil
import json
import tifffile
import importlib

from region_growing_code import region_growing

# Import constants module
loader = importlib.machinery.SourceFileLoader( 'mc_constants', r'C:\Users\KAVYA\Abhiram\microscopy\constants\constants.py')
spec = importlib.util.spec_from_loader( 'mc_constants', loader )
mc_constants = importlib.util.module_from_spec( spec )
loader.exec_module( mc_constants)

slide_name = mc_constants.Slide1_64x64_1
StackName = "StackName"
path=os.path.join(r'C:\Users\KAVYA\Abhiram\microscopy\16bitimages',slide_name)
annotation_path = os.path.join(path,'count_annotaion')

annotated_path = os.path.join(r'C:\Users\KAVYA\Abhiram\microscopy\16bitimages',slide_name+"_annotated")
if not os.path.exists(annotated_path):
    os.makedirs(annotated_path)
count_annotated_folder_path = os.path.join(path,'count_annotated_images')
if not os.path.exists(count_annotated_folder_path):
    os.makedirs(count_annotated_folder_path)

image_folder = 'NeoCx'
image_path = os.path.join(path,image_folder)

count_annotation_dir = 'count_annotaion'

image_dir = os.listdir(image_path)
count_annotations = os.listdir(path+r'\count_annotaion')[0] 
area_thresh = 4

annotation_json = {}
annotated_stacks_list = []
slide_json_arr = []
try:
    with open(os.path.join(annotated_path, "ManualMaskAnnotation.json"), 'r+') as fp:
       annotation_json = json.load(fp)
    slide_json_arr = annotation_json[image_folder]
    annotated_stacks_list = [ stack[StackName] for stack in slide_json_arr ]
except:
    print("Manual Maska annotation not found")

try:
    with open(os.path.join(path,count_annotation_dir, count_annotations), 'r') as ref_fp:
        annotation_dict = json.load(ref_fp)
except:
    print('Reference annotation json not available.')

annotation_dict = annotation_dict[image_folder]
centroid_dict = {}
for stackNo,stack in enumerate(annotation_dict):
    stackName = stack[StackName].split('_')
    centroid_dict[stackName[-2]+'_'+stackName[-1]] = []
    for cell in stack['cells']:
        centroid_arr = cell['centroid']
        centroid_arr.insert(0,cell['sliceNo'])
        centroid_dict[stackName[-2]+'_'+stackName[-1]].append(centroid_arr)

labels_arr  = []
region_growing_treshold = 30

total_count=0
no_of_stacks=0
area_count=0
for stack in image_dir:
    stackFullName = slide_name+'_'+image_folder+'_'+stack
    # if stack in ["1_30","1_31"]:
    #     continue
    if stackFullName not in annotated_stacks_list:  
        try:
            gt_centroids_arr = np.array(centroid_dict[stack])
        except KeyError:
            print("Count annotation not found")
            continue
        no_of_stacks+=1
        stack_count = 0
        stack_path = os.path.join(image_path,stack)
        annotated_stack_path = os.path.join(count_annotated_folder_path,image_folder,stack)
        # print(stack_path)
        # print(annotated_stack_path)
        if os.path.exists(annotated_stack_path):
            shutil.rmtree(annotated_stack_path)
        shutil.copytree(stack_path,annotated_stack_path)
        
        if not os.path.exists(os.path.join(annotated_path,image_folder,stack)):
            os.makedirs(os.path.join(annotated_path,image_folder,stack))

        images = []  
        images1 = [] 
        mask_labels = []
        centroids_arr = [] 
        mask_centroids_arr = []
        k=(7,7)
        print("stack ",stack)
        # print(sliceNames[0])
        # print(masks[stack])
        width=0
        height=0
        slices = os.listdir(os.path.join(path,image_folder,stack))
        slices = sorted(slices)
        # print(centroid_dict[stack])
        cells=[]
        stack_json_dict={}
        for sliceNo,sliceName in enumerate(slices): 
            img_path = os.path.join(path,image_folder,stack,sliceName)  
            image = cv2.imread(img_path, -1)#
            image = image/255
            image = np.uint8(image)
            # cv2.imshow("input image",image)
            # cv2.waitKey()
            # input_img = deepcopy(image)
            (width,height) = image.shape
            print("image ",str(sliceNo))
            maskName=sliceName[:-4]
            mask_image_path = os.path.join(annotated_path,image_folder,stack,maskName+'.png')
            # if not os.path.exists(mask_image_path):
            emp_img = np.zeros((width,height))
            cv2.imwrite(mask_image_path,emp_img)
            
            gt_centroids = []
            for i in gt_centroids_arr:
                if i[0] == sliceNo:
                    gt_centroids.append([i[1],i[2]])
            gt_centroids = np.array(gt_centroids)


        
            for i,j in enumerate(np.int64(gt_centroids)):
                # print("point coordinates",(j[0],j[1]))
                # try:
                # cv2.imshow("input-img2",image)
                # cv2.waitKey()
                # cv2.imshow("output image",out_img)
                # cv2.waitKey()
                # img_contours,_ = cv2.findContours(out_img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                thres = region_growing_treshold
                # while(1):
                #     out_img=region_growing.apply_region_growing(image,(j[0],j[1]),thres)
                #     (_, _, stats, _) = cv2.connectedComponentsWithStats(out_img,8, cv2.CV_32S)
                #     area = stats[1,cv2.CC_STAT_AREA]
                #     if(area >area_thresh):
                #         break
                #     if(thres <=0):
                #         print("Thresh less than zero")
                #     if(thres < 10):
                #         thres = thres - 3
                #     else:
                #         thres = thres - 5

                out_img=region_growing.apply_region_growing(image,(j[0],j[1]),thres)
                (_, _, stats, _) = cv2.connectedComponentsWithStats(out_img,8, cv2.CV_32S)
                area = stats[1,cv2.CC_STAT_AREA]

                if(area<=area_thresh):
                    out_img=region_growing.apply_region_growing(image,(j[0],j[1]),region_growing_treshold-20)
                    (_, _, stats, _) = cv2.connectedComponentsWithStats(out_img,8, cv2.CV_32S)
                    area = stats[1,cv2.CC_STAT_AREA]

                if(area <= area_thresh):
                    area_count+1;




                # print(area)
                # M=cv2.moments(img_contours[0])
                # area = M["m00"]
                # if(area <= 2.0):
                #     out_img=region_growing.apply_region_growing(image,(j[0],j[1]),region_growing_treshold-5)
                #     (_, _, stats, _) = cv2.connectedComponentsWithStats(out_img,8, cv2.CV_32S)
                #     area = stats[1,cv2.CC_STAT_AREA]
                # if(area <=area_thresh):
                #     area_count+=1
                
                #     out_img=region_growing.apply_region_growing(image,(j[0],j[1]),region_growing_treshold - 5)
            

                
                # except:
                    # print("unable to apply region growing")
                    # exit()
            
                contours,_=cv2.findContours(out_img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = contours[-2:]
                # (,,_,centroids)=cv2.connectedComponentsWithStats(out_img,8, cv2.CV_32S)
                stack_count+=1
                # cv2.imshow("out image",out_img)
                # cv2.waitKey()
                # print(mask_image_path)
                input_img = cv2.imread(mask_image_path)[...,2]
                # input_img = input_img/255
                # input_img = np.uint8(input_img)

                # print("out image shape",out_img.shape)
                # print("mask image shape",input_img.shape)
                # print("out image type",type(out_img[0,9]))
                # print("mask image type",type(input_img[0,9]))
                # print(input_img[0,0])
                # out_img=np.uint16(out_img)
                # cv2.imshow("input img",input_img)
                # cv2.waitKey()
                # cv2.imshow("output img",out_img)
                # cv2.waitKey()
                # print(out_img.shape)
                # print(input_img.shape)
                result_img = cv2.bitwise_or(out_img,input_img)
                # cv2.imshow("result image",result_img)
                # cv2.waitKey()
                cv2.imwrite(mask_image_path,result_img)
                M = cv2.moments(contours[0])
                try:
                    cX = int(M["m10"] / M["m00"])  # cOMPUTE CENTROID
                    cY = int(M["m01"] / M["m00"])
                except:
                    cX,cY=-1,-1
                    print("area of the particle is close to zero")
                cell={}
                cell['centroid'] = (cX, cY)
                cell['area'] = M['m00']
                cell['sliceNo'] = sliceNo
                cells.append(cell)
        # print("cells: ",cells)
       
        # print(stackFullName)
        stack_json_dict = {StackName:stackFullName,"cells":cells,"count":stack_count,"height":height,"width":width}
        slide_json_arr.append(stack_json_dict)
        # print(stack_json_dict)
        total_count +=stack_count
        annotation_json = {str(image_folder):slide_json_arr,"total_count":total_count,"total_stacks":no_of_stacks}
        with open(os.path.join(annotated_path, "ManualMaskAnnotation.json"), 'w') as fp:
            json.dump(annotation_json, fp, sort_keys=True, indent=2)
print("No of particles with area less than "+str(area_thresh)+" = "+str(area_count))
print("total stacks",no_of_stacks)
