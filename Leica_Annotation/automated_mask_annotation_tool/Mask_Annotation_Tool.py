
# from pickle import TRUE
# from tkinter.tix import IMAGE
# from tokenize import Name
from decimal import DivisionByZero
import PIL.Image
from PIL import ImageTk
import json
# import asyncio
# import time
import tkinter as tk
import math
# from matplotlib.pyplot import contour
import numpy as np
from tkinter import filedialog
from tkinter.filedialog import askdirectory
from tkinter import messagebox
import os
import sys
import cv2
from multiprocessing import Queue
# from PutDisector_OnImage import PutDisectorOnImage
import threading
from copy import deepcopy
import time
from sort_slice_name_lst import *
queue = Queue()
import copy
from tqdm import tqdm

global mouseX, mouseY
global videoSpeed
global rearFlag
global pauseFlag
global CANVAS_IMAGE_X_SHIFT, VALID_CELL_FLAG, CANVAS_IMAGE_Y_SHIFT
global line_id_dict
global lenOfImages
global direction_flag
global total_correction_count
global total_time_taken
global overall_avg_dice_coeff
rearFlag = False
pauseFlag = False
line_id_dict={}
direction_flag = 0
CANVAS_IMAGE_X_SHIFT = 500  # x start of image on canvas
CANVAS_IMAGE_Y_SHIFT = 100  # -100
SCALE_FACTOR = 8 # Must be Interger - scale image by this factor for better visual
VALID_CELL_FLAG = True  # if started drawing within the disector box
total_correction_count=0
total_time_taken = 0
overall_avg_dice_coeff = 0


'''
TBD 

1) percentage default value
2) percentage use input
3) Clicking outside the box
4) Verify mode and annotate mode

'''

# Modifications by Palak:Start
lastx, lasty = 0, 0
canvas_width = 3000
canvas_height = 1500
cell = {}
sliceNo = 0
root = tk.Tk()
root.title("MASK ANNOTATION TOOL")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
# root.attributes('-fullscreen', True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root['background']='white'
# xscrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL)
# xscrollbar.grid(row=1, column=0, sticky=tk.E+tk.W)
# yscrollbar = tk.Scrollbar(root)
# yscrollbar.grid(row=0, column=1, sticky=tk.N+tk.S)

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
# canvas1 = tk.Canvas(root, width=2, height=50)
# canvas = tk.Canvas(root,width = canvas_width, height = canvas_height,
#                    scrollregion=(0, 0, canvas_width, canvas_height),
#                    xscrollcommand=xscrollbar.set,
#                 yscrollcommand=yscrollbar.set)
# canvas1.grid(row=0,sticky='ew')
canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))


# xscrollbar.config(command=canvas.xview)
# yscrollbar.config(command=canvas.yview)
sectionNameStartsWith = 'N'
stackNameStartsWith = ''  # 'S'[]
imgDirInStackDir = ''  # 'newStack'  # 'DABStack' #dir within Stack dir
MODE = 'update'  # In which mode to run this tool. 'update' - to update already existing annotations. 'new' - for new annotation.

stacks_newly_annotated = []

def xy(event):
    global lastx, lasty, i
    global cell
    global ANNOTATION_DICT, DISECTOR_PARAM, VALID_CELL_FLAG
    # check if clicked within disector box
    if (isPointOutSideDisectorBox(DISECTOR_PARAM['disectorwidth'], DISECTOR_PARAM['leftCornerX'],
                                  DISECTOR_PARAM['leftCornerY'], event.x, event.y)):
        VALID_CELL_FLAG = False
        return
    lastx, lasty = event.x, event.y
    cell = {}
    canvas.focus_set()
    cell['sliceNo'] = i
    cell['centroid'] = ()  # centroid of contour (cX,cY)
    cell['area'] = 0
    cell['contour'] = []
    cell['contour'].append((event.x - CANVAS_IMAGE_X_SHIFT, event.y - CANVAS_IMAGE_Y_SHIFT))


def addLine(event):
    global lastx, lasty, i
    global cell
    global ANNOTATION_DICT, DISECTOR_PARAM, VALID_CELL_FLAG
    if not VALID_CELL_FLAG:
        return
    canvas.focus_set()
    canvas.create_line((lastx, lasty, event.x, event.y), width=2, fill="#00ff00")
    cell['contour'].append((event.x - CANVAS_IMAGE_X_SHIFT,
                            event.y - CANVAS_IMAGE_Y_SHIFT))  # uncomment if want to save point of contours for each cell in annotation json
    lastx, lasty = event.x, event.y


def key(event):
    global i
    global RETURN
    global IMAGE_UPDATE, IS_VALID_STACK
    global direction_flag
    # canvas.focus_set()
    # pressedKey=repr(event.char)
    # print("Key pressed.")
    # print("pressed", event.char)
    args = event.keysym, event.keycode, event.char
    # print("Symbol: {}, Code: {}, Char: {}".format(*args))
    if event.keysym == "Up":
        i = i - 1
        direction_flag = 1
        IMAGE_UPDATE = True
        print("Up arrow pressed")
    elif event.keysym == "Down":
        i = i + 1
        direction_flag = 2
        IMAGE_UPDATE = True
        print("Down arrow pressed")
    elif event.char == "s" and MODE == 'new':
        print("Skipping")
        IS_VALID_STACK = False
        RETURN = True
    elif event.keysym == 'Return':
        direction_flag=0
        print('Enter pressed')
        if len(ANNOTATION_DICT['cells']) != REF_ANN_STACK_COUNT:
            answer = messagebox.askyesno('Enter Pressed', 'Ref ann has {} and you have drawn {} cells. Move to next '
                                                          'stack?'.format(REF_ANN_STACK_COUNT,
                                                                          len(ANNOTATION_DICT['cells'])))

                                                                    
            if answer:
                RETURN = True
        else:
            RETURN = True

    elif event.keysym == 'BackSpace':
        print('BackSpace pressed. Deleting last annotation.')
        delete_cell = ANNOTATION_DICT['cells'].pop()
        delete_cell['contour'][:] = delete_cell['contour'][:] * [SCALE_FACTOR, SCALE_FACTOR] + [CANVAS_IMAGE_X_SHIFT,
                                                                                                CANVAS_IMAGE_Y_SHIFT]  # convert image coordinates to canvas coordinates
        '''
        for idx in range(len(delete_cell['contour'])-1):
            x1 = delete_cell['contour'][idx][0][0] #+ CANVAS_IMAGE_X_SHIFT
            y1 = delete_cell['contour'][idx][0][1] #+ CANVAS_IMAGE_Y_SHIFT
            x2 = delete_cell['contour'][idx+1][0][0] #+ CANVAS_IMAGE_X_SHIFT
            y2 = delete_cell['contour'][idx+1][0][1] #+ CANVAS_IMAGE_Y_SHIFT
            canvas.create_line(x1, y1, x2, y2, width=2, fill="red")
        x1 = delete_cell['contour'][idx][0][0]  # + CANVAS_IMAGE_X_SHIFT
        y1 = delete_cell['contour'][idx][0][1]  # + CANVAS_IMAGE_Y_SHIFT
        x2 = delete_cell['contour'][0][0][0]  # + CANVAS_IMAGE_X_SHIFT
        y2 = delete_cell['contour'][0][0][1]  # + CANVAS_IMAGE_Y_SHIFT
        canvas.create_line(x1, y1, x2, y2, width=2, fill="red")        
        '''
        # draw_old_annotation(i%lenOfImages)

        
        # Make the last annotation contour red to indicate deleted
        canvas.create_line(tuple(delete_cell['contour'].ravel()), width=2, fill="red")
        x1 = delete_cell['contour'][-1][0][0]
        y1 = delete_cell['contour'][-1][0][1]
        x2 = delete_cell['contour'][0][0][0]
        y2 = delete_cell['contour'][0][0][1]
        canvas.create_line(x1, y1, x2, y2, width=2, fill="red")  # join last and first point
    else:
        print("Invalid key pressed")


def save_cell(event):
    global cell
    global ANNOTATION_DICT, MASKS, i, VALID_CELL_FLAG
    global lenOfImages
    global funcId1,funcId2,funcId3
    # print(cell)
    # print(len(cell['contour']))
    if not VALID_CELL_FLAG:
        VALID_CELL_FLAG = True
        return
    # ANNOTATION_DICT['cells'].append(cell)
    # print(ANNOTATION_DICT)
    # cell.clear()
    print(lenOfImages)
    height, width = MASKS[i % lenOfImages].shape[:2]
    temp = np.zeros((height * SCALE_FACTOR, width * SCALE_FACTOR),
                    np.uint8)  # bcz contour originally drawn on scaled size. Scale down after cleaning.
    cv2.drawContours(temp, np.array([cell['contour']]), -1, (255, 255, 255), 1)
    # cv2.imshow('mask',temp)
    # cv2.waitKey()
    image_path = listOfImages[(i%lenOfImages)][:-4]
    input_img=cv2.imread(os.path.join(save_folder,image_path+'.png'),-1)
    # cv2.imshow("input_img",input_img)
    # and_img = cv2.bitwise_or(input_img,temp)
    # cv2.imshow('and_img')
    # discard extra part outside the cell due to drawing mistake
    print(height)
    print(width)
    mask = np.zeros((height * SCALE_FACTOR + 2, width * SCALE_FACTOR + 2), np.uint8)
    M = cv2.moments(np.array([cell['contour']]))  # calculate moment for comtour
    # print(M)
    try:
        cX = int(M["m10"] / M["m00"])  # COMPUTE CENTROID
        cY = int(M["m01"] / M["m00"])
    except:
        print("Area is equal to zero")
        cX=0
        cY=0

    # print((cX, cY))
    # Clip the centroid to remain within image
    if cX < 0:
        cX = 0
    elif cX > width * SCALE_FACTOR:
        cX = width * SCALE_FACTOR
    if cY < 0:
        cY = 0
    elif cY > height * SCALE_FACTOR:
        cY = height * SCALE_FACTOR

    # cell['centroid'] = (cX, cY)  # save centroid to write in annotation
    # cell['area'] = M["m00"]  # save area to write in annotation
    # temp=copy.deepcopy(MASKS[i % 10])
    temp_inv = 255 - copy.deepcopy(temp)
    cv2.floodFill(temp, mask, (cX, cY), (255, 255, 255))
    # cv2.imshow('mask-filled', temp)
    # temp_inv=255-copy.deepcopy(MASKS[i % 10])
    # cv2.imshow('contour inv',temp_inv)
    mainBlob = cv2.bitwise_and(temp_inv, temp)
    # cv2.imshow('contour final', mainBlob)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(mainBlob, kernel, iterations=1)
    # cv2.imshow('difference',temp-mainBlob)
    # cv2.imshow('difference dilated', temp - dilated)
    # MASKS[% 10] = copy.deepcopy(dilated)
    # print(MASKS[i % 10].shape)
    # print(dilated.shape)
    # resize to original image size -> becomes gray, convert to binary
    _, dilated = cv2.threshold(
        cv2.resize(dilated, (int(dilated.shape[1] / SCALE_FACTOR), int(dilated.shape[0] / SCALE_FACTOR)),
                   interpolation=cv2.INTER_CUBIC), 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Cell Drawn', dilated)
    # cv2.imwrite("./cell_drawn.png",dilated)
    # (_, _, stats, _) = cv2.connectedComponentsWithStats(dilated,8, cv2.CV_32S)
    # new_area = stats[1,cv2.CC_STAT_AREA]
    new_img = dilated
    global corrected_masks
    global old_img
    # area_diff = abs(new_area - old_area)
    dice_coeff = 2*np.logical_and(old_img,new_img).sum()/(np.count_nonzero(old_img.astype(int)) + np.count_nonzero(new_img.astype(int)))

    corrected_masks.append({"sliceNo":i,"dice Coefficient":float(np.round(dice_coeff,2))})
    or_img = cv2.bitwise_or(dilated,input_img)
    cv2.imwrite(os.path.join(save_folder,image_path+'.png'),or_img)
    # cv2.imshow('and_img',and_img)

    cleaned_contour, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                         -2:]  # take last two returned values to make it independent of version of cv2
    if len(cleaned_contour) != 1:
        tk.messagebox.showwarning(title=None, message='More than one contour detected after cleaning. Will exit '
                                                           'the application.')
        exit()
    cell['contour'] = cleaned_contour[0]
    print('contour len:', len(cell['contour']))
    M = cv2.moments(cell['contour'])  # calculate moment for comtour
    # cX = int(M["m10"] / M["m00"])  # cOMPUTE CENTROID
    # cY = int(M["m01"] / M["m00"])
    # cell['centroid'] = (cX, cY)  # save centroid to write in annotation
    cell['area'] = M["m00"]  # save area to write in annotation
    # MASKS[i % 10] = cv2.bitwise_or(MASKS[i % 10], dilated) # save the cleaned blob in masks
    # now fill inner holes
    '''
    cv2.floodFill(temp, mask, (0, 0), (255, 255, 255))
    cv2.imshow("mask-filled-same",temp)
    cv2.imshow("mask-filled-same-inv",255-temp)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    temp=cv2.dilate(255-temp, kernel, iterations=1)
    final_blob=cv2.bitwise_or(mainBlob,temp)
    cv2.imshow("Final blob",final_blob)
    cv2.waitKey(500)
    '''
    # save the cell
    ANNOTATION_DICT['cells'].append(cell)
    global correction_count
    correction_count=correction_count+1
    canvas.unbind("<B1-Motion>",funcId1)
    canvas.unbind("<Button-1>",funcId2)
    canvas.unbind("<ButtonRelease-1>",funcId3)
# Modifications by Palak:End

#modifications by Abhiram
def save_cell1(event):
    global  markedx, markedy
    global i
    global ImageList
    global IMAGE_UPDATE
    global old_img
    markedx = (event.x - 500)/SCALE_FACTOR
    markedy = (event.y - 100)/SCALE_FACTOR
    canvas.create_line(event.x-2,  event.y+2,  event.x+2,  event.y-2, width=2, fill="#201DC3")
    canvas.create_line( event.x-2,  event.y-2,  event.x+2,  event.y+2, width=2, fill="#201DC3")
    print("original coordinates: ({}, {})".format(markedx,markedy))
    maskName = listOfImages[i % lenOfImages][:-4]
    pathToAnnotatedImage = os.path.join(save_folder,maskName+'.png')
    im = cv2.imread(pathToAnnotatedImage,-1)
    unique_count = np.unique(im)
    # im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # print("Now you can start drawing the boundary.")
    # messagebox.showinfo("NOTE","Now you can start drawing the boundary.")
    # print(im.shape)
    width = im.shape[0]
    height = im.shape[1]
    # im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # print(im[markedy,markedx])
    contours,hv2 = cv2.findContours(im,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # for j in line_id_dict[i]:
    #     for k in j:
    #         canvas.delete(k)
    in_if = 0
    for l,y in enumerate(contours):
        ppt_result = cv2.pointPolygonTest(y,(markedx,markedy),False)
        print("ppt_result",ppt_result)
        if(ppt_result >= 0):
            print("found contour")
            dup_img = np.uint8(np.ones((width,height)))
            in_if=1;
            # dup_img1 = np.zeros((width,height))
            # cv2.drawContours(dup_img1,[y],0,(255,255,255),-1)
            cv2.drawContours(dup_img,contours,l,(0,0,0),-1)
            dup_img1 = cv2.subtract(255,dup_img*255)
            old_img = dup_img1
            # cv2.imshow("dup img",dup_img)
            # cv2.waitKey()
            # cv2.imshow("dup img 1",dup_img1)
            # cv2.waitKey()
            # (_, _, stats, _) = cv2.connectedComponentsWithStats(dup_img1,8, cv2.CV_32S)
            # old_area = stats[1,cv2.CC_STAT_AREA]

            # cv2.imshow("corrected particle",dup_img)
            print(im.shape)
            # cv2.imshow("input img",im)
            # cv2.waitKey()
            # cv2.imshow("dup img",255*dup_img)
            # cv2.waitKey()
            and_img = cv2.bitwise_and(im,dup_img,mask=None)
            # and_img_unique = np.unique(and_img)
            
            and_img = (and_img.astype(int))*255
            # cv2.imshow("output image",and_img)
            # cv2.waitKey()
            # ImageList[i%lenOfImages] =   cv2.resize(and_img, (and_img.shape[1] * SCALE_FACTOR, and_img.shape[0] * SCALE_FACTOR), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(pathToAnnotatedImage,and_img)
            draw_old_annotation(i%lenOfImages)
            # i=i+1
            # i=i-1
            break
    # cells = ANNOTATION_DICT['cells']
    
    canvas.unbind("<ButtonRelease-1>",funcId4)
    # draw_old_annotation(i % lenOfImages)
    if in_if == 1:
     new_correct()
#modifications by Abhiram:End


def getScaleValue(v):
    # thread.start_new_thread(scaleValue(v), ())
    scaleValue(v)


def scaleValue(v):
    global videoSpeed
    videoSpeed = v
    videoSpeed = int(videoSpeed) * 1000


def CreateThread():
    # thread.start_new_thread(iterateFunction, ())
    # p = Process(target=iterateFunction)
    # p.start()
    iterateFunction()


def euclidean_distance(listOfloc, x, y):
    foundMatch = False
    for loc in listOfloc:
        # print((x - loc[0])**2 + (y - loc[1]**2))
        # print('x {} loc[0] {} distance {}'.format(x,loc[0],math.pow(x - loc[0],2)))
        # print('euclidean distance is {}'.format(math.sqrt(math.pow(x - loc[0],2) + math.pow(y - loc[1],2))))
        if (math.sqrt(math.pow(x - loc[0], 2) + math.pow(y - loc[1], 2))) < 8:
            foundMatch = True
            # print('Removing')
            listOfloc.remove((loc[0], loc[1]))
    return listOfloc, foundMatch


def isPointOutSideDisectorBox(disectorwidth, LeftCornerX, LeftCornerY, x, y):
    OutsideFlag = False
    # print('leftCornerX {}, leftCornerY {}, rightCornerX {}, rightCornerY {}, mouseX {}, mouseY {}'.format(LeftCornerX,LeftCornerY,LeftCornerX+disectorwidth,LeftCornerY+disectorwidth,x,y))
    if not ((y >= LeftCornerX) and (y <= (LeftCornerX + disectorwidth)) and (x >= LeftCornerY) and (
            x <= LeftCornerY + disectorwidth)):
        OutsideFlag = True
    return OutsideFlag


def getMiddleTenFrames(listOfImages):
    totalImages = len(listOfImages)
    totalImagesToRemove = totalImages - 10
    totalImageToRemoveFromLeft = totalImagesToRemove / 2.0
    totalImageToRemoveFromRight = totalImagesToRemove / 2.0
    totalImageToRemoveFromLeft = math.floor(totalImageToRemoveFromLeft)
    totalImageToRemoveFromRight = math.ceil(totalImageToRemoveFromRight)
    listOf10Images = listOfImages[totalImageToRemoveFromLeft:-totalImageToRemoveFromRight]
    return listOf10Images


def getPoints(event, x, y, flags, param):
    global mouseX, mouseY
    global rearFlag
    global pauseFlag
    global listOfPoints
    global i
    foundMatch = False
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        # print(mouseX,mouseY)
        # rearFlag = False
        # pauseFlag = False
        if mouseX != 0 and mouseY != 0:
            # print('Now checking matches {} {}'.format(mouseX,mouseY))I
            listOfPoints, foundMatch = euclidean_distance(listOfPoints, mouseX, mouseY)

        if (isPointOutSideDisectorBox(param[0], param[1], param[2], mouseX, mouseY)):
            foundMatch = True
        # print(foundMatch)
        # if (PointsOutSideFlag):
        # continue
        if not (foundMatch):
            # print('print adding to the list')
            if (mouseX, mouseY) in listOfPoints:
                # print('Item already added to the list')
                pass
            elif mouseX != 0 and mouseY != 0:
                listOfPoints.append((math.floor(mouseX), math.floor(mouseY), i))

        # listOfPoints.append((math.floor(mouseX), math.floor(mouseY)))


def euclidean_distance(listOfloc, x, y):
    foundMatch = False
    for loc in listOfloc:
        # print((x - loc[0])**2 + (y - loc[1]**2))
        # print('x {} loc[0] {} distance {}'.format(x,loc[0],math.pow(x - loc[0],2)))
        # print('euclidean distance is {}'.format(math.sqrt(math.pow(x - loc[0],2) + math.pow(y - loc[1],2))))
        if (math.sqrt(math.pow(x - loc[0], 2) + math.pow(y - loc[1], 2))) < 8:
            foundMatch = True
            # print('Removing')
            listOfloc.remove((loc[0], loc[1], loc[2]))
    return listOfloc, foundMatch


def isPointOutSideDisectorBox(disectorwidth, LeftCornerX, LeftCornerY, x, y):
    OutsideFlag = False
    # print('leftCornerX {}, leftCornerY {}, rightCornerX {}, rightCornerY {}, mouseX {}, mouseY {}'.format(LeftCornerX,LeftCornerY,LeftCornerX+disectorwidth,LeftCornerY+disectorwidth,x,y))
    if not ((y >= LeftCornerX) and (y <= (LeftCornerX + disectorwidth)) and (x >= LeftCornerY) and (
            x <= LeftCornerY + disectorwidth)):
        OutsideFlag = True
    return OutsideFlag


def draw_cross(img, listOfLocation):
    global i
    for loc in listOfLocation:
        if i == loc[2]:
            x = loc[0]
            y = loc[1]
            cv2.line(img, (x - 2, y), (x + 2, y), (0, 255, 0), 1)
            cv2.line(img, (x, y - 2), (x, y + 2), (0, 255, 0), 1)
        else:
            pass
    return img


def draw_ref_annotations(img_list, ref_annotation_stack_cells):
    global lenOfImages
    for cell in ref_annotation_stack_cells:
        x = cell['centroid'][0]*SCALE_FACTOR
        y = cell['centroid'][1]*SCALE_FACTOR
        sliceNo = cell['sliceNo']
        cv2.line(img_list[sliceNo], (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), 2)
        cv2.line(img_list[sliceNo], (x + 1, y - 1), (x - 1, y + 1), (0, 255, 0), 2)


def draw_ann_blobs_on_masks(MASKS, annotation_stack_cells):
    for cell in annotation_stack_cells:
        contour = cell['contour']
        sliceNo = cell['sliceNo']
        cv2.drawContours(MASKS[sliceNo], [contour], -1, 255, -1)
    return MASKS


# function to draw already available cells in 'update' mode.
def draw_old_annotation(i):
    # global save_folder
    # global line_id_array
    # global line_id_dict
    # global direction_flag
    global  lenOfImages
    # global  ImageList
    global img
    global ImageList
    global imgID
    global stack_name
    mask_name = listOfImages[i][:-4]
    print("Index of Image",i)
    pathToAnnotatedImage = os.path.join(save_folder,mask_name+'.png')
    annotated_img=cv2.imread(pathToAnnotatedImage,-1)
    annotated_img = cv2.resize(annotated_img, (annotated_img.shape[1] * SCALE_FACTOR, annotated_img.shape[0] * SCALE_FACTOR), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite('./resized.png',annotated_img)
    ret,annotated_img = cv2.threshold(annotated_img,127,255,cv2.THRESH_BINARY)
    contours,hv2 = cv2.findContours(annotated_img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("image 1",ImageList[i%lenOfImages])
    # cv2.waitKey()
    disp_img = deepcopy(ImageList[i%lenOfImages])
    # disp_img=np.ndarray(disp_img)
    # cv2.cvtColor(disp_img,cv2.COLOR_RGB2BGR)
    # cv2.imshow("image 2",reversed_img)
    # cv2.waitKey()
    cv2.drawContours(disp_img,contours,-1,(0,255,0),1)
    # cv2.imshow("image 3",disp_img)
    # cv2.waitKey()
    reversed_img=disp_img[:,:,::-1]
    if not os.path.exists(os.path.join('./contoured_images',stack_name)):
        os.makedirs(os.path.join('./contoured_images',stack_name))
    cv2.imwrite(os.path.join('./contoured_images',stack_name,str(i)+'.png'),reversed_img)
    # cv2.imshow("disp_img",disp_img)
    # cv2.waitKey()
    # img = ImageTk.PhotoImage(PIL.Image.fromarray(ImageList[i % lenOfImages]))
    img=ImageTk.PhotoImage(PIL.Image.fromarray(disp_img))
    # global imgID
   
    canvas.itemconfig(imgID, image=img)
    
   





    # j=i
    # if(direction_flag == 2):
    #     if(j == 0):
    #         j= lenOfImages - 1
    #     else:
    #         j=j-1

    #     for l in line_id_dict[j]:
    #         for n in l:
    #             canvas.delete(n)
    # elif direction_flag == 1:
    #     if( j == lenOfImages - 1):
    #         j=0
    #     else:
    #         j=j+1
    #     for l in line_id_dict[j]:
    #         for n in l:
    #             canvas.delete(n)
        


    

   
    # line_id_array = []
    # print(i)
    # print(pathToAnnotatedImage)
   
    # annotated_img = cv2.cvtColor(annotated_img,cv2.COLOR_BGR2GRAY)
    # if np.count_nonzero(annotated_img[annotated_img > 0]) > 0:
    #     annotated_img  = cv2.resize(annotated_img,(annotated_img.shape[0]*SCALE_FACTOR,annotated_img.shape[1]*SCALE_FACTOR),cv2.INTER_CUBIC)
    #     ann_contours,hv2 = cv2.findContours(annotated_img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        
    #     for j,x in enumerate(ann_contours):
    #         line_id_inner_arr = []
    #         for k,y in enumerate(x):
    #             if(k == len(x) -1):
    #                 break
    #             z=canvas.create_line((x[k][0][0]+500,x[k][0][1]+100,x[k+1][0][0]+500,x[k+1][0][1]+100), width=0, fill="#00ff00") 
    #             line_id_inner_arr.append(z)
    #         z=canvas.create_line((x[-1][0][0]+500,x[-1][0][1]+100,x[0][0][0]+500,x[0][0][1]+100), width=0, fill="#00ff00") 
    #         line_id_inner_arr.append(z)
    #         line_id_array.append(line_id_inner_arr)
    # line_id_dict[i] = line_id_array
    # print(line_id_dict)
    
               

    # for cell in old_annotation_stack_cells:
    #     r = 4  # randius of the circle
    #     x, y = cell['centroid'][0] + CANVAS_IMAGE_X_SHIFT, cell['centroid'][1] + CANVAS_IMAGE_Y_SHIFT  # center of the circle
    #     x0 = x - r
    #     y0 = y - r
    #     x1 = x + r
    #     y1 = y + r
    #     canvas.create_oval((x0, y0, x1, y1), width=2, outline='yellow')


def saveImages(saveTo_folder, imgName_list, imgs_list, listOfLocation):
    for loc in listOfLocation:
        i = loc[2]
        x = loc[0]
        y = loc[1]
        cv2.line(imgs_list[i], (x - 5, y), (x + 5, y), (0, 255, 0), 2)
        cv2.line(imgs_list[i], (x, y - 5), (x, y + 5), (0, 255, 0), 2)
    for img, imgName in zip(imgs_list, imgName_list):

        cv2.imwrite(os.path.join(saveTo_folder, imgName), img)


def save_masks(saveTo_folder, imgName_list, imgs_list):

    for img, imgName in zip(imgs_list, imgName_list):
        print(img)
        image_path = imgName[:-4]
        cv2.imwrite(os.path.join(saveTo_folder, image_path+'.png'), img)


def ReadSequenceOfImages(image_folder, saveTo_folder, NameOfStack, ref_annotation_stack_cells,old_annotation_stack_cells,section):
    global videoSpeed
    global mouseX, mouseY
    global pauseFlag
    global rearFlag
    global i
    global img
    global RETURN
    global IMAGE_UPDATE
    global ANNOTATION_DICT, cell, MASKS, DISECTOR_PARAM, REF_IMG_DIR, IS_VALID_STACK
    global lenOfImages
    global save_folder
    global ImageList
    global listOfImages
    global correction_count
    correction_count=0
    global corrected_masks
    corrected_masks = []
    RETURN = False
    QUIT = False
    i = 0
    pauseFlag = False
    rearFlag = False
    save_folder = saveTo_folder

    # read ref annotation image and display for ref
    # print(os.path.join(REF_IMG_DIR,NameOfStack+'.png'))
    try:
        # if reference annotation dir is provided, open reference annotation on EDF image
        # NameOfStack = NameOfStack.replace(" ", '')
        refImg = cv2.imread(os.path.join(REF_IMG_DIR, NameOfStack + '.png'))
        cv2.imshow('Ref Annotation', refImg)
    except:
        print('Reference/Count annotation EDF image is not available.')

        # IS_VALID_STACK = False
        # return IS_VALID_STACK

    # listOfImages = os.listdir(image_folder)
    print("Annotated Image not found")
    listOfImages = os.listdir(os.path.join(image_folder, imgDirInStackDir))
    lenOfImages = len(listOfImages)

    listOfImagesOnly = []
    for imgName in listOfImages:
        if not os.path.isdir(os.path.join(image_folder, imgDirInStackDir, imgName)):
            listOfImagesOnly.append(imgName)
    listOfImages = listOfImagesOnly
    listOfImages = sorted(listOfImages)  # , key=sort_slice_name_lst)
    if len(listOfImages) > 10:
        listOfImages = getMiddleTenFrames(listOfImages)
    elif len(listOfImages) < 10:
        print("{} has {} images. Skipping this stack.".format(NameOfStack, len(listOfImages)))
        IS_VALID_STACK = False
        return IS_VALID_STACK
    ImageList = []
    # ImageList_COPY = []
    for img_name in listOfImages:
        img = cv2.imread(os.path.join(image_folder, imgDirInStackDir, img_name), -1)
        # print(os.path.join(image_folder, imgDirInStackDir, img_name))
        img = img/256
        img = np.reshape(img,(img.shape[0],img.shape[1],1))
        blank_image = np.zeros(img.shape)
        img = np.concatenate((blank_image,blank_image,img),axis=2)
        # cv2.imshow("input img",img)
        # cv2.waitKey()
        img = np.uint8(img)
        # cv2.imshow("input 8 bit img",img)
        # cv2.waitKey()

        [w, h, _] = img.shape  # original image shape
        img = cv2.resize(img, (img.shape[1] * SCALE_FACTOR, img.shape[0] * SCALE_FACTOR), interpolation=cv2.INTER_CUBIC)
        # img, disectorwidth, leftCornerX, leftCornerY = PutDisectorOnImage(img, 25)
        disectorwidth = img.shape[0]
        leftCornerX = 0
        leftCornerY = 0
        # param = [disectorwidth,leftCornerX,leftCornerY]
        # convert bgr to rgb for PIL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ImageList.append(img)
    DISECTOR_PARAM = {}
    DISECTOR_PARAM['disectorwidth'] = disectorwidth
    DISECTOR_PARAM['leftCornerX'] = leftCornerX + CANVAS_IMAGE_Y_SHIFT
    DISECTOR_PARAM['leftCornerY'] = leftCornerY + CANVAS_IMAGE_X_SHIFT
    # ImageList_COPY = copy.deepcopy(ImageList)

    # draw reference anotation on corresponding best focus slices
    try:
        draw_ref_annotations(ImageList, ref_annotation_stack_cells)
    except:
        print('Could not draw reference annotation.')

    # counter = 4
    global listOfPoints
    listOfPoints = []

    # while(True):
    # print(w,h)
    mouseX = 0
    mouseY = 0

    IMAGE_UPDATE = True
    with open(r'./newly_annotated_stacks.txt', 'a') as fp:
        fp.write("%s\n" % NameOfStack)
    ANNOTATION_DICT = {'StackName': NameOfStack, 'width': w, 'height': h, 'cells': []}
    if MODE == 'new':  # initialize annotation dict and MASKS for this stack if mode is 'new'. Append to already available if mode is 'update'
        print("before masks",lenOfImages)
        MASKS = np.zeros((lenOfImages, w, h), np.uint8)
    elif MODE == 'update':
        # read already annotated masks
        MASKS = []
        for mask_no,mask_name in enumerate(listOfImages):
            img_name = mask_name[:-4]
            # print(os.path.join(saveTo_folder, mask_name))
            mask = cv2.imread(os.path.join(saveTo_folder, img_name+'.png'), -1)

            MASKS.append(mask)

    # print(MASKS.shape)
    # cell={} 
    img = ImageTk.PhotoImage(PIL.Image.fromarray(ImageList[i % 10]))
    global imgID
    imgID=canvas.create_image(CANVAS_IMAGE_X_SHIFT, CANVAS_IMAGE_Y_SHIFT, image=img, anchor=tk.NW)
    
    
    new_correct_button.grid(column=10, row=0, padx=0, pady=0,sticky=tk.E)
    old_correct_button.grid(column=15, row=0, padx=5, pady=0,sticky=tk.E)

    canvas.focus_set()
    canvas.bind("<Key>", key)
    # draw already available annotation

    # draw_old_annotation(i)
    while not RETURN:
        # print("Inside while")
        

        if IMAGE_UPDATE:
            IMAGE_UPDATE = False
            # ImageList[i % 10] = copy.deepcopy(ImageList_COPY[i % 10])
            # imgWithAnnotation = draw_cross(ImageList[i % 10], listOfPoints)
            # print(canvas.bbox(tk.ALL))
          
           
            # imgID = canvas.create_image(600, 0, image=img, anchor=tk.NW)
            # img = ImageTk.PhotoImage(PIL.Image.fromarray(ImageList[i % 10]))
            # canvas.create_image(img.width() + 20, 400, image=img, anchor=tk.W)
            draw_old_annotation(i%lenOfImages)
       
        
        if i == len(listOfImages):
            i = 0
        if i < 0:
            i = len(listOfImages) - 1
        # RETURN=True
        canvas.update() 
        
       

    print("Returned")
    global total_correction_count
    # global corrected_masks
    total_correction_count=total_correction_count + correction_count
    total_dice_coeff_stack=0
   
    for corrected_mask in corrected_masks:
        total_dice_coeff_stack+=corrected_mask["dice Coefficient"]
    try:
        avg_dice_coeff_stack = np.round(total_dice_coeff_stack/len(corrected_masks),2)
    except ZeroDivisionError:
        avg_dice_coeff_stack=0
        print("No corrections in the stack ",NameOfStack)

    visited["stacks"].append({"stackName":NameOfStack,"correctedMaskCount":correction_count,"correctedMasks":corrected_masks,"AvgDiceCoeff":float(avg_dice_coeff_stack)})
    # cv2.destroyWindow('Ref Annotation')

    if IS_VALID_STACK:
        if not os.path.exists(saveTo_folder):
            os.makedirs(saveTo_folder)  # create stack dir
        # draw the contours on masks and save the masks
        # draw_ann_blobs_on_masks(MASKS, ANNOTATION_DICT['cells'])
        # save_masks(saveTo_folder, listOfImages, MASKS)

    return IS_VALID_STACK
    # contours=[]
    # for c in ANNOTATION_DICT['cells']:
    #    contours.append(c['contour'])
    # print(contours)
    # mask=np.zeros_like(imgWithAnnotation)
    # cv2.drawContours(mask,np.array(contours),-1,(255,255,255),-1)
    # cv2.imshow('mask',mask)
    # cv2.waitKey()
    # print(ANNOTATION_DICT)
    # print(len(ANNOTATION_DICT['cells'][2]['contour']))


def iterateFunction():
    global mouseX, mouseY, ANNOTATION_DICT, REF_IMG_DIR, IS_VALID_STACK,i
    global REF_ANN_STACK_COUNT  # number of cells in the reference annotation if available
    global PathToAnnotatedImage
    global path2Case
    global stack_name
    global visited
    global correction_count
    global total_correction_count
    global overall_avg_dice_coeff
    global total_time_taken
    global corrected_masks
    path2Case = dirname
    visited={"stacks":[],"totalCorrectedMasks":0}
    correction_count = 0
    corrected_masks= []
    # print(dirname)
    # print(dirname)
    if not os.path.exists(dirname + "_annotated"):
        os.makedirs(dirname + "_annotated")

    # read reference annotation json if available
    try:
        with open(os.path.join(REF_IMG_DIR, 'ManualAnnotation.json'), 'r') as ref_fp:
            ref_annotation_dict = json.load(ref_fp)
    except:
        print('Reference annotation json not available.')

    try:
        with open(os.path.join(dirname + "_annotated", "visited.json"), 'r') as read_fp:
            visited = json.load(read_fp)
        total_correction_count = visited["totalCorrectedMasks"]
        total_time_taken = visited["totalTimeTakenInMinutes"]
    except:
        print("create a new visited json")
        total_correction_count=0
        total_time_taken = 0
    
    overall_avg_dice_coeff = 0
    annotation_case = {'total_stacks': 0, 'total_count': 0}
    Sections = os.listdir(path2Case)
    for section in tqdm(Sections):
        if (section.startswith(sectionNameStartsWith) and os.path.isdir(os.path.join(path2Case, section))):
            print('Section:', section)

            Stacks = os.listdir(os.path.join(path2Case, section))
            for stack in Stacks:
                start_time = time.time()
                stack_name = stack
                if (stack.startswith(stackNameStartsWith) and os.path.isdir(os.path.join(path2Case, section, stack))):
                    # print('Now working')
                    NameOfStack = os.path.basename(os.path.normpath(dirname)) + '_' + section + '_' + stack
                    NameOfStack = NameOfStack.replace(" ", '')
                    IS_VALID_STACK = True  # assume that this stack is valid. A stack is invalid when less than 10 slices or skipped by pressing 's'
                    # check if this stack is already annotated
                   

                    try:
                        with open(os.path.join(dirname + "_annotated", "ManualMaskAnnotation.json"), 'r+') as read_fp:
                            annotation_case = json.load(read_fp)
                            if section in annotation_case and any(ann.get('StackName', None) == NameOfStack 
                                                for ann in annotation_case[section]):
                                print('Already annotated {}'.format(NameOfStack))
                                if MODE == 'new':
                                    continue  # skip this stack if already annotated and the mode is new
                                elif MODE == 'update':
                                    try:
                                        for ann in annotation_case[section]:
                                            if ann.get('StackName', None) == NameOfStack:
                                                ANNOTATION_DICT_old = copy.deepcopy(ann)  # already available annotation
                                                # dictionary in
                                                # this stack. Append new annotation to it.
                                                old_count = ANNOTATION_DICT_old['count']
                                                old_annotation_stack_cells = ANNOTATION_DICT_old['cells']
                                                annotation_case[section].remove(ann)  # remove this stack from case
                                                # dictionary and add after updating
                                                # # draw already available annotation on canvas
                                                # for mark in ANNOTATION_DICT['cells']:
                                                #     x = mark['centroid'][0] + CANVAS_IMAGE_X_SHIFT
                                                #     y = mark['centroid'][1] + CANVAS_IMAGE_Y_SHIFT
                                                #     canvas.create_line(x - 5, y + 5, x + 5, y - 5, width=3, fill="#00ff00")
                                                #     canvas.create_line(x - 5, y - 5, x + 5, y + 5, width=3, fill="#00ff00")
                                                #     canvas.update()
                                                print('Mask Annotation json for this stack is found.')
                                        print('Please continue updating the annotation.')
                                    except:
                                        print('No already available annotation found for this stack to update.')
                                        sys.exit()
                           
                    except IOError:
                        print("Creating new annotation file.")

                    # check if reference annotation for this stack is available
                    stack_is_annotated = 0
                    try:
                        for ann in ref_annotation_dict[section]:
                            # print("ann.get('StackName', None)",ann.get('StackName', None))
                            # print("stack name",NameOfStack)
                            if ann.get('StackName', None) == NameOfStack:
                                stack_is_annotated=1
                                ref_annotation_stack_cells = ann.get('cells', None)
                                REF_ANN_STACK_COUNT = len(
                                    ref_annotation_stack_cells)  # number of of cells in reference annotation
                                print('Reference annotation json for this stack is found.')
                            
                    except:
                        ref_annotation_stack_cells = None
                        REF_ANN_STACK_COUNT = 0
                        print('Reference annotation json for this stack is not available.')

                    if(stack_is_annotated == 0):
                        print("stack is not annotated",NameOfStack)
                        continue

                    saveTo_folder = os.path.join(path2Case + "_annotated", section, stack)
                    # if not os.path.exists(saveTo_folder):
                    #    os.makedirs(saveTo_folder)
                    print(os.path.join(path2Case, section, stack))
                    if MODE == 'update':
                        # continue to next stack if same number of cells in available mask annotation and ref annotaion
                        stack_found=False
                        try:
                            for visited_stack in visited["stacks"]:
                                print("visited stacks:",visited_stack["stackName"])
                                if NameOfStack == visited_stack["stackName"]:
                                    stack_found=True
                                    break
                            if(stack_found):
                                print("stack {} is already verified",NameOfStack)
                                continue
                            else:
                                print("stack needs to be updated")
                        except:
                            print("error while checking visited json")
                        # if NameOfStack in  visited["stacks"]:
                        #     print('Already visited.Update not needed.')
                        #     continue
                        # else:
                        #     # messagebox.showinfo(NameOfStack,
                        #     #                     "There is a difference of {} cells between reference annotation and "
                        #     #                     "available mask annotation. Continue to update.".format(
                        #     #                         REF_ANN_STACK_COUNT - old_count))
                        #     # messagebox.showinfo("")
                        #     print(NameOfStack," needs to be updated")
                        print("UPDATE MODE")
                    else:
                        old_annotation_stack_cells = []  # create empty list for old annotation. Because it is paased as an arg to readsequenceofimages to draw.
                    
                    IS_VALID_STACK = ReadSequenceOfImages(os.path.join(path2Case, section, stack), saveTo_folder,
                                                          NameOfStack,
                                                          ref_annotation_stack_cells,old_annotation_stack_cells,section)
                    if not section in annotation_case:
                        annotation_case[section] = []

                    if IS_VALID_STACK:
                    #     # Remove contour points from ANNOTATION_DICT before saving. No need to save the big list for each cell
                        ANNOTATION_DICT['cells'] = [{key: value for key, value in dict.items() if key != 'contour'} for
                                                    dict in ANNOTATION_DICT['cells']]
                        if MODE == 'update':  # append already available annotation to new update
                            ANNOTATION_DICT['cells'] += ANNOTATION_DICT_old['cells']
                            annotation_case['total_count'] -= ANNOTATION_DICT_old[
                                'count']  # delete old count and add updated count later

                    #     # Count of cells in this stack
                        ANNOTATION_DICT['count'] = len(ANNOTATION_DICT['cells'])
                    #     # Add this stack annotation to case annotation dict
                        annotation_case[section].append(ANNOTATION_DICT)
                        annotation_case['total_count'] += ANNOTATION_DICT['count']
                        if MODE == 'new':
                            annotation_case['total_stacks'] += 1
                            print("old annotations")
                    else:
                        # rename the invalid stack with prefix 'invalid_'
                        os.rename(os.path.join(path2Case, section, stack),
                                  os.path.join(path2Case, section, 'invalid_' + stack))
                    # print(annotation_case)
                    visited["totalCorrectedMasks"] = total_correction_count
                    total_time_taken += (time.time() - start_time)/60
                    visited["totalTimeTakenInMinutes"] = float(np.round(total_time_taken,3))
                    avg_dice_coeff_count = 0
                    for stack1 in visited["stacks"]:
                        if stack1["correctedMaskCount"] > 0:
                            avg_dice_coeff_count+=1
                            overall_avg_dice_coeff +=stack1["AvgDiceCoeff"]

                    if avg_dice_coeff_count > 0:
                        overall_avg_dice_coeff = np.round(float(overall_avg_dice_coeff/avg_dice_coeff_count),3)
                    else:
                        overall_avg_dice_coeff = 0
                    visited["AvgDiceCoefficient"] = overall_avg_dice_coeff
                    with open(os.path.join(dirname + "_annotated", "ManualMaskAnnotation.json"), 'w') as fp:
                        json.dump(annotation_case, fp, sort_keys=True, indent=2)
                    with open(os.path.join(dirname + "_annotated", "visited.json"), 'w') as fp:
                        json.dump(visited, fp, sort_keys=True, indent=2)

    print('Case completed. Please open a new case or close the tool.')
    messagebox.showinfo("Mask Annotation Tool", "Case completed. Please open a new case or close the tool.")
    # flush stored annotation from completed case if any.
    annotation_case.clear()
    ANNOTATION_DICT.clear()


def start_foo_thread(event):
    global foo_thread
    foo_thread = threading.Thread(target=main)
    foo_thread.daemon = True
    # progressbar.start()
    foo_thread.start()
    # root.after(20, check_foo_thread)


def check_foo_thread():
    if foo_thread.is_alive():
        root.after(20, check_foo_thread)
    else:
        pass


def get_dirname():
    global dirname
    dirname = askdirectory(initialdir=os.getcwd(), title='Please select a directory')
    if len(dirname) > 0:
        # print ("You chose %s" % dirname)
        return dirname
    else:
        dirname = os.getcwd()
        # print ("\nNo directory selected - initializing with %s \n" % os.getcwd())
        return dirname


def get_refImgDir():
    global REF_IMG_DIR
    REF_IMG_DIR = askdirectory(initialdir=os.getcwd(), title='Please select a directory')
    if len(REF_IMG_DIR) > 0:
        # print ("You chose %s" % dirname)
        return REF_IMG_DIR
    else:
        REF_IMG_DIR = os.getcwd()
        # print ("\nNo directory selected - initializing with %s \n" % os.getcwd())
        return REF_IMG_DIR


def printVideoSpeed():
    print("Tkinter is easy to use!")
    print('global varibale value is {}'.format(videoSpeed))


def openFolder():
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    return root.filename


def calltoCreateScale(frame):
    # thread.start_new_thread(createScale, ())
    createScale(frame)


def createScale(frame):
    fps = tk.Scale(frame, bg='white', bd=5, length=400, command=getScaleValue, orient="horizontal", from_=1, to=4,
                   tickinterval=0.5)
    fps.grid(column=1, row=1, sticky=tk.W + tk.E)


def createGUI():
    # thread.start_new_thread(None, ())
    global mouseX, mouseY
    global new_correct_button
    global old_correct_button
    # root.title('Disector Video Annotation Tool')
    # root.geometry("800x160+200+10")
    # root.resizable(width=False,height=False)
    # frame = tk.Frame(root)

    # canvas.pack()
    # canvas.columnconfigure(0,weight=1)
    # canvas.columnconfigure(1,weight=1)
    # canvas.columnconfigure(2,weight=1)
    # canvas.columnconfigure(3,weight=1)
    # canvas.columnconfigure(4,weight=1)
    # canvas.rowconfigure(0,weight=1)
    # canvas.rowconfigure(1,weight=1)
  
    canvas.config(width=canvas_width, height=canvas_height)

    OpenFolderButton = tk.Button(canvas,
                                 text="Open Case",
                                 fg="blue",
                                 highlightbackground='#3E4149',
                                 command=get_dirname)
    Start_Button = tk.Button(canvas,
                             text="Start",
                             fg="blue",
                             highlightbackground='#3E4149', command=lambda: iterateFunction())
    Quit_button = tk.Button(canvas,
                            text="QUIT",
                            fg="red",
                            highlightbackground='#3E4149',
                            command=lambda: sys.exit())
    refImgDir_button = tk.Button(canvas,
                                 text="Ref img dir",
                                 fg="blue",
                                 highlightbackground='#3E4149',
                              command=get_refImgDir)  # to get the dir of previously annotated EDFs for ref
    new_correct_button = tk.Button(canvas,
                                text="DRAW NEW MASK",
                                fg="red",
                                highlightbackground='#3E4149',
                                command=new_correct
                                )
    old_correct_button = tk.Button(canvas,
                                text="CORRECT OLD MASK",
                                fg="red",
                                highlightbackground='#3E4149',
                                command=old_correct
                                )

    # calltoCreateScale(frame)
    OpenFolderButton.grid(column=0, row=0, padx=10, pady=10, sticky=tk.W + tk.E)
    Start_Button.grid(column=1, row=0, padx=10, pady=10, sticky=tk.W + tk.E)
    # Pause_Button.grid(column=2,row=0,padx=10,pady=10,sticky=tk.W+tk.E)
    # Resume_Button.grid(column=3,row=0,padx=10,pady=10,sticky=tk.W+tk.E)
    Quit_button.grid(column=2, row=0, padx=10, pady=10, sticky=tk.W + tk.E)
    refImgDir_button.grid(column=3, row=0, padx=10, pady=10, sticky=tk.W + tk.E)

    # testval = fps.get()
    root.mainloop()

def new_correct():
    global funcId1,funcId2,funcId3
    # messagebox.showinfo("NOTE","You can start drawing the boundary on the particle.")
    funcId1=canvas.tag_bind(imgID, "<B1-Motion>", addLine)
    funcId2=canvas.tag_bind(imgID, "<Button-1>", xy)
    funcId3=canvas.tag_bind(imgID, "<ButtonRelease-1>", save_cell)

def old_correct():
    global funcId4
    global  markedx, markedy
    global i
    global lenOfImages
    global line_id_dict
    funcId4= canvas.tag_bind(imgID, "<ButtonRelease-1>", save_cell1)

   



    
def main():
    # start_foo_thread(None)
    # print('Now in main function')
    createGUI()


if __name__ == "__main__":
    main()
 