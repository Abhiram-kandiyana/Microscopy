import PIL.Image
from PIL import ImageTk
import json

import tkinter as tk
import math
import numpy as np
from tkinter import filedialog
from tkinter.filedialog import askdirectory
from tkinter import messagebox
import os
import sys
import cv2
from multiprocessing import Queue
from PutDisector_OnImage import PutDisectorOnImage
import threading
queue = Queue()
import copy
from tqdm import tqdm
import copy
#from sort_slice_name_lst import *

global mouseX,mouseY
global videoSpeed
global rearFlag
global pauseFlag
global SFACTOR
global StackSize
StackSize=10
SFACTOR = 4  # scaling factor for viz
MARKER_SIZE = 2 #size of cross or circle marking clicks
MARKER_WIDTH = 2 # thickness of the marker lines
global CANVAS_IMAGE_X_SHIFT, VALID_CELL_FLAG, CANVAS_IMAGE_Y_SHIFT
rearFlag = False
pauseFlag = False
CANVAS_IMAGE_X_SHIFT=450 #x start of image on canvas
CANVAS_IMAGE_Y_SHIFT=150
VALID_CELL_FLAG = True #if started drawing within the disector box

'''
TBD 

1) percentage default value
2) percentage use input
3) Clicking outside the box
4) Verify mode and annotate mode

'''
#Modifications by Abhiram:Start
#Reading in the predicted centroids 
try:
    with open(r"C:\Users\KAVYA\Abhiram\microscopy\mean-shift\ms_predicted_centroids_40.json", 'r') as centroids_fp:
        centroids_dict = json.load(centroids_fp)
except Exception as e:
    print('Centroids json not available. Error: ',e)

#Modifications by Palak:Start
lastx, lasty = 0, 0
canvas_width = 3000
canvas_height = 1500
cell = {}
sliceNo=0
root = tk.Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
#root.attributes('-fullscreen', True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
# xscrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL)
# xscrollbar.grid(row=1, column=0, sticky=tk.E+tk.W)
# yscrollbar = tk.Scrollbar(root)
# yscrollbar.grid(row=0, column=1, sticky=tk.N+tk.S)

canvas = tk.Canvas(root,width = canvas_width, height = canvas_height)
# canvas = tk.Canvas(root,width = canvas_width, height = canvas_height,
#                    scrollregion=(0, 0, canvas_width, canvas_height),
#                    xscrollcommand=xscrollbar.set,
#                 yscrollcommand=yscrollbar.set)
canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
# xscrollbar.config(command=canvas.xview)
# yscrollbar.config(command=canvas.yview)
sectionNameStartsWith = ''
stackNameStartsWith = ''
#imgDirInStackDir = 'Stack'  # dir within Stack dir from where to pick stack images to display
imgDirInStackDir = ''
EDFDirInStackDir = 'EDF'
EDF_name = 'EDF.bmp'  # have to change based on dataset. NeuN dual stain - 'EDF_FULL_Vahadane_pp.png', NeuN single stain - 'EDF_FULL_rgb2gray.png'
MODE = 'new'  # In which mode to run this tool. 'update' - to update already existing annotations. 'new' - for new annotation.
# if MODE == 'new':
#     save_annotation_folder_name = 'count_annotaion'
# elif MODE == 'update':
#     save_annotation_folder_name = 'updated_count_annotaion'
# else:
#     print('Invalid mode.Please select "new" for new annotation and "update" to update already existing '
#                   'annotation based on prediction from pre-trained model.')
#     sys.exit()
save_annotation_folder_name = 'count_annotaion'

def xy(event):
    global lastx, lasty,i
    global cell
    global ANNOTATION_DICT, DISECTOR_PARAM, VALID_CELL_FLAG
    # check if clicked within disector box
    #print(event.x,event.y)
    if isPointOutSideDisectorBox(DISECTOR_PARAM['disectorwidth'], DISECTOR_PARAM['leftCornerX'], DISECTOR_PARAM['leftCornerY'], event.x, event.y):
        VALID_CELL_FLAG = False
        return
    lastx, lasty = event.x, event.y
    cell = {}
    canvas.focus_set()
    cell['sliceNo'] = i
    cell['centroid'] = (round((lastx-CANVAS_IMAGE_X_SHIFT)/SFACTOR), round((lasty-CANVAS_IMAGE_Y_SHIFT)/SFACTOR)) #centroid of contour (cX,cY)
    print(cell['centroid'])
def addLine(event):
    global lastx, lasty,i
    global cell
    global ANNOTATION_DICT, DISECTOR_PARAM, VALID_CELL_FLAG
    if not VALID_CELL_FLAG:
        return
    canvas.focus_set()
    canvas.create_line(lastx, lasty, event.x, event.y, width=1)
    cell['contour'].append((event.x-CANVAS_IMAGE_X_SHIFT, event.y-CANVAS_IMAGE_Y_SHIFT))
    lastx, lasty = event.x, event.y

def key(event):
    global i
    global RETURN
    global IMAGE_UPDATE, lastx, lasty, ANNOTATION_DICT, IS_VALID_STACK
    #canvas.focus_set()
    #pressedKey=repr(event.char)
    #print("Key pressed.")
    #print("pressed",event.char)
    args = event.keysym, event.keycode, event.char
    #print("Symbol: {}, Code: {}, Char: {}".format(*args))
    if event.keysym == "Up":
        upArrowPressed = True
        i = i - 1
        IMAGE_UPDATE = True
        print("Up arrow pressed")
    elif event.keysym == "Down":
        downArrowPressed = True
        i = i + 1
        IMAGE_UPDATE = True
        print("Down arrow pressed")
    elif event.char == "s":
        print("Skipping")
        IS_VALID_STACK = False
        RETURN=True
    elif event.keysym == 'Return':
        print('Enter pressed')
        answer = messagebox.askyesno('Enter Pressed','Enter pressed. Do you want to move to the next stack?')
        if answer:
            RETURN = True
    elif event.keysym == 'BackSpace':
        print('BackSpace pressed. Deleting last annotation.')
        delete_cell = ANNOTATION_DICT['cells'].pop()
        x = int(delete_cell['centroid'][0])*SFACTOR + CANVAS_IMAGE_X_SHIFT
        y = int(delete_cell['centroid'][1])*SFACTOR + CANVAS_IMAGE_Y_SHIFT
        print("original coordinates: ({}, {})".format(x,y))
        canvas.create_line(x - MARKER_SIZE, y + MARKER_SIZE, x + MARKER_SIZE, y - MARKER_SIZE, width=MARKER_WIDTH, fill="#000000")
        canvas.create_line(x - MARKER_SIZE, y - MARKER_SIZE, x + MARKER_SIZE, y + MARKER_SIZE, width=MARKER_WIDTH, fill="#000000")
        canvas.update()

    else:
        print("Invalid key pressed")

def save_cell(event):
    global cell, lastx, lasty
    global ANNOTATION_DICT, MASKS, i, VALID_CELL_FLAG
    #print(cell)
    #print(len(cell['contour']))
    if not VALID_CELL_FLAG:
        VALID_CELL_FLAG=True
        return
    #ANNOTATION_DICT['cells'].append(cell)
    #print(ANNOTATION_DICT)
    #cell.clear()
    #height, width = MASKS[i % 10].shape[:2]
    #temp = np.zeros((height , width ), np.uint8)
    #cv2.drawContours(temp,np.array([cell['contour']]),-1,(255,255,255),1)
    #cv2.imshow('mask',temp)
    x = lastx
    y = lasty
    canvas.create_line(x-MARKER_SIZE, y+MARKER_SIZE, x+MARKER_SIZE, y-MARKER_SIZE, width=MARKER_WIDTH, fill="#00ff00")
    canvas.create_line(x-MARKER_SIZE, y-MARKER_SIZE, x+MARKER_SIZE, y+MARKER_SIZE, width=MARKER_WIDTH, fill="#00ff00")
    print("original coordinates: ({}, {})".format(x,y))
    
    #save the cell
    ANNOTATION_DICT['cells'].append(cell)
#Modifications by Palak:End

def getScaleValue(v):
    #thread.start_new_thread(scaleValue(v), ())
    scaleValue(v)
def scaleValue(v):
    global videoSpeed
    videoSpeed = v
    videoSpeed = int(videoSpeed)*1000
def CreateThread():
    #thread.start_new_thread(iterateFunction, ())
    #p = Process(target=iterateFunction)
    #p.start()
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
   # print(x,y)
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


def getPoints(event,x,y,flags,param):
    global mouseX,mouseY
    global rearFlag
    global pauseFlag
    global listOfPoints
    global i
    foundMatch = False
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x,y)
        mouseX,mouseY = x,y
        #print(mouseX,mouseY)
        #rearFlag = False
        #pauseFlag = False
        if mouseX !=0 and mouseY !=0:
            #print('Now checking matches {} {}'.format(mouseX,mouseY))I
            listOfPoints,foundMatch = euclidean_distance(listOfPoints,mouseX,mouseY)

        if (isPointOutSideDisectorBox(param[0],param[1],param[2],mouseX,mouseY)):
            foundMatch = True
        #print(foundMatch)
        #if (PointsOutSideFlag):
            #continue
        if not (foundMatch):
            #print('print adding to the list')
            if (mouseX,mouseY) in listOfPoints:
                #print('Item already added to the list')
                pass
            elif mouseX != 0 and mouseY !=0:
                
                listOfPoints.append((math.floor(mouseX),math.floor(mouseY), i))

        #listOfPoints.append((math.floor(mouseX), math.floor(mouseY)))
def euclidean_distance(listOfloc,x,y):
    foundMatch = False
    for loc in listOfloc:
        #print((x - loc[0])**2 + (y - loc[1]**2))
        #print('x {} loc[0] {} distance {}'.format(x,loc[0],math.pow(x - loc[0],2)))
        #print('euclidean distance is {}'.format(math.sqrt(math.pow(x - loc[0],2) + math.pow(y - loc[1],2))))
        if (math.sqrt(math.pow(x - loc[0],2) + math.pow(y - loc[1],2))) < 8:
            foundMatch = True
            #print('Removing')
            listOfloc.remove((loc[0],loc[1],loc[2]))
    return listOfloc,foundMatch
def isPointOutSideDisectorBox(disectorwidth,LeftCornerX,LeftCornerY,x,y):
    OutsideFlag = False
    #print('leftCornerX {}, leftCornerY {}, rightCornerX {}, rightCornerY {}, mouseX {}, mouseY {}'.format(LeftCornerX,LeftCornerY,LeftCornerX+disectorwidth,LeftCornerY+disectorwidth,x,y))
    if not ((y >= LeftCornerX) and (y <= (LeftCornerX+disectorwidth)) and (x >=LeftCornerY) and (x <=LeftCornerY+disectorwidth)):
        OutsideFlag = True
    return OutsideFlag

def draw_cross(img,listOfLocation):
    global i
    for loc in listOfLocation:
        if 1:#i==loc[2]:
            x = loc['centroid'][0]
            y = loc['centroid'][1]
            cv2.line(img,(x-1,y+1),(x+1,y-1),(0,255,0),1)
            cv2.line(img,(x-1,y-1),(x+1,y+1),(0,255,0),1)
            #cv2.circle(img,(x,y), 1, (0,255,0), -1)
            #print('txt')
        else:
            pass
    return img

def saveImages(saveTo_folder, imgName_list, imgs_list, listOfLocation):
    for loc in listOfLocation:
        i = loc[2]
        x = loc[0]
        y = loc[1]
        cv2.line(imgs_list[i], (x - 5, y), (x + 5, y), (0, 255, 0), 2)
        cv2.line(imgs_list[i], (x, y - 5), (x, y + 5), (0, 255, 0), 2)
    for img, imgName in zip(imgs_list, imgName_list):
        cv2.imwrite(os.path.join(saveTo_folder,imgName),img)


def save_masks(saveTo_folder, imgName_list, imgs_list):
    for img, imgName in zip(imgs_list, imgName_list):
        cv2.imwrite(os.path.join(saveTo_folder, imgName), img)



def ReadSequenceOfImages(image_folder, NameOfStack,stack):
    global videoSpeed
    global mouseX,mouseY
    global pauseFlag
    global rearFlag
    global i
    global RETURN
    global IMAGE_UPDATE
    global ANNOTATION_DICT, cell, MASKS, DISECTOR_PARAM, REF_IMG_DIR, IS_VALID_STACK
    RETURN = False
    QUIT = False
    i = 0
    pauseFlag = False
    rearFlag = False


    # read ref annotation image and display for ref
    #print(os.path.join(REF_IMG_DIR,NameOfStack+'.png'))
    try:
        if MODE == 'new':
            refImg = cv2.imread(os.path.join(REF_IMG_DIR, NameOfStack+'.bmp'), -1)
        elif MODE == 'update':  # update already existing annotation based on pre-trained model predication
            # read and show pre-trained model predication images.
            refImg = cv2.imread(os.path.join(REF_IMG_DIR, NameOfStack + '_pred.png'), -1)
        else:
            print('Invalid mode. Please select "new" for new annotation and "update" to update already existing '
              'annotation based on prediction from pre-trained model.')
            sys.exit()
        refImg = cv2.resize(refImg, None, fx=SFACTOR, fy=SFACTOR, interpolation=cv2.INTER_CUBIC) # resize by scale factor
        blank = np.zeros_like(refImg)
        refImg = cv2.merge([blank, blank, refImg]) # make it color
        refImg, _, _, _ = PutDisectorOnImage(refImg, 75)
        cv2.imshow('Ref EDF Image', refImg)
    except:
        print('Reference annotation image not available.')





    #listOfImages = os.listdir(image_folder)
    listOfImages = os.listdir(os.path.join(image_folder, imgDirInStackDir))

    listOfImagesOnly = []
    for imgName in listOfImages:
        if not os.path.isdir(os.path.join(image_folder,imgDirInStackDir,imgName)):
            listOfImagesOnly.append(imgName)
    listOfImages = listOfImagesOnly
    # listOfImages = sorted(listOfImages, key=sort_slice_name_lst)
    listOfImages = sorted(listOfImages)
    # listOfImages.reverse()
    # if len(listOfImages) > 10:
    #     listOfImages = getMiddleTenFrames(listOfImages)
    # elif len(listOfImages) < 10:
    #     print("{} has {}".format(NameOfStack,len(listOfImages)))
    #     IS_VALID_STACK = False
    #     return IS_VALID_STACK
    
    
    ImageList = []
    
    
    # stack_size = len(listOfImages)
    # print(stack_size)
    # if stack_size % 2 == 0:
    #     center_number = math.ceil(stack_size /2) + 0.5
    # else:
    #     center_number = math.ceil(stack_size / 2)

# stack_size = 41
# center_number = 21

    # BoxSizeMicrons = 18
    # probe_diameter = 16
    # probe_radius = probe_diameter / 2

    # image_number = 1
    #ImageList_COPY = []
    imageToIndexMap = {"26":0,"27":1,"28":2,"29":3,"30":4,"31":5,"32":6,"33":7,"34":8,"35":9}
    stack_centroids = np.array(centroids_dict[stack])
    for img_no,img_name in enumerate(listOfImages):
        
        img = cv2.imread(os.path.join(image_folder,imgDirInStackDir,img_name),-1)
        # convert 16-bit gray images to 8-bit by linear mapping and converting to color image for viz
        img = img/255
        # print(img_name)
        # print(img.shape)
        img=np.reshape(img,(img.shape[0],img.shape[1],1))
        ch2 = np.zeros(img.shape)
        img  =np.concatenate((ch2,ch2,img),axis=2)
        img = np.uint8(img)

        for index,centroid in enumerate(stack_centroids):
            if(centroid[0] == img_no):
                centroid_x = centroid[1]
                centroid_y = centroid[2]
                img = cv2.circle(img, (centroid_x,centroid_y), radius=0, color=(0, 0, 255), thickness=-1)
                np.delete(stack_centroids,index)






        

        #cv2.imwrite(img_name, img)
        #param = [disectorwidth,leftCornerX,leftCornerY]
        [w,h,_] = img.shape
        img = cv2.resize(img,(h*SFACTOR,w*SFACTOR),interpolation=cv2.INTER_CUBIC)
        # cv2.imshow("img1",img)
        # cv2.waitKey()
        # # leftCornerX = leftCornerX *SFACTOR
        # leftCornerY = leftCornerY *SFACTOR
        # disectorwidth = disectorwidth *SFACTOR
        img,disectorwidth,leftCornerX,leftCornerY = PutDisectorOnImage(img,75)
        
        #convert bgr to rgb for PIL
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        ImageList.append(img)
    DISECTOR_PARAM={}
    DISECTOR_PARAM['disectorwidth'] = disectorwidth
    DISECTOR_PARAM['leftCornerX']= leftCornerX + CANVAS_IMAGE_Y_SHIFT
    DISECTOR_PARAM['leftCornerY']= leftCornerY + CANVAS_IMAGE_X_SHIFT
    #ImageList_COPY = copy.deepcopy(ImageList)


    #counter = 4
    global listOfPoints
    listOfPoints = []

    #while(True):
    #print(w,h)
    mouseX = 0
    mouseY = 0
    # Modifications by Palak: start
    IMAGE_UPDATE = True
    if MODE == 'new':
        ANNOTATION_DICT={'StackName': NameOfStack, 'width':w,'height':h,'cells': []}  # initialize annotation dict
        # for this stack if mode is 'new'. Append to already available if mode is 'update'
    #MASKS = np.zeros_like(ImageList)
    MASKS = np.zeros((10,w,h),np.uint8)
    #print(MASKS.shape)
    #cell={}
    img = ImageTk.PhotoImage(PIL.Image.fromarray(ImageList[i % 20]))
    imgID = canvas.create_image(CANVAS_IMAGE_X_SHIFT, CANVAS_IMAGE_Y_SHIFT, image=img, anchor=tk.NW)
    canvas.focus_set()
    canvas.bind("<Key>", key)
    canvas.tag_bind(imgID, "<Button-1>", xy)
    #canvas.tag_bind(imgID, "<B1-Motion>", addLine)
    canvas.tag_bind(imgID, "<ButtonRelease-1>", save_cell)

    # draw already available annotation on canvas
    for mark in ANNOTATION_DICT['cells']:
        x = mark['centroid'][0] + CANVAS_IMAGE_X_SHIFT
        y = mark['centroid'][1] + CANVAS_IMAGE_Y_SHIFT
        canvas.create_line(x - 5, y + 5, x + 5, y - 5, width=3, fill="#00ff00")
        canvas.create_line(x - 5, y - 5, x + 5, y + 5, width=3, fill="#00ff00")
        canvas.update()

    while not RETURN:
        #print("Inside while")
        if IMAGE_UPDATE:
            IMAGE_UPDATE = False
            #ImageList[i % 10] = copy.deepcopy(ImageList_COPY[i % 10])
            #imgWithAnnotation = draw_cross(ImageList[i % 10], listOfPoints)
            #print(canvas.bbox(tk.ALL))
            img = ImageTk.PhotoImage(PIL.Image.fromarray(ImageList[i % StackSize]))
            #imgID = canvas.create_image(600, 0, image=img, anchor=tk.NW)
            canvas.itemconfig(imgID,image=img)
            #canvas.create_image(img.width() + 20, 400, image=img, anchor=tk.W)
        if i == StackSize:
            i = 0
        if i < 0:
            i = StackSize-1
        canvas.update()
    print("After Return")
    #save_masks(saveTo_folder, listOfImages, MASKS)
    # cv2.destroyWindow('Ref Annotation')
    return IS_VALID_STACK


def iterateFunction():
    global mouseX, mouseY, ANNOTATION_DICT, IS_VALID_STACK
    path2Case = dirname
    #print(dirname)
    #print(dirname)
    if not os.path.exists(os.path.join(dirname, save_annotation_folder_name)):
        os.makedirs(os.path.join(dirname, save_annotation_folder_name))
    #text_file = open(os.path.join(dirname+"_annotated","ManualAnnotation.txt"), "a+")
    # This function is called only upon pressing 'start' button on GUI. Clear all, if any.
    ANNOTATION_DICT = {}
    annotation_case = {}
    annotation_case['total_count'] = 0
    annotation_case['all_stacks'] = 0
    annotation_case['valid_stacks'] = 0

    Sections = os.listdir(path2Case)
    for section in tqdm(Sections):
        if (section.startswith(sectionNameStartsWith) and os.path.isdir(os.path.join(path2Case,section))):
            print('Section:',section)

            Stacks = os.listdir(os.path.join(path2Case,section))
            for stack in Stacks:
                print("stack name",stack)
                if (stack.startswith(stackNameStartsWith) and os.path.isdir(os.path.join(path2Case,section,stack))):
                    #print('Now working')
                    NameOfStack = os.path.basename(os.path.normpath(dirname))+'_'+section+'_'+stack
                    NameOfStack = NameOfStack.replace(" ", '')
                    print('NameOfStack:',NameOfStack)
                    IS_VALID_STACK = True  # valid stack if stack has atleast 10 images and not 'skipped' (by
                    # pressing 's') during annotation
                    #checkforNameOfStackInFile = False
                    #OpenFile = open(os.path.join(dirname+"_annotated","ManualAnnotation.txt"),'r')
                    try:
                        with open(os.path.join(dirname, save_annotation_folder_name, "ManualAnnotation.json"), 'r+') as read_fp:
                            annotation_case = json.load(read_fp)
                            if section in annotation_case and any(ann.get('StackName',None)==NameOfStack
                                                                      for ann in annotation_case[section]):
                                print('Already annotated {}'.format(NameOfStack))
                                if MODE == 'new':
                                    continue
                                elif MODE == 'update':
                                    try:
                                        for ann in annotation_case[section]:
                                            if ann.get('StackName', None) == NameOfStack:
                                                ANNOTATION_DICT = copy.deepcopy(ann)  # already available annotation
                                                # dictionary in
                                                # this stack. Append new annotation to it.
                                                old_count = ANNOTATION_DICT['count']
                                                annotation_case[section].remove(ann)  # remove this stack from case
                                                # dictionary and add after updating
                                                # # draw already available annotation on canvas
                                                # for mark in ANNOTATION_DICT['cells']:
                                                #     x = mark['centroid'][0] + CANVAS_IMAGE_X_SHIFT
                                                #     y = mark['centroid'][1] + CANVAS_IMAGE_Y_SHIFT
                                                #     canvas.create_line(x - 5, y + 5, x + 5, y - 5, width=3, fill="#00ff00")
                                                #     canvas.create_line(x - 5, y - 5, x + 5, y + 5, width=3, fill="#00ff00")
                                                #     canvas.update()
                                                print('Annotation json for this stack is found.')
                                        print('Please continue updating the annotation.')
                                    except:
                                        print('No already available annotation found for this stack to update.')
                                        sys.exit()

                    except IOError:
                        print("Creating new annotation file.")

                    print(os.path.join(path2Case,section,stack))
                    if MODE == 'new':
                        annotation_case['all_stacks'] += 1  # keep track of total stacks in the case being annotated
                    IS_VALID_STACK = ReadSequenceOfImages(os.path.join(path2Case,section,stack),NameOfStack,stack)

                    if MODE == 'update':
                        #delete old annotation
                        annotation_case['total_count'] -= old_count  # subtract old count and add updated count if valid stack
                        os.remove(os.path.join(path2Case, save_annotation_folder_name, NameOfStack+'.png'))
                        annotation_case['valid_stacks'] -= 1  # remove from valid stack count and add later if valid stack.

                    # save annotation if valid stack
                    if IS_VALID_STACK:
                        annotation_case['valid_stacks'] += 1  # keep track of valid stacks in the case being annotated
                        # draw annotation on EDF image and save image
                        # EDF_img = cv2.imread(os.path.join(path2Case,section,stack, EDFDirInStackDir, EDF_name), cv2.IMREAD_COLOR)
                        # #EDF_img, _, _, _ = PutDisectorOnImage(EDF_img, 25)
                        EDF_img = cv2.imread(os.path.join(REF_IMG_DIR, NameOfStack + '.bmp'), -1)
                        #EDF_img = cv2.resize(refImg, None, fx=SFACTOR, fy=SFACTOR, interpolation=cv2.INTER_CUBIC)  # resize by scale factor
                        blank = np.zeros_like(EDF_img)
                        EDF_img = cv2.merge([blank, blank, EDF_img])  # make it color
                        EDF_img, _, _, _ = PutDisectorOnImage(EDF_img, 75)
                        EDF_img = draw_cross(EDF_img, ANNOTATION_DICT['cells'][:])
                        cv2.imwrite(os.path.join(path2Case, save_annotation_folder_name, NameOfStack+'.png'), EDF_img)

                        if not section in annotation_case:
                            annotation_case[section] = []
                        ANNOTATION_DICT['count'] = len(ANNOTATION_DICT['cells'])
                        annotation_case['total_count'] += ANNOTATION_DICT['count']  # add in both modes
                        annotation_case[section].append(ANNOTATION_DICT)
                        #print(annotation_case)
                    else:
                        # rename the invalid stack with prefix 'invalid_'
                        os.rename(os.path.join(path2Case,section,stack), os.path.join(path2Case,section,'invalid_'+stack))
                    # update json in case of valid and invalid stacks both because 'all_stacks' field is updated in
                    # both cases
                    with open(os.path.join(dirname, save_annotation_folder_name, "ManualAnnotation.json"), 'w') as fp:
                        json.dump(annotation_case, fp, sort_keys=True, indent=2)
                    #OpenFile.close()
    #with open(os.path.join(dirname+"_annotated","ManualAnnotation.json"), 'w') as fp:
        #json.dump(annotation_case, fp)
    print('Case completed. Please open a new case or close the tool.')
    messagebox.showinfo("Count Annotation Tool", "Case completed. Please open a new case or close the tool.")
    # flush stored annotation from completed case if any.
    annotation_case.clear()
    ANNOTATION_DICT.clear()
def start_foo_thread(event):
    global foo_thread
    foo_thread = threading.Thread(target=main)
    foo_thread.daemon = True
    #progressbar.start()
    foo_thread.start()
    #root.after(20, check_foo_thread)

def check_foo_thread():
    if foo_thread.is_alive():
        root.after(20, check_foo_thread)
    else:
        pass
def get_dirname():
    global dirname
    dirname = askdirectory(initialdir=os.getcwd(),title='Please select a directory')
    if len(dirname) > 0:
        #print ("You chose %s" % dirname)
        return dirname
    else: 
        dirname = os.getcwd()
        #print ("\nNo directory selected - initializing with %s \n" % os.getcwd())
        return dirname
def get_refImgDir():
    global REF_IMG_DIR
    REF_IMG_DIR = askdirectory(initialdir=os.getcwd(),title='Please select a directory')
    if len(REF_IMG_DIR) > 0:
        #print ("You chose %s" % dirname)
        return REF_IMG_DIR
    else:
        REF_IMG_DIR = os.getcwd()
        #print ("\nNo directory selected - initializing with %s \n" % os.getcwd())
        return REF_IMG_DIR

def printVideoSpeed():
    print("Tkinter is easy to use!")
    print('global varibale value is {}'.format(videoSpeed))

    
def openFolder():
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    return root.filename
def calltoCreateScale(frame):
    #thread.start_new_thread(createScale, ())
    createScale(frame)
def createScale(frame):
    fps = tk.Scale(frame,bg='white',bd=5,length=400,command=getScaleValue,orient="horizontal",from_=1,to=4,tickinterval=0.5)
    fps.grid(column=1,row=1,sticky=tk.W+tk.E)

def createGUI():
    #thread.start_new_thread(None, ())
    global mouseX, mouseY
    # root = tk.Tk()
    # root.title('Disector Video Annotation Tool')
    # root.geometry("800x160+200+10")
    # root.resizable(width=False,height=False)
    # frame = tk.Frame(root)

    #canvas.pack()
    # canvas.columnconfigure(0,weight=1)
    # canvas.columnconfigure(1,weight=1)
    # canvas.columnconfigure(2,weight=1)
    # canvas.columnconfigure(3,weight=1)
    # canvas.columnconfigure(4,weight=1)
    # canvas.rowconfigure(0,weight=1)
    # canvas.rowconfigure(1,weight=1)
    canvas.config(width=canvas_width,height=canvas_height)

    OpenFolderButton = tk.Button(canvas,
                    text="Open Case",
                    fg="blue",
                    bg="peachpuff",
                    command=get_dirname)
    Start_Button = tk.Button(canvas,
                    text="Start",
                    fg="blue",
                    bg="peachpuff",command=lambda:iterateFunction())
    Quit_button = tk.Button(canvas,
                    text="QUIT", 
                    fg="red",
                    bg="peachpuff",
                    command=lambda:sys.exit())
    refImgDir_button = tk.Button(canvas,
                                 text="Ref img dir",
                                 fg="blue",
                                 bg="peachpuff",
                                 command=get_refImgDir) #to get the dir of previously annotated EDFs for ref

    #calltoCreateScale(frame)

    OpenFolderButton.grid(column=0,row=0,padx=10,pady=10,sticky=tk.W+tk.E)
    Start_Button.grid(column=1,row=0,padx=10,pady=10,sticky=tk.W+tk.E)
    #Pause_Button.grid(column=2,row=0,padx=10,pady=10,sticky=tk.W+tk.E)
    #Resume_Button.grid(column=3,row=0,padx=10,pady=10,sticky=tk.W+tk.E)
    Quit_button.grid(column=2,row=0,padx=10,pady=10,sticky=tk.W+tk.E)
    refImgDir_button.grid(column=3, row=0, padx=10, pady=10, sticky=tk.W + tk.E)



    #testval = fps.get()
    root.mainloop()
def main():
    #start_foo_thread(None)
    #print('Now in main function')
    createGUI()

if __name__ == "__main__":
        main()