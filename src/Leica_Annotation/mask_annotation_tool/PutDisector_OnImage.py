

import cv2 
import numpy as np
import math

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
    I[range(1,x+disectorWidth),y] = [0,0,255]
    I[x+disectorWidth,range(y,y+disectorWidth)] = [0,0,255]
    I[range(x+disectorWidth,I.shape[0]),y+disectorWidth] = [0,0,255]
    return I,disectorWidth,x,y
