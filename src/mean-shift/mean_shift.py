import cv2
import os
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


img = cv2.imread('./data/merged.png')
path='./10_3'

# filter to reduce noise

# flatten the image
flat_image = img.reshape((-1,3))
flat_image = np.float32(flat_image)
sliceNames = os.listdir(path)
images = []  
k=(7,7)
for sliceName in sliceNames:
    image = cv2.imread(os.path.join(path, sliceName), -1)
    image = cv2.GaussianBlur(image,k,0)
    images.append(image[:,:,2])
images_array= np.array(images)  
data=[]
data1=[]
for z in range(10):
    for x in range(123):
        for y in range(123):
            data.append([z,x,y,images_array[z,x,y]])
            if(images_array[z,x,y] > 145):
                 data1.append([float(z/10),float(x/123),float(y/123),float(images_array[z,x,y]/255)])
                # data1.append([z,x,y,images_array[z,x,y]])
            # else:
                # data.append([0,0,0,0])
print("after for loop")
data=np.array(data)
flat_image=np.float32(data)
print("before meanshift")
# meanshift
def bandwidth(data,steps=50):
    all_data_centroid = abs(np.average(data, axis=0))
    all_data_norm = np.linalg.norm(all_data_centroid)
    return all_data_norm / steps
print(bandwidth(flat_image))
bandwidth = estimate_bandwidth(flat_image, quantile=0.06, n_samples=10000)
print(bandwidth)
ms = MeanShift(bin_seeding=True,max_iter=800)
ms.fit(flat_image)
labeled=ms.labels_
# print(labeled)
print("after meanshift")
# for i in labeled:
#     if(data1[i][3]  == -1):
#         data1[i][3] = -1
#     else:
#         data1[i][3] = labeled[i]
# # get number of segments
# segments = np.unique(labeled)
# print('Number of segments: ', segments.shape[0])
# print(type(labeled[0]))
# cv2.namedWindow("labeled_img", cv2.WINDOW_NORMAL)
print(labeled)
exit()
labeled_img = np.uint8(labeled.reshape((10,123,123)))
for z in range(10):
    for x in range(123):
        for y in range(123):
            if(images_array[z,x,y] < 145):
                # print("in if")
                labeled_img[z][x][y] = 25
                # data1.append([z,x,y,images_array[z,x,y]])
            # else:
                # data.append([0,0,0,0]))
segments = np.unique(labeled_img)
print("No of segments: ",segments.size)

# contour = np.array([[10,10], [190, 10], [190, 80], [10, 80]])
# print(labeled_img[0].shape)
# os.mkdir('./data/labeled_ms_10_3')

colors = [[255,0,0],[255,128,0],[255,255,0],[0,255,255],[153,204,255],[0,0,153],[127,0,255],[255,0,255],[255,0,127],[128,128,128],[51,102,0],[153,255,153],[0,0,0]]
for i in range(10):

    labeled_color = cv2.cvtColor(labeled_img[i],cv2.COLOR_GRAY2BGR)
    # print(labeled_color.shape)
    for k in range(123):
        for l in range(123):
            if  labeled_color[k][l][0] == 25:
                labeled_color[k][l][0] = colors[12][0]
                labeled_color[k][l][1] = colors[12][1]
                labeled_color[k][l][2] = colors[12][2]
            else:
                labeled_color[k][l][0], labeled_color[k][l][1], labeled_color[k][l][2] = colors[labeled_color[k][l][0]][0],colors[labeled_color[k][l][0]][1],colors[labeled_color[k][l][0]][2]


    # labeled_3 = cv2.merge([labeled_color[i],labeled_color[i],labeled_img[i]])
    
    cv2.imwrite('./data/labeled_ms_10_3/labeled_img'+str(i)+'.png',labeled_color)



# # get the average color of each segment
# total = np.zeros((segments.shape[0], 3), dtype=float)
# count = np.zeros(total.shape, dtype=float)
# for i, label in enumerate(labeled):
#     total[label] = total[label] + flat_image[i]
#     count[label] += 1
# avg = total/count
# avg = np.uint8(avg)

# res = avg[labeled]
# result = res.reshape((img.shape))

# # show the result
# cv2.imwrite('./data/result_0.9.png',result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


