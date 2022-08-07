import cv2
import os
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth,KMeans
from sklearn.metrics import davies_bouldin_score,silhouette_score


path=r'C:\Users\KAVYA\Abhiram\shared_with_Abhiram\Data\images2\Neo_cx\10_3'

# filter to reduce noise

# flatten the image
# flat_image = img.reshape((-1,3))
# flat_image = np.float32(flat_image)
sliceNames = os.listdir(path)
images = []
k=(3,3)
for sliceName in sliceNames:
    image = cv2.imread(os.path.join(path, sliceName), -1)
    image = cv2.GaussianBlur(image,k,0)
    images.append(image[:,:,2])
images_array= np.array(images)  
data=[]
for z in range(10):
    for x in range(123):
        for y in range(123):
            if(images_array[z,x,y] > 135):
                data.append([z,x,y,images_array[z,x,y]])
            else:
                data.append([-1,-1,-1,-1])

data=np.array(data)
flat_image=data
flat_image = np.float32(flat_image)
# os.mkdir('./data/flat_image')

for i in range(10):
    cv2.imwrite('./data/flat_images/flat_img'+str(i)+'.png',np.uint8(flat_image)[i]*10)

#knn_clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts=20
K=10
min_dbs = 100
max_dbs = -2
best_k = 0
# print(flat_image.shape)
for k in range(2,K):
    ret,label,center=cv2.kmeans(flat_image,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    dbs = davies_bouldin_score(flat_image,label)
    print(dbs)
    print(k)
    if(dbs < min_dbs):
        min_dbs = dbs
        best_k = k
print(best_k)
print(min_dbs)

# get number of segments
segments = np.unique(label)
print('Number of segments: ', segments.shape[0])
print(type(label[0]))
# cv2.namedWindow("labeled_img", cv2.WINDOW_NORMAL)
# os.mkdir('./data/labeled_knn')
for i in range(10):
    
    cv2.imwrite('./data/labeled_knn/labeled_img'+str(i)+'.png',np.uint8(label.reshape((10,123,123)))[i]*25)



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


