import numpy as np
import cv2
import os

img = cv2.imread('./data/merged.png')
path='./10_4'

# filter to reduce noise

# flatten the image
flat_image = img.reshape((-1,3))
flat_image = np.float32(flat_image)
sliceNames = os.listdir(path)
images = []  
k=(7,7)
for sliceName in sliceNames:
    image = cv2.imread(os.path.join(path, sliceName), -1)
    # image = cv2.GaussianBlur(image,k,0)
    images.append(image[:,:,2])
images_array= np.array(images)  
data=[]
data1=[]
for z in range(10):
    for x in range(123):
        for y in range(123):
            data.append([z,x,y,images_array[z,x,y]])
            if(images_array[z,x,y] > 145):
                 data1.append([z,x,y,images_array[z,x,y]])
                # data1.append([z,x,y,images_array[z,x,y]])
            # else:
                # data.append([0,0,0,0])
data=np.array(data)
data = np.float32(data)

class Mean_Shift:

    def __init__(self, radius=2):
        self.radius = radius

    def fit(self, data,data1,max_iter):
        centroids = data1
        while True:
            new_centroids = list()
            for i in centroids:
                in_bandwidth = []
                centroid = i
                iter_count = 0
                for featureset in data:
                    featureset[1],featureset[2],featureset[3] = 0.75*featureset[1],0.75*featureset[2],0.25*featureset[3]
                    centroid[1],centroid[2],centroid[3] = 0.75*centroid[1],0.75*centroid[2],0.25*centroid[3]
                    if(iter_count >= max_iter):
                        # print("in if")
                        break

                    if np.linalg.norm(featureset-centroid) < self.radius:
                            # print("in second if")
                        in_bandwidth.append(featureset)
                    # elif(abs(featureset[0] - centroid[0]) == 0):
                    #     if np.linalg.norm(featureset-centroid) < self.radius:
                    #         in_bandwidth.append(featureset)
                    iter_count +=1
                if(len(in_bandwidth) > 0):
                    new_centroid = np.average(in_bandwidth,axis=0)
                    print(new_centroid)
                    new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))
            # print(uniques)
            prev_centroids = centroids

            centroids = []
            for i in range(len(uniques)):
                centroids.append(np.array(uniques[i]))    
            # print(prev_centroids)
            optimized = True
            # print(in_bandwidth)
            # print(centroids)
            for i in range(len(centroids)):
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
                
            if optimized:
                break
        # print(centroids)
        self.centroids = centroids

m = Mean_Shift()
m.fit(data,data1,800)
print(m.centroids)    