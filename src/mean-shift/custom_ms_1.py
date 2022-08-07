import numpy as np
import cv2
import os

# img = cv2.imread('./data/merged.png')
path=r'C:\Users\KAVYA\Abhiram\microscopy\16bitimages\slide1-64x64_1\Neo_cx'

# filter to reduce noise

# flatten the image
# flat_image = img.reshape((-1,3))
# flat_image = np.float32(flat_image)
sliceNames = os.listdir(path)
images = []  
images1 = []
centroids_arr = []
k=(7,7)
for sliceNo,sliceName in enumerate(sliceNames):
    image = cv2.imread(os.path.join(path, sliceName), -1)#
    ret,image1 = cv2.threshold(image[:,:,2],145,255,cv2.THRESH_TOZERO)
    output = cv2.connectedComponentsWithStats(image1,8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output # discard the first centroid 
    # cv2.imshow("image1",image)
    # cv2.waitKey()
    # image = cv2.GaussianBlur(image,k,0)
    images.append(image[:,:,2])
    images1.append(image[:,:,2])
    for i in centroids[1:]:
        centroids_arr.append(np.append(i,int(sliceNo)))
images_array= np.array(images) 
images_array1 = np.array(images1)
centroids_arr = np.array(centroids_arr)
# print(images_array[0,0,1])
# print(centroids_arr)
data=[]
data1=[]
for z in range(10):
    for x in range(123):
        for y in range(123):
            # data.append([z,x,y,images_array[z,x,y]])  
            if(images_array[z,x,y] > 145):
                data.append([float(z/10),float(x/123),float(y/123),float(images_array[z,x,y]/255)])
            # if(centroids_arr[z][x][0] == data[z   b n])
for i in centroids_arr:
    data1.append([float(i[2]/10),float(i[1]/123),float(i[0]/123),float(images_array1[int(i[2]),int(i[1]),int(i[0])]/255)])
data,data1=np.array(data),np.array(data1)
data,data1 = np.float32(data),np.float32(data1)
copy=[]
for i in data:
    copy.append(np.append(i,-1))
def  weightedEuclideanDist(a,b,w):
    q = a-b
    return np.sqrt((w*q*q).sum())
class Mean_Shift:
    def __init__(self, radius=0.098,max_iter=800):
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
# print("centroids",centroids)
classifications = clf.classifications
# print("classifications:",classifications)
labels = np.array([100]*151290)
labels = np.reshape(labels,(10,123,123))
for i in classifications:
    for j in classifications[i]:
        labels[j[0]*10,j[1]*123,j[2]*123]=i
# print("copy:")
# print(copy)

# print(labels.shape)

labels = np.uint8(labels)
print(np.unique(labels))
colors = [[0,0,128],[40,110,170],[0,128,128],[128,128,0],[128,0,0],[75,25,230],[48,130,245],[25,225,255],[60,245,210],[75,180,60],[240,240,70],[200,130,0],[180,30,145],[230,50,240],[128,128,128],[212,190,250],[180,215,255],[0,0,0]]#add few more colors
for i in range(10):

    labeled_color = cv2.cvtColor(labels[i],cv2.COLOR_GRAY2BGR)
    # print(labeled_color.shape)
    for k in range(123):
        for l in range(123):
            if  labeled_color[k][l][0] == 100:
                labeled_color[k][l][0] = colors[17][0]
                labeled_color[k][l][1] = colors[17][1]
                labeled_color[k][l][2] = colors[17][2]
            else:
                labeled_color[k][l][0], labeled_color[k][l][1], labeled_color[k][l][2] = colors[labeled_color[k][l][0]][0],colors[labeled_color[k][l][0]][1],colors[labeled_color[k][l][0]][2]


    # labeled_3 = cv2.merge([labeled_color[i],labeled_color[i],labeled_img[i]])
    
    cv2.imwrite('./custom_ms_results_1/labeled_img'+str(i)+'.png',labeled_color)
