# the range to implement kmean is a grid of image not the whole image
# split the image into grid of square
# collect super pixel of each piece
import numpy as np 
import math
import random
from heapq import nlargest
import cv2 as cv 
from matplotlib import pyplot as plt 

def rgb2lab(inputColor):

    num = 0
    RGB = [0, 0, 0]

    for value in inputColor:
        value = float(value) / 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        RGB[num] = value * 100
        num = num + 1

    XYZ = [0, 0, 0, ]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)

    # Observer= 2°, Illuminant= D65
    XYZ[0] = float(XYZ[0]) / 95.047         # ref_X =  95.047
    XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883

    num = 0
    for value in XYZ:

        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)

        XYZ[num] = value
        num = num + 1

    Lab = [0, 0, 0]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    Lab[0] = round(L, 4)
    Lab[1] = round(a, 4)
    Lab[2] = round(b, 4)

    return Lab

def initiate_k_center(X,k,low_x,high_x,low_y,high_y):
    # randomly initiate first cluster
    # return an array that contain centers of each cluster
    # each center is describle by the number x,y of their position on the picture
    random.seed(0)
    centers = []
    for i in range(0,k):
        x = random.randrange(low_x,high_x)
        y = random.randrange(low_y,high_y)
        center = [X[x][y][0],X[x][y][1],X[x][y][2]]
        center.append(x)
        center.append(y)
        centers.append(center)
    return centers
def classify_data(X,k,clusters,labels,low_x,high_x,low_y,high_y):
    # classify a pixel base on the distance of it to the centers
    # pixel will have class of the neareast cluster
    # the class will be the position of the center in cluster array
    # modify the 2D labels array and return
    # each center is [r,g,b,x,y]
    # labels is a 2D array of all picture
    # need to rethink about range of label
    for i in range(low_x,high_x):
        for j in range(low_y,high_y):
            point_distance=[]
            point = [X[i][j][0],X[i][j][1],X[i][j][2]]
            point.append(i)
            point.append(j)
            for center in clusters:
                point_distance.append(slic_distance(point,center))
            labels[i][j] = point_distance.index(min(point_distance))
    return labels
def update_center(X,k,cluster,labels,low_x,high_x,low_y,high_y):
    # change the para x,y of each elements in cluster array then return cluster array
    # x,y change base on distance to other pixel 
    # clusters array must have k elements
    # each element is (r,g,b,x,y)
    for center_id in range (0,k):
        number_of_pixel = 0
        sum_x = 0
        sum_y = 0 
        for i in range(low_x,high_x):
            for j in range(low_y,high_y):
                if labels[i][j] == center_id:
                    sum_x = sum_x + i
                    sum_y = sum_y + j
                    number_of_pixel = number_of_pixel + 1
        ave_x = sum_x/number_of_pixel
        ave_y = sum_y/number_of_pixel
        cluster[center_id][3] = int(ave_x)
        cluster[center_id][4] = int(ave_y)
    return cluster

def is_cluster_changed(old_cluster,new_cluster):
    
    # check if the new cluster set is different from the one
    if len(old_cluster)!= len(new_cluster):
        return True
    for i in range(0,len(old_cluster)):
        old_x = old_cluster[i][3]
        old_y = old_cluster[i][4]
        new_x = old_cluster[i][3]
        new_y = old_cluster[i][4]
        if old_x != new_x or new_y != old_y:
            return True
    return False
def slic_distance(x,y):
    # X, Y is 2 diff point which have (r,g,b,x,y)
    # X = [r,g,b,x,y]
    d = []
    for i in range(0,len(x)):
        # print("{} : {} {} ".format(i,x[i],y[i]))
        d.append(x[i]-y[i])
    d_color = d[0]*d[0]+d[1]*d[1]+d[2]*d[2]
    d_space = d[3]*d[3]+d[4]*d[4]
    weight_color = 50
    weight_space = math.sqrt(500/8)
    distance = math.sqrt(d_color/(weight_color*weight_color) + d_space/(weight_space*weight_space))
    return distance
def kmeans(X,k,labels,low_x,high_x,low_y,high_y):
    #IMPLEMENT KMEAN
    # the range to implement kmean is a grid of image not the whole image
    # the input should be the image, the boundary of grid and the superpixel of each grid
    # the ouput should be the clusters after clustered
    # or the input shouble be a grid, not all image
    # change the original labels
    centers = []
    old_cluster = []
    i = 0
    centers = initiate_k_center(X,k,low_x,high_x,low_y,high_y)
    while i<2 and is_cluster_changed(old_cluster,centers) == True:
        labels = classify_data(X,k,centers,labels,low_x,high_x,low_y,high_y)
        old_cluster = centers
        centers = update_center(X,k,centers,labels,low_x,high_x,low_y,high_y)
    clusters = [labels,centers]
    return clusters
def get_super_pixel():

    return None
def difine_color(label):
    # convert label into color
    # the formula should depend on mod of 3
    # 0 : [30,0,0]
    # 1 : [0,30,0]
    # 2 : [0,0,30]
    # 3 : [60,0,0]
    # 4 : [30,30,0]
    # 5 : [30,0,30]
    # 6 : [30,0,60]
    # 7 : [0,30,60]
    # 8 : [0,0,90]

    color = [0,0,0]
    devide_3 = int(label/3)
    mod_3 = int(label - 3*devide_3)
    color[mod_3] = color[mod_3]+37
    while devide_3 != 0:
        if devide_3<3:
            color[devide_3] = color[devide_3] + 37
        tmp = devide_3
        devide_3 = int(devide_3/3)
        mod_3 = int(tmp - 3*devide_3)
        color[mod_3] = color[mod_3]+29 
    return color
    # if label < 5:
    #     color[0] = (label+1)*50
    # elif label < 10 and label >4:
    #     color[1] = (label-5+1)*50
    # elif label < 15 and label >9:
    #     color[2] = (label-10+1)*50
    # else :
    #     color[0] = (label-15+1)*50
    #     color[1] = (label-15+1)*50
    # return color
def get_grid_number(length,grid_size):
    grid_number = int(length/grid_size)

    # if grid_number*grid_size >= length:
    grid_number = grid_number - 3
    return grid_number
def slic_initiate_center(X,cluster,h,w,S):
    # initiate centers of superpixels in slic
    # take central points of each grid unit
    clusters = []
    vertical_grid = int(w/S)
    horizonal_grid = int(h/S)
    x = 0
    y = 0
    for i in range(0,horizonal_grid):
        for j in range(0,vertical_grid):
            center = []
            new_x = int(S*(x+i+i+1)/2)
            new_y = int(S*(y+j+j+1)/2)
            center = X[new_x][new_y]
            center.append(new_x)
            center.append(new_y)
            cluster.append(center)
    return clusters
def slic_assign_labels(X,clusters,k,S,labels,distances):
    h,w,channels = X.shape
    for center in clusters:
        if center[3] <= 2*S:
            low_X = 0
        else:
            low_X = center[3]-2*S

        if h-center[3] <= 2*S:
            high_X = h-1
        else:
            high_X = center[3]+2*S 

        if center[4] <= 2*S:
            low_Y = 0
        else:
            low_Y = center[4]-2*S
        
        if w-center[4] <= 2*S:
            high_Y = w-1
        else:
            high_Y = center[4]+2*S

        for i in range(low_X,high_X):
            for j in range(low_Y,high_Y):
                point = X[i][j]
                point.append(i)
                point.append(j)
                di = slic_distance(center,point)
                if di<distances[i][j]:
                    distances[i][j] = di
                    labels[i][j] = clusters.index(center)

def slic_update_center(X,clusters,threshold):
    # update center using slic algorithm 
    # move center to lowest gradient position in area 3x3 around old center
    # graident may be color ???? 
    for center in clusters:
        tmp = center
        min_dis = 100000
        min_tmp = clusters[i]
        for i in range(center[3]-1,center[3]+1):
            for j in range(center[4]-1,center[4]+1):
                tmp_dis = slic_distance(X[i][j],center)
                tmp = X[i][j]
                tmp.append(i)
                tmp.append(j)
                if min_dis > tmp_dis and compute_residual_error(center,tmp)>threshold:
                    min_dis = tmp_dis
                    min_tmp = tmp
        clusters[clusters.index(center)] = min_tmp
    return clusters
def slic_stop_condition(old_cluster,new_cluster,threshold):
    for i in range(0,len(old_cluster)):
        if compute_residual_error(old_cluster[i],new_cluster[i]) > threshold:
            return False
    return True
def slic_kmean():
    
    pass 
def compute_residual_error(new_center,old_center):
    norm = math.sqrt(pow(old_center[3]-new_center[3],2)+pow(old_center[4]-new_center[4],2))
    return norm 
if __name__ == "__main__":
    image = cv.imread("image.jpg")
    h,w,channels = image.shape
    distances = np.zeros((h,w))
    labels = np.zeros((h,w))
    for i in range(0,h):
        for j in range(0,w):
            labels[i][j] = -1
            distances[i][j] = 100000

    print("{} {}".format(h,w))
    # N is number of total pixels
    # k is desired number of superpixels
    # S is grid size interval
    N = h*w
    k = 8000
    S = math.sqrt(N/k)
    # size_of_super_pixel = int(N/number_of_superpixel)
    error_threshold = 0.02
    new_pic = np.zeros((h,w,3))
    start_x = 0
    start_y = 0
    
    # plt.imshow(new_pic[low_x:high_x,low_y:high_y,::-1])
    plt.imshow(new_pic[:,:,::-1])
    plt.show()