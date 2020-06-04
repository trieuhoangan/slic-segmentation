import numpy as np 
import math
import random
from heapq import nlargest
def initiate_k_center(X,k,low_x,high_x,low_y,high_y):
    # randomly initiate first cluster
    # return an array that contain centers of each cluster
    # each center is describle by the number x,y of their position on the picture
    random.seed(0)
    centers = []
    for i in range(0,k):
        x = random.randrange(low_x,high_x)
        y = random.randrange(low_y,high_y)
        center = X[x][y]
        center.extend(x)
        center.extend(y)
        centers.append(center)
    return centers
def classify_data(X,k,low_x,high_x,low_y,high_y,clusters,labels):
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
            point = X[i][j]
            point.extend(i)
            point.extend(j)
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
        d[i] = x[i]-y[i]
    d_color = d[0]*d[0]+d[1]*d[1]+d[2]*d[2]
    d_space = d[3]*d[3]+d[4]*d[4]
    weight_color = 2
    weight_space = 40
    distance = math.sqrt(d_color/(weight_color*weight_color) + d_space/(weight_space*weight_space))
    return distance
def train():
    # traing the model
    return None
def kmeans(X,k):
    #IMPLEMENT KMEAN
    labels = []
    center = []
    clusters = (labels,center)
    return clusters