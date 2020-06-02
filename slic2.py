import numpy as np 
import math
import random
def initiate_k_center(k,low_x,high_x,low_y,high_y):
    # randomly initiate first cluster
    # return an array that contain centers of each cluster
    # each center is describle by the number x,y of their position on the picture
    random.seed(0)
    centers = []
    for i in range(0,k):
        x = random.randrange(low_x,high_x)
        y = random.randrange(low_y,high_y)
        centers.append((x,y))
    return centers
def classify_data(X,k,clusters,labels):
    # classify a pixel base on the distance of it to the centers
    # pixel will have class of the neareast cluster
    # the class will be the position of the center in cluster array
    # modify the 2D labels array and return

    return labels
def update_center(X,k,cluster):
    return None
def is_changed(old_cluster,new_cluster):
    return True
def distance(x,y):
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
    return None
def kmeans(X,k):
    labels = []
    center = []
    clusters = (labels,center)
    return clusters