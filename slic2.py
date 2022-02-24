# the range to implement kmean is a grid of image not the whole image
# split the image into grid of square
# collect super pixel of each piece
import numpy as np 
import math
import random
from heapq import nlargest
import cv2 as cv 
from matplotlib import pyplot as plt 
import os
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
    weight_color = 10
    weight_space = 100
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
    clusters = centers
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
def slic_initiate_center(X,h,w,S):
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
            new_x = int(S*(i+0.5))
            new_y = int(S*(j+0.5))
            # print(len(X[new_x][new_y]))
            center.extend( X[new_x][new_y])

            center.append(new_x)
            center.append(new_y)
            # print(len(center))
            clusters.append(center)
    return clusters
def slic_assign_labels(X,clusters,k,S,labels,distances):
    h,w,channels = X.shape

    for l in range(0,len(clusters)):
        center = clusters[l]
        if center[3] <= 2*S:
            low_X = 0
        else:
            low_X = center[3]-2*S

        if h - center[3] <= 2*S:
            high_X = h-1
        else:
            high_X = center[3]+2*S 

        if center[4] <= 2*S:
            low_Y = 0
        else:
            low_Y = center[4] - 2*S
        
        if w-center[4] <= 2*S:
            high_Y = w-1
        else:
            high_Y = center[4]+2*S
        
        for i in range(int(low_X),int(high_X)):
            for j in range(int(low_Y),int(high_Y)):
                point = []
                point.extend(X[i][j])
                point.append(i)
                point.append(j)
                di = slic_distance(center,point)
                # print(di)
                if di < distances[i][j]:
                    distances[i][j] = di
                    labels[i][j] = l
    return labels
def slic_update_center(X,clusters,threshold):
    # update center using slic algorithm 
    # move center to lowest gradient position in area 3x3 around old center
    # graident may be color ???? 
    for center in clusters:
        tmp = center
        min_dis = 100000
        min_tmp = clusters[0]
        for i in range(center[3]-1,center[3]+1):
            for j in range(center[4]-1,center[4]+1):
                tmp = []
                tmp.extend(X[i][j])
                tmp.append(i)
                tmp.append(j)
                tmp_dis = slic_distance(tmp,center)
                if min_dis > tmp_dis and compute_residual_error(center,tmp)>threshold:
                    min_dis = tmp_dis
                    min_tmp = tmp
        clusters[clusters.index(center)] = min_tmp
    return clusters
def slic_stop_condition(old_cluster,new_cluster,threshold):
    if len(old_cluster) == 0:
        return False
    for i in range(0,len(old_cluster)):
        if compute_residual_error(old_cluster[i],new_cluster[i]) > threshold:
            return False
    return True
def slic_kmean(X,N,k,S,cluster,labels,distances,threshold):
    # step 1 : init
    # step 2 : assign labels
    # step 3 : update centers
    # step 4 : check stop condition
    h,w,c = X.shape
    old_cluster = []
    clusters = slic_initiate_center(X,h,w,S)
    slic_assign_labels(X,clusters,k,S,labels,distances)
    while slic_stop_condition(old_cluster,clusters,threshold) == False:
        old_cluster = clusters
        clusters = slic_update_center(X,clusters,threshold)
        slic_assign_labels(X,clusters,k,S,labels,distances)
    return clusters
def compute_residual_error(new_center,old_center):
    norm = math.sqrt(pow(old_center[3]-new_center[3],2)+pow(old_center[4]-new_center[4],2))
    return norm
def segment_cloud(ima_folder,ima_name,slic_path,result_path,indexed_path):
    image = cv.imread("{}\{}".format(ima_folder,ima_name))
    h,w,channels = image.shape
    distances = np.zeros((h,w))
    labels = np.zeros((h,w),dtype=int)
    CIELAB_pic = np.zeros((h,w,3))
    for i in range(0,h):
        for j in range(0,w):
            labels[i][j] = -1
            distances[i][j] = 100000
            CIELAB_pic[i][j] = rgb2lab(image[i][j])
    
    print("{} {}".format(h,w))
    # N is number of total pixels
    # k is desired number of superpixels
    # S is grid size interval
    N = h*w
    S = 96
    k = int(N/(S*S)) + 1
    error_threshold = 0.2
    cluster = []
    cluster = slic_kmean(CIELAB_pic,N,k,S,cluster,labels,distances,error_threshold)
    # cluster = slic_initiate_center(CIELAB_pic,h,w,S)
    # labels = slic_assign_labels(CIELAB_pic,cluster,k,S,labels,distances)

    new_pic = np.zeros((h,w,3),dtype = int)
    print(len(cluster))
    

    print("Done clustering")
    color_code = np.zeros((len(cluster),3),dtype=int)
    color_count = np.zeros((len(cluster)),dtype=int)
    for i in range(0,h):
        for j in range(0,w):
            label = labels[i][j]
            for k in range (0,2):
                color_code[label][k] = color_code[label][k] + image[i][j][k]
            color_count[label] = color_count[label] + 1
    for i in range(0,len(color_code)):
        for k in range(0,2):
            color_code[i][k] = int(color_code[i][k]/color_count[i])
    for i in range(0,h):
        for j in range(0,w):
            label = labels[i][j]
            new_pic[i][j] = color_code[label]
    for i in range(1,h-1):
        for j in range(1,w-1):
            if labels[i][j] < 0 :
                new_pic[i][j] = [0,0,0]
                print(-1)
            else:
                label = labels[i][j]
                
                for k in range(i-1,i+1):
                    for l in range(j-1,j+1):
                        if labels[k][l] != label:
                            image[k][l] = [0,255,255]
    
    cv.imwrite("{}\{}".format(slic_path,ima_name),image)       
    cv.imwrite("{}\{}".format(indexed_path,ima_name),new_pic)   
    for i in range(0,h):
        for j in range(0,w):
            ave = (new_pic[i][j][0] + new_pic[i][j][1] + new_pic[i][j][2])/3
            if ave > 60:
                new_pic[i][j] = [255,255,255]
            else:
                new_pic[i][j] = [0,0,0]
    cv.imwrite("{}\{}".format(result_path,ima_name),new_pic)   
    # print(new_pic.shape)
    
    # plt.imshow(image[:,:,::-1])
    # plt.show()
def check_GT(GTMaps_path,result_path):
    GTs = []
    results = []
    for filename in os.listdir(GTMaps_path):
        GTs.append(filename)
    for filename in os.listdir(result_path):
        results.append(filename)
    record_file = open("custom_slic_results.txt","w",encoding="utf-8")
    total_acc = 0
    for i in range(0,len(results)):
        result = "{}/{}".format(result_path,results[i])
        GT = "{}/{}".format(GTMaps_path,GTs[i])
        result_ima = cv.imread(result,0)
        GT_ima = cv.imread(GT,0)
        h,w = GT_ima.shape
        same_path = 0
        for k in range(0,h):
            for l in range(0,w):
                if result_ima[k][l] == GT_ima[k][l]:
                    same_path = same_path + 1
        acc = same_path/(h*w)
        total_acc = total_acc + acc
        record_file.write("{} have accuracy {}\n".format(results[i],acc))
        print(i)
    total_acc = total_acc/len(results)
    record_file.write("average accuracy :{}\n".format(total_acc))
    record_file.close
if __name__ == "__main__":
    images_path  = "images"
    GTMaps_path = "GTmaps"
    result_path = "custom_slic_16superpix_res"
    indexed_path = "custom_slic_16spuperpix_index"
    slic_path = "custom_slic_superpix16_image"
    counter = 0
    for filename in os.listdir(images_path):
        segment_cloud(images_path,filename,slic_path,result_path,indexed_path)
        counter = counter + 1
        print(counter)
    check_GT(GTMaps_path,result_path)