import numpy as np
import os
import sys
from fast_slic import Slic
from PIL import Image
from matplotlib import pyplot as plt   
import cv2 as cv

def segment(image_folder,image_name):
    # with Image.open("{}\{}".format(image_folder,image_name)) as f:
    #     image = np.array(f)
    image = cv.imread("{}\{}".format(image_folder,image_name))
    # h,w,channels = image.shape
    num_superpixels = 1000
    slic = Slic(num_components=num_superpixels, compactness=5)
    assignment = slic.iterate(image) # Cluster Map
    h = len(assignment)
    w = len(assignment[0])
    new_image = cv.imread("{}\{}".format(image_folder,image_name))
    segmented_picture = np.zeros((h,w,3),dtype=int)
    color_code = np.zeros((num_superpixels,3),dtype=int)
    color_count = np.zeros((num_superpixels),dtype=int)
    binary_picture = np.zeros((h,w),dtype=int)
    print("{} {}".format(h,w))
    for i in range(1,h-1):
        for j in range(1,w-1):  
            # print("{} {}".format(i,j))
            color_code[assignment[i][j]][0] = color_code[assignment[i][j]][0] + new_image[i][j][0]
            color_code[assignment[i][j]][1] = color_code[assignment[i][j]][1] + new_image[i][j][1]
            color_code[assignment[i][j]][2] = color_code[assignment[i][j]][2] + new_image[i][j][2]
            color_count[assignment[i][j]] = color_count[assignment[i][j]] + 1 
    for i in range(0,num_superpixels):
        if color_count[i]!=0:
            color_code[i][0] = int(color_code[i][0]/color_count[i])
            color_code[i][1] = int(color_code[i][1]/color_count[i])
            color_code[i][2] = int(color_code[i][2]/color_count[i])
    # print(color_code)
    for i in range(1,h):
        for j in range(1,w):  
            label = assignment[i][j]
            segmented_picture[i][j] = color_code[label]
    cv.imwrite("indexed_ima/{}".format(image_name),segmented_picture)
    for i in range(1,h):
        for j in range(1,w):
            for k in range (i-1,i+1):
                for l in range ( j-1,j+1):
                    if assignment[i][j] != assignment[k][l]:
                        new_image[k][l] = [0,255,255]
    for i in range(1,h):
        for j in range(1,w): 
            ave_val = int((segmented_picture[i][j][0]+segmented_picture[i][j][1]+segmented_picture[i][j][2])/3)
            if ave_val > 100:
                binary_picture[i][j] = 255
            else:
                binary_picture[i][j] = 0

    cv.imwrite("results/{}".format(image_name),binary_picture)
    cv.imwrite("slic/{}".format(image_name),new_image)
    #         # segmented_picture[i][j] = 
    # plt.subplot(121),plt.imshow(new_image[:,:,::-1])
    # plt.title('slic image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(segmented_picture[:,:,::-1])
    # plt.title('segmented image'), plt.xticks([]), plt.yticks([])
    # plt.show()

if __name__ == "__main__":
    images_path  = "images"
    GTMaps_path = "GTmaps"
    result_path = "results"
    counter = 0
    for filename in os.listdir(images_path):
        segment(images_path,filename)
        counter = counter + 1
        print(counter)
    GTs = []
    results = []
    for filename in os.listdir(GTMaps_path):
        GTs.append(filename)
    for filename in os.listdir(result_path):
        results.append(filename)
    record_file = open("results.txt","w",encoding="utf-8")
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
    total_acc = total_acc/len(GTs)
    record_file.write("average accuracy :{}\n".format(total_acc))
    record_file.close

