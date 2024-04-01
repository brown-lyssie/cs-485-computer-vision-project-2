# Author: Lyssie Brown
# Instructions: Dr. Emily Hand
# Project name: Project 2
# Purpose: Implement functions for loading, displaying, 

import numpy as np
import cv2
import math
moravec_threshold = 11000;



# Load an image, return an image (numpy array)
def load_img(file_name):
    img = cv2.imread(file_name, 1) # will do in RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # BGR -> gray
    return img

def display_img(img):
    cv2.imshow("Displaying Image", img)
    cv2.waitKey(0)

def moravec_detector(img):
    feature_coords = []
    pad_amt = 2
    img_padded = np.pad(img, ((pad_amt,pad_amt),(pad_amt,pad_amt)), mode="constant")
    for y in range(pad_amt+2,len(img_padded)-pad_amt-2):
        for x in range(pad_amt+2,len(img_padded[0])-pad_amt-2):
            s = 0;
            # grab the window
            center_window = get_3x3_window(x,y, img_padded)
            for window_y in (-1,0,1):
                for window_x in (-1,0,1):
                    new_window = get_3x3_window(x+window_x, y+window_y, img_padded);
                    s_xy = subtract_and_square_elementwise(center_window, new_window)
                    if s_xy > s:
                        s = s_xy
            if s > moravec_threshold:
                feature_coords.append([x-pad_amt,y-pad_amt])
    return feature_coords


def get_3x3_window(center_x, center_y, array):
    # print(array[center_y-1:center_y+2, center_x-1:center_x+2])
    return array[center_y-1:center_y+2, center_x-1:center_x+2]

def subtract_and_square_elementwise(window1, window2):
    total = 0
    for r_index in range(len(window1)):
        for c_index in range(len(window2)):
            total += float(window1[r_index][c_index]) - float(window2[r_index][c_index])
    return total**2


def plot_keypoints(image, keypoints):
    plotted_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x,y in keypoints:
        plotted_img[y,x] = [0,0,255]
    return plotted_img

def extract_LBP(image, keypoint):
    lbps = []
    for delta_y in range(-7,9):
        for delta_x in range (-7,9):
            window = get_3x3_window(keypoint[0]+delta_x, keypoint[1]+delta_y, image)
            lbp = calc_lbp_for_window(window)
            lbps.append(lbp)
    print(len(lbps))
    return make_histogram(lbps)

def calc_lbp_for_window(window):
    lbp = 0;
    bitnum = 7;
    for x,y in ([0,0], [0,1], [0,2], [1,0], [1,2], [2,0], [2,1], [2,2]):
        if window[y][x] >= window[1][1]:
            lbp |= 2**bitnum
        bitnum-=1
    return lbp
def make_histogram(array2d):
    histogram = [float(0) for i in range(256)]
    for val in array2d:
        print(val)
        histogram[val] += 1
    for i in range(256):
        histogram[i] /= 256
    return histogram