# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:14:31 2019

Here we want to propose a metrics that can measure the similarity of 
predicted vs real label data in soil erosion detection project. 

we measure the morphological difference between two binary matrices

The algorithm find closest point iterative, steps are:

1. For each point in the predicted data, match the closest point in the real data;
2. Measure the distance between these two points; (Manhattan distance)
3. Iterate (through all the points in the predicted dataset).

4. If there are unmatched points in the real data, for each points, match the closest point
   in the predicted data;
5. Measure the distance between two points;
6. Iterate through all the unmatched points.

@author: shirui
"""

import numpy as np
import matplotlib.pyplot as plt

def match_point(point, real_idx): ## Brute force
    """
    return Distance of nearest cell having 1 in a binary matrix
    """
    x_, y_ = point
    mymindist = 1e5
    for i in range(len(real_idx[0])):
        dist = abs(real_idx[0][i]-x_)+abs(real_idx[1][i]-y_) #Manhattan distance 
        if dist<mymindist:
            mymindist = dist
            matched_point = (real_idx[0][i], real_idx[1][i])
        
    return mymindist, matched_point
         
def distance_measure(real, pred):
    """ 
    real and pred are 2D numpy arrays with same dimension
    Measure the morphological difference between two binary matrices
    """
    
    real_idx = np.nonzero(real)
    pred_idx = np.nonzero(pred)
    distSum = 0
    matched_point_list = set()
    
    if (len(real_idx[0])==0):
        if (len(pred_idx[0])==0):
            return 0
        else: 
            distSum = len(pred_idx[0])
            return distSum
          
    for i in range(len(pred_idx[0])):
        point = (pred_idx[0][i], pred_idx[1][i])
        dist, matched_point = match_point(point, real_idx)
        distSum += dist
        matched_point_list.add(matched_point)

    real_idx_converted = set()  
    for i in range(len(real_idx[0])):
        real_idx_converted.add((real_idx[0][i],real_idx[1][i]))
        
    if real_idx_converted.difference(matched_point_list) is not None:
        for points in (real_idx_converted.difference(matched_point_list)):
            dist, matched_point = match_point(points, pred_idx)
            distSum += dist
    
    """
    plt.imshow(real)
    plt.show()
    plt.imshow(pred)
    plt.show()
    print('distance between two matrices: %d' %distSum)        
    """        
    return distSum       
        
    
if __name__ == "__main__":
    real = np.random.randint(2, size=36)
    real = np.reshape(real, (6,6))
    pred = np.random.randint(2, size=36)
    pred = np.reshape(pred, (6,6)) 
    
    real = np.array([[0, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0]])
    
    pred1 = np.array([[0, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]])
    
    pred2 = np.array([[0, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 1, 1, 0],
     [0, 0, 0, 0, 0, 0]])
    
    pred3 = np.array([[0, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1, 0],
     [1, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0]])
    
    
    distance_measure(real, pred1)
    distance_measure(real, pred2)
    distance_measure(real, pred3)
