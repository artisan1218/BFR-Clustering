#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import random
import math


# In[2]:


# Test datapoints
num_points = 200
points_list = list()
for _ in range(num_points):
    p = random.randint(0,10)
    if p<6:
        point = (random.uniform(0, 7), random.uniform(0, 7))
    elif p>=6 and p<8:
        point = (random.uniform(6, 8), random.uniform(6, 8))
    else:
        point = (random.uniform(7, 10), random.uniform(7, 10))
    points_list.append(point)


# In[3]:


plt.scatter([pair[0] for pair in points_list],[pair[1] for pair in points_list], marker = '.')


# In[82]:


def kmeans(k, points_list, max_iter, initialization):
    centroids_dict = dict() # map centroid tuple to unique and index
    if initialization=='random':
        for cluster_idx in range(k):
            random_idx = random.randrange(0, len(points_list))
            centroids_dict[points_list[random_idx]] = cluster_idx
        # centroids is the list of initial centorids
    elif initialization=='farthest':
        #initialize the initial centroids by farthest distance from each other
        curr_sum_distance = 0
        closest_distance = 0 
        for kth_centroid in range(k): #first centroid is still randomly picked
            if kth_centroid == 0:
                random_idx = random.randrange(0, len(points_list))
                centroids_dict[points_list[random_idx]] = kth_centroid
            else:
                curr_sum_distance = 0
                # compute sum of distance to the current centroids to each other point
                for point in points_list:
                    if point not in centroids_dict.keys():
                        for curr_cent in centroids_dict.keys():
                            curr_sum_distance += euclideanDist(curr_cent, point)
                        if curr_sum_distance > closest_distance:
                            closest_distance = curr_sum_distance
                            closest_centroid = point
                        curr_sum_distance = 0
                centroids_dict[closest_centroid] = kth_centroid
    else:
        return 'initialization method not recognized'

    convergence = False
    curr_iter = 0
    while not convergence and curr_iter<max_iter:
        # assign points to clusters
        cluster_list = list() # [(point, centroid_idx),(point, centroid_idx), ...]
        for point in points_list:
            #this point should be assigned to return value centroid
            centroid = findCentroid(point, centroids_dict.keys()) 
            centroid_idx = centroids_dict[centroid]
            cluster_list.append((point, centroid_idx))

        # update centroids
        centroids_comp = list()
        centroids_dict_items = list(centroids_dict.items())
        for old_centroid, idx in centroids_dict_items:
            idx_cluster = list(filter(lambda pair: pair[1]==idx, cluster_list))

            # list of points in this cluster
            points = [pair[0] for pair in idx_cluster]
            # calculate new centroid
            new_centroid = calculateNewCentroid(points, len(old_centroid))
            # update centroids_dict
            if new_centroid != old_centroid:
                # new centroid has same cluster index
                # add new centroid
                centroids_dict[new_centroid] = centroids_dict[old_centroid] 
                # delete old centroid
                del centroids_dict[old_centroid]
            
            centroids_comp.append((old_centroid, new_centroid))
        
        # test convergence 
        curr_iter += 1
        convergence = all([euclideanDist(pair[0], pair[1])<=0.01 for pair in centroids_comp])
        
    cluster_list = list() # [(point, centroid_idx),(point, centroid_idx), ...]
    for point in points_list:
        #this point should be assigned to return value centroid
        centroid = findCentroid(point, centroids_dict.keys()) 
        centroid_idx = centroids_dict[centroid]
        cluster_list.append((point, centroid_idx))
    return cluster_list, list(centroids_dict.keys())


# In[71]:


def calculateNewCentroid(clusterPoints, dimensions):
    coord_list = list()
    for d in range(dimensions):
        coord_list.append([pt[d] for pt in clusterPoints]) # get d-th coordinates of all points 
    return tuple([sum(d)/len(d) for d in coord_list])

def euclideanDist(pt, cent):
    sumsq = 0
    for d in range(len(pt)):
        sumsq += (pt[d] - cent[d]) ** 2
    return math.sqrt(sumsq)

def findCentroid(pt, centroids):
    reuslt_list = list()
    for centroid in centroids:
        ED = euclideanDist(pt, centroid) #Euclidean distance
        reuslt_list.append((ED, centroid))
    return sorted(reuslt_list, key=lambda pair:pair[0])[0][1]# return the centroid with min ED to pt


# In[83]:


k = 4
max_iterations = 10

result, centroids = kmeans(k, points_list, max_iterations, 'farthest')

idx_points_list = list()
for i in range(k):
    idx_points_list.append([pair[0] for pair in list(filter(lambda pair: pair[1]==i, result))])

plt.figure(1)
for idx_points in idx_points_list:
    plt.scatter([pair[0] for pair in idx_points],[pair[1] for pair in idx_points], marker = '.')


# In[ ]:




