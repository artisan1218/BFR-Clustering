# BFR-Clustering
Python implementation of BFR/K-Means algorithms used for large data clustering

## 1. K-Means
This implementation uses randomly generated test dataset.
Note that the dataset here is two-dimensional only for visualization, it supports higher dimensional data however.

![image](https://user-images.githubusercontent.com/25105806/114322176-9a27ab00-9ad3-11eb-9f84-76a9a03dec4c.png)

#### Function call: 
`
result, centroids = kmeans(k, points_list, max_iterations, initialization='farthest')
`
* k is the number of clusters
* points_list is the data to be clustered in form of list of tuple
* max_iterations limit the max number of iterations kmeans will perform before convergence
* initialization specifies the initialization method used to generate initial centroids. Either 'random' or 'farthest', which is to choose the data points that are far away from each other as possible

#### Result:
two variabels will be returned, clustering result and clustering centroids: 
`
result, centroids
`
The clustering result is shown below


```
k = 4
max_iterations = 10
result, centroids = kmeans(k, points_list, max_iterations, 'farthest')
idx_points_list = list()
for i in range(k):
    idx_points_list.append([pair[0] for pair in list(filter(lambda pair: pair[1]==i, result))])

plt.figure(1)
for idx_points in idx_points_list:
    plt.scatter([pair[0] for pair in idx_points],[pair[1] for pair in idx_points], marker = '.')
```

![image](https://user-images.githubusercontent.com/25105806/114322336-8466b580-9ad4-11eb-8cf8-1177ab61cb10.png)
