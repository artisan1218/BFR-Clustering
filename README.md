# BFR-Clustering
Python implementation of BFR/K-Means algorithms used for large data clustering

## 1. K-Means
This implementation uses randomly generated test dataset.
Note that the dataset here is two-dimensional only for visualization, it supports higher dimensional data however.

Sample data:

![image](https://user-images.githubusercontent.com/25105806/115360470-7ff56900-a174-11eb-81df-42ef344b6d06.png)


#### Function call: 
`
result, centroids = kmeans(k, points_list, max_iterations, initialization='farthest')
`
* k is the number of clusters
* points_list is the data to be clustered in form of list of tuple
* max_iterations limit the max number of iterations kmeans will perform before convergence
* initialization specifies the initialization method used to generate initial centroids. Either 'random' or 'farthest', which is to choose the data points that are far away from each other as possible
* n_init specifies the number of times that kmeans will run on the given dataset. The default number of 3. Kmeans will use inertia to measure how well the clustering is. Running multiple times will prevent the bad clustering result caused by bad initialization of cluster centroids

#### Result:
two variabels will be returned, clustering result and clustering centroids: 
`
result, centroids
`
The clustering result is shown below


```
k = 3
max_iterations = 10
result, centroids= kmeans(k, points_list, max_iterations, 'random', 3)
idx_points_list = list()
for i in range(k):
    idx_points_list.append([pair[0] for pair in list(filter(lambda pair: pair[1]==i, result))])

plt.figure(1)
for idx_points in idx_points_list:
    plt.scatter([pair[0] for pair in idx_points],[pair[1] for pair in idx_points], marker = '.')
```

Result:

![image](https://user-images.githubusercontent.com/25105806/115360928-f003ef00-a174-11eb-9153-736ded71d668.png)

Scikit-learn KMeans result on the same dataset for comparsion:

![image](https://user-images.githubusercontent.com/25105806/115361066-1164db00-a175-11eb-867d-7f9cc2c22060.png)


## 2. Bradley, Fayyad and Reina (BFR) algorithm
Note: the implementation uses Spark to load the data from sample dataset.

#### Algorithm introduction:
BFR only keeps track of three different type of sets:
* DS: Discard Set, which includes points that are close enough to be summarized.
* CS: Compression Set, which includes group of points that are close enough together but not close to any existing centroids.
* RS: Retained Set, which includes points that are not close to any of the centroids or other points.

Points in DS and CS will be summarized using
* N: number of points in this set.
* SUM: the sum of points coordinates in each dimension.
* SUMSQ: the square of sum of points coordinates in each dimension.

Then these summarized points will be discarded.

#### Function call: 
Since BFR only cares about above-mentioned three types of set, the detail process of generating these 3 sets and ways of assigning points to them are not fixed. So there are several hyperparameters to control the generation and point assignment.
