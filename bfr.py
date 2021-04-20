import findspark
findspark.init()

from pyspark import SparkContext, SparkConf
import time
import sys
import os
import random
import math
import json
from itertools import combinations
from copy import deepcopy


startTime = time.time()

conf = SparkConf().setMaster("local[*]")         .set("spark.executor.memory", "4g")         .set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf).getOrCreate()


def summarize(points_cluster_list):
    summary = list()
    for cluster in points_cluster_list:
        cluster_dict = dict() # {N:n, SUM:sum, SUMSQ:sumsq}
        cluster_dict['N'] = len(cluster)
        SUM, SUMSQ = getSUMandSUMSQ(cluster)
        cluster_dict['SUM'] = SUM
        cluster_dict['SUMSQ'] = SUMSQ
        summary.append(cluster_dict)
    return summary
            
def getSUMandSUMSQ(cluster):
    cluster_points = [pair[0] for pair in cluster]
    for idx, point in enumerate(cluster_points):
        sq_point = tuple([digit**2 for digit in point])
        if idx==0:
            SUM_result = point
            SUMSQ_result = sq_point
        else:
            SUM_result = [sum(x) for x in zip(SUM_result, point)]
            SUMSQ_result = [sum(x) for x in zip(SUMSQ_result, sq_point)]
    return tuple(SUM_result), tuple(SUMSQ_result)

def groupPointsByCluster(k, cluster_result):
    points_cluster_list = list()
    for kth in range(k):
        l = list(filter(lambda pair:pair[1]==kth, cluster_result))
        if len(l)!=0:
            points_cluster_list.append(l)
    points_cluster_list = sorted(points_cluster_list, key=lambda cluster:len(cluster))
    return points_cluster_list

def kmeans(k, points_list, max_iter, initialization, n_init=1):
    best_centroids = dict()
    best_inertia = float('inf')
    for n in range(n_init):
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
                    closest_centroid = points_list[0] # in case there is only one point in points_list
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
                # a cluster might be empty due to bad initialization
                if len(idx_cluster) == 0:
                    # simply assign a random point to that cluster
                    points = [points_list[random.randrange(0, len(points_list))]]
                else:
                    # list of points in this cluster
                    points = [pair[0] for pair in idx_cluster]

                # calculate new centroid
                new_centroid = calculateNewCentroid(points, len(old_centroid))
                # update centroids_dict
                if new_centroid != old_centroid:
                    # new centroid has same cluster index
                    # add new centroid
                    centroids_dict[new_centroid] = idx
                    # delete old centroid
                    del centroids_dict[old_centroid]

                centroids_comp.append((old_centroid, new_centroid))

            # test convergence 
            curr_iter += 1
            convergence = all([euclideanDist(pair[0], pair[1])==0 for pair in centroids_comp])
        # after convergence, calculate interia
        curr_inertia = 0
        for point in points_list:
            #this point should be assigned to return value centroid
            centroid = findCentroid(point, centroids_dict.keys())
            curr_inertia += euclideanDist(point, centroid) ** 2
        if curr_inertia < best_inertia:
            best_inertia = curr_inertia
            best_centroids = deepcopy(centroids_dict)
        
    cluster_list = list() # [(point, centroid_idx),(point, centroid_idx), ...]
    for point in points_list:
        #this point should be assigned to return value centroid
        centroid = findCentroid(point, best_centroids.keys()) 
        centroid_idx = best_centroids[centroid]
        cluster_list.append((point, centroid_idx))
    return cluster_list, list(best_centroids.keys())

def calculateNewCentroid(clusterPoints, dimensions):
    coord_list = list()
    for d in range(dimensions):
        coord_list.append([pt[d] for pt in clusterPoints]) # get d-th coordinates of all points
    return tuple([sum(dim)/len(dim) for dim in coord_list])

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

def updateSummary(summary, idx, point):
    summary[idx]['N'] += 1
    summary[idx]['SUM'] = tuple([sum(x) for x in zip(summary[idx]['SUM'], point)])
    sq_point = tuple([digit**2 for digit in point])
    summary[idx]['SUMSQ'] = tuple([sum(x) for x in zip(summary[idx]['SUMSQ'], sq_point)])

def mahalanobisDist(point, summary):
    md = 0
    N = summary['N']
    for d in range(len(point)):
        d_SUM = summary['SUM'][d]
        d_SUMSQ = summary['SUMSQ'][d]
        d_centroid = d_SUM / N
        sd = math.sqrt((d_SUMSQ / N) - (d_SUM / N) ** 2)
        if sd==0:
            normalization = 0
        else:
            normalization = ((point[d] - d_centroid) / sd) ** 2
        md += normalization
    return math.sqrt(md)

def mergeCS(CS_summary):
    merged_CS_idx_set = set()
    new_CS_list = list()

    MD_pair_list = list()
    # generate combinations of CS based on their indices
    CS_idx_comb = list(combinations(range(len(CS_summary)), 2))
    for CS_idx_tuple in CS_idx_comb: 
        summary1 = CS_summary[CS_idx_tuple[0]]
        summary2 = CS_summary[CS_idx_tuple[1]]
        MD, thres = getNearness(summary1, summary2)
        if MD < thres:
            MD_pair_list.append((MD, CS_idx_tuple))
    MD_pair_list = sorted(MD_pair_list, key=lambda pair:pair[0])
    
    for MD_pair in MD_pair_list:
        MD = MD_pair[0]
        CS_idx_tuple = MD_pair[1]
        # if none of these two CS have been merged before
        if CS_idx_tuple[0] not in merged_CS_idx_set and CS_idx_tuple[1] not in merged_CS_idx_set:
            # merge these two CS into one
            summary1 = CS_summary[CS_idx_tuple[0]]
            summary2 = CS_summary[CS_idx_tuple[1]]
            N = summary1['N'] + summary2['N']
            SUM = tuple([sum(zipped) for zipped in zip(summary1['SUM'], summary2['SUM'])])
            SUMSQ = tuple([sum(zipped) for zipped in zip(summary1['SUMSQ'], summary2['SUMSQ'])])
            new_CS = {'N':N, 'SUM':SUM, 'SUMSQ':SUMSQ}
            new_CS_list.append(new_CS) # append merged CS
            merged_CS_idx_set.add(CS_idx_tuple[0])
            merged_CS_idx_set.add(CS_idx_tuple[1])
            
    # get the list of CS indices that are not merged
    unmerged_CS_idx_list = list(set(range(len(CS_summary))) - merged_CS_idx_set)
    
    # append the not merged CS back to CS list
    for unmerged_CS_idx in unmerged_CS_idx_list:
        new_CS_list.append(CS_summary[unmerged_CS_idx])
    
    return new_CS_list 
        
def getNearness(CS1, CS2):
    # nearness is defined as the average MD between a centroid of a cluster to another cluster and vice versa
    cent1 = tuple([d/CS1['N'] for d in CS1['SUM']])
    cent2 = tuple([d/CS2['N'] for d in CS2['SUM']])
    CS1cent2CS2_MD = mahalanobisDist(cent1, CS2)
    CS2cent2CS1_MD = mahalanobisDist(cent2, CS1)
    threshold = alpha_threshold * math.sqrt(len(cent1))
    return (CS1cent2CS2_MD + CS2cent2CS1_MD)/2, threshold

def generateIntermediateResult(round_num, DS_summary, CS_summary, RS_list):
    nof_point_discard = str(sum([DS['N'] for DS in DS_summary]))
    nof_point_compression = str(sum([CS['N'] for CS in CS_summary]))
    line = str(round_num) + ',' + str(len(DS_summary)) + ','            + nof_point_discard + ',' + str(len(CS_summary)) + ','            + nof_point_compression + ',' + str(len(RS_list))
    return line



input_folder_path = r'C:\Users\11921\Downloads\data\test3'
k = 10
output_path_cluster = r'points_assignment.json'
output_path_intermediate_result = r'intermediate_result.csv'


csv_files_list = os.listdir(input_folder_path)
large_num_centroids = 3 * k
max_iterations = 10
very_few = 500
alpha_threshold = 2
sample_ratio = 0.7
kmeans_initial_cent = 'random' #random
generateCSRS = 'complement' #'outliers', method of generating CS and RS, either from complement points or from outliers
cache_datapoints = True
merge_CS = False
all_datapoints = list()

intermediate_result_list = list()
for round_idx, csv in enumerate(sorted(csv_files_list)):
    '''
    Step 1, 7. load data 
    '''
    if cache_datapoints:
        full_csv_path = input_folder_path + "\\" + csv
        raw_points = (sc.textFile(full_csv_path)  # read a data chunk
                        .map(lambda line:line.split(',')) # split the raw data string by comma                         
                        .map(lambda pt:[float(coord) for coord in pt]) # cast the str coordinates to float
                        .map(lambda pt:(str(int(pt[0])), tuple(pt[1:])))  # form tuple of idx and coordinates
                        .collect()
                       )
        points_list = [pair[1] for pair in raw_points]
        all_datapoints.extend(raw_points)
        
    else:
        full_csv_path = input_folder_path + "\\" + csv
        points_list = (sc.textFile(full_csv_path)  # read a data chunk
                        .map(lambda line:line.split(',')) # split the raw data string by comma                         
                        .map(lambda pt:[float(coord) for coord in pt]) # cast the str coordinates to float
                        .map(lambda pt: tuple(pt[1:])) # keep only the point coordinates
                        .collect()
                       )
    
    if round_idx == 0: # first round, run k-means on a subset to initialize
        # sample the points list to make it smaller
        subset_points_list = random.sample(points_list, int(len(points_list) * sample_ratio))
        comp_points_list = list(set(points_list) - set(subset_points_list))# remaining points of the first load of data
        
        if generateCSRS == 'outliers':
            '''
            Step 2. run k-means on a subset of data points (to generate outliers and inliers)
            '''
            # cluster_result: [((point1), cluster_index), ((point2), cluster_index), ...]
            # centroids: [(centroid1), (centroid2), ...]
            cluster_result, _ = kmeans(large_num_centroids, subset_points_list, max_iterations, kmeans_initial_cent)

            # put together the points that have been clustered into a same cluster 
            points_cluster_list = groupPointsByCluster(large_num_centroids, cluster_result)
            del cluster_result
        
            '''
            Step 3. move clusters with few points to outliers, otherwise to inliers
            '''
            outliers_list = list()
            inliers_list = list()
            for cluster in points_cluster_list:
                if len(cluster) <= very_few:
                    outliers_list.append(cluster)
                else:
                    inliers_list.append(cluster)
            del points_cluster_list
            outlier_points_list = [pair[0] for cluster in outliers_list for pair in cluster]

            '''
            Step 4. run kmeans again on inliers to cluster inlier data to k clusters. Use k clusters as DS and summarize
            '''
            inlier_points_list = [pair[0] for cluster in inliers_list for pair in cluster]
            inliers_points_result, _ = kmeans(k, inlier_points_list, max_iterations, kmeans_initial_cent)

            # put together the points that have been clustered into a same cluster 
            inliers_cluster_result = groupPointsByCluster(k, inliers_points_result)
            DS_summary = summarize(inliers_cluster_result)
            # discard the points
            del inliers_list # original inliers points with index
            del inlier_points_list # original inliers points
            del inliers_points_result # cluster result of inliers points
            del inliers_cluster_result # cluster result of inliers points grouped by cluster index

            '''
            Step 5. run kmeans on outliers, create CS and RS from the result, keep only the summary of CS
            '''
            outliers_points_result, _ = kmeans(large_num_centroids, outlier_points_list, max_iterations, kmeans_initial_cent)
            outliers_cluster_result = groupPointsByCluster(large_num_centroids, outliers_points_result)
            CS_list, RS_list = list(), list()
            for outliers_cluster in outliers_cluster_result:
                if len(outliers_cluster) > 1:
                    CS_list.append(outliers_cluster)
                else:
                    RS_list.append(outliers_cluster)
            RS_list = [point[0] for cluster in RS_list for point in cluster]
            CS_summary = summarize(CS_list)
            del CS_list, outliers_cluster_result, outlier_points_list

            '''
            Step 6,7,8,9,10. Load the remaining data of the first round, assign each of them to DS, CS or RS
            '''
            for point in comp_points_list:
                min_DS_MD = float('inf')
                min_CS_MD = float('inf')
                found_qualified_DS_point = False
                for curr_DS_idx, DS_cluster in enumerate(DS_summary):
                    if DS_cluster['N'] != 0:
                        curr_DS_MD = mahalanobisDist(point, DS_cluster)
                        if curr_DS_MD < alpha_threshold * math.sqrt(len(point)):
                            if curr_DS_MD < min_DS_MD:
                                min_DS_MD = curr_DS_MD
                                DS_idx = curr_DS_idx
                                found_qualified_DS_point = True
                if found_qualified_DS_point:
                    updateSummary(DS_summary, DS_idx, point)
                else:
                    found_qualified_CS_point = False
                    for curr_CS_idx, CS_cluster in enumerate(CS_summary):
                        curr_CS_MD = mahalanobisDist(point, CS_cluster)
                        if curr_CS_MD < alpha_threshold * math.sqrt(len(point)):
                            if curr_CS_MD < min_CS_MD:
                                min_CS_MD = curr_CS_MD
                                CS_idx = curr_CS_idx
                                found_qualified_CS_point = True
                    if found_qualified_CS_point:
                        updateSummary(CS_summary, CS_idx, point)
                    else:
                        RS_list.append(point)
        elif generateCSRS == 'complement':
            #generate DS
            subset_points_result, _ = kmeans(k, subset_points_list, max_iterations, kmeans_initial_cent)

            # put together the points that have been clustered into a same cluster 
            subset_points_cluster_result = groupPointsByCluster(k, subset_points_result)
            DS_summary = summarize(subset_points_cluster_result)
            # discard the points
            del subset_points_result 
            del subset_points_cluster_result
            
            #generate CS/RS
            comp_points_result, _ = kmeans(large_num_centroids, comp_points_list, max_iterations, kmeans_initial_cent)
            comp_points_cluster_result = groupPointsByCluster(large_num_centroids, comp_points_result)
            
            CS_list = list()
            RS_list = list()
            CS_summary = list()
            for cluster in comp_points_cluster_result:
                if len(cluster) > 1:
                    CS_list.append(cluster)
                elif len(cluster) == 1:
                    RS_list.append(cluster)
            CS_summary = summarize(CS_list)
            RS_list = [point[0] for cluster in RS_list for point in cluster]
            del comp_points_result, comp_points_cluster_result, CS_list
            
    
    else: # not first round, Repeat step 6 to 12.    
        '''
        Step 8,9,10. Assign each of the data points to DS, CS or RS
        '''
        for point in points_list: # go through each point
            min_DS_MD = float('inf')
            min_CS_MD = float('inf')
            found_qualified_DS_point = False
            for curr_DS_idx, DS_cluster in enumerate(DS_summary): # go through each DS cluster
                curr_DS_MD = mahalanobisDist(point, DS_cluster)
                if curr_DS_MD < alpha_threshold * math.sqrt(len(point)):
                    if curr_DS_MD < min_DS_MD: 
                        min_DS_MD = curr_DS_MD
                        DS_idx = curr_DS_idx
                        found_qualified_DS_point = True
            if found_qualified_DS_point:
                updateSummary(DS_summary, DS_idx, point)
            else:
                if len(CS_summary) > 0:
                    found_qualified_CS_point = False
                    for curr_CS_idx, CS_cluster in enumerate(CS_summary): # go through each CS cluster
                        curr_CS_MD = mahalanobisDist(point, CS_cluster)
                        if curr_CS_MD < alpha_threshold * math.sqrt(len(point)):
                            if curr_CS_MD < min_CS_MD:
                                min_CS_MD = curr_CS_MD
                                CS_idx = curr_CS_idx
                                found_qualified_CS_point = True
                    if found_qualified_CS_point:
                        updateSummary(CS_summary, CS_idx, point)
                    else:
                        RS_list.append(point)
                else:
                    RS_list.append(point)

    # if CS_summary is empty, run this
    if len(CS_summary) == 0:
        '''
        Step 11. Run kmeans on current RS list, split clusters into CS and RS. 
        generate new CS summary and add it to CS summary, keep remaining RS list
        '''
        RS_points_result, _ = kmeans(large_num_centroids, RS_list, max_iterations, kmeans_initial_cent)
        RS_cluster_result = groupPointsByCluster(large_num_centroids, RS_points_result)

        RS_list = list() # delete the old RS_list
        for RS_cluster in RS_cluster_result:
            if len(RS_cluster) > 1:
                new_CS_summary = summarize([RS_cluster])[0] # only one CS cluster in the list
                CS_summary.append(new_CS_summary)
            elif len(RS_cluster) == 1: # RS_cluster might be empty when RS has less points than centroids
                RS_list.append(RS_cluster[0][0]) # keep only the points coordinates

        del RS_points_result # remove points
        del RS_cluster_result

    '''
    Step 12. Merge CS clusters that are close enough. The nearness is calculated based on the average of 
    Mahalanobis Distance of one cluster centroid to another CS cluster
    '''
    if merge_CS:
        CS_summary = mergeCS(CS_summary)

    '''
    Last round, merge CS clusters into closest DS cluster
    '''
    if round_idx == len(sorted(csv_files_list)) - 1: # 
        if len(CS_summary) > 0:
            for CS in CS_summary:
                min_ED_CSDS = float('inf')
                target_DS_idx = 0
                CS_cent = tuple([d/CS['N'] for d in CS['SUM']])
                for DS_idx, DS in enumerate(DS_summary): # go through each DS, found the cloest one
                    DS_cent = tuple([d/DS['N'] for d in DS['SUM']])
                    curr_ED_CSDS = euclideanDist(CS_cent, DS_cent)
                    if curr_ED_CSDS < min_ED_CSDS:
                        min_ED_CSDS = curr_ED_CSDS
                        target_DS_idx = DS_idx
                # merge current CS to DS with min_ED_CSDS 
                DS_summary[target_DS_idx]['N'] = DS_summary[target_DS_idx]['N'] + CS['N']
                new_SUM = tuple([sum(zipped) for zipped in zip(DS_summary[target_DS_idx]['SUM'], CS['SUM'])])
                new_SUMSQ = tuple([sum(zipped) for zipped in zip(DS_summary[target_DS_idx]['SUMSQ'], CS['SUMSQ'])])
                DS_summary[target_DS_idx]['SUM'] = new_SUM
                DS_summary[target_DS_idx]['SUMSQ'] = new_SUMSQ
            CS_summary = []

        
        '''
        for RS_point in RS_list:
            min_ED_RSDS = float('inf')
            target_DS_idx = 0
            for DS_idx, DS in enumerate(DS_summary): # go through each DS, found the cloest one
                DS_cent = tuple([d/DS['N'] for d in DS['SUM']])
                curr_ED_RSDS = euclideanDist(RS_point, DS_cent)
                if curr_ED_RSDS < min_ED_RSDS:
                    min_ED_RSDS = curr_ED_RSDS
                    target_DS_idx = DS_idx
            # merge current CS to DS with min_ED_CSDS 
            updateSummary(DS_summary, target_DS_idx, RS_point) 
        RS_list = []
        '''
    
    '''
    Step 13. Output the intermediate result
    By now, we have DS_summary, CS_summary and RS_list
    '''
    intermediate_result = generateIntermediateResult(round_idx, DS_summary, CS_summary, RS_list)
    intermediate_result_list.append(intermediate_result)
del CS_summary # all points info are now in the DS_summary

'''
Assign each point to corresponding DS cluster 
'''
result_dict = dict()
if cache_datapoints:
    for idx_point_pair in all_datapoints:
        min_ED = float('inf')
        pt_idx = str(idx_point_pair[0])
        point = idx_point_pair[1]
        DS_idx = 0
        if point in RS_list:
            result_dict[pt_idx] = -1
        else:
            for curr_DS_idx, DS in enumerate(DS_summary):
                DS_cent = tuple([d/DS['N'] for d in DS['SUM']])
                curr_ED = euclideanDist(point, DS_cent)
                if curr_ED < min_ED:
                    min_ED = curr_ED
                    DS_idx = curr_DS_idx
            result_dict[pt_idx] = DS_idx
    del RS_list
else:
    for csv in sorted(csv_files_list):
        full_csv_path = input_folder_path + "\\" + csv
        points_list = (sc.textFile(full_csv_path)  # read a data chunk
                        .map(lambda line:line.split(',')) # split the raw data string by comma                         
                        .map(lambda pt:[float(coord) for coord in pt]) # cast the str coordinates to float
                        .map(lambda pt:(str(int(pt[0])), tuple(pt[1:]))) # form tuple of idx and coordinates
                        .collect()
                       )
        for idx_point_pair in points_list:
            min_ED = float('inf')
            pt_idx = str(idx_point_pair[0])
            point = idx_point_pair[1]
            DS_idx = 0
            for curr_DS_idx, DS in enumerate(DS_summary):
                DS_cent = tuple([d/DS['N'] for d in DS['SUM']])
                curr_ED = euclideanDist(point, DS_cent)
                if curr_ED < min_ED:
                    min_ED = curr_ED
                    DS_idx = curr_DS_idx
            result_dict[pt_idx] = DS_idx

output_cluster = open(output_path_cluster, 'w')
output_cluster.write(json.dumps(result_dict))
output_cluster.close()

output_inter = open(output_path_intermediate_result, 'w')
header = 'round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained\n'
output_inter.write(header)
for intermediate_result in intermediate_result_list:
    output_inter.write(intermediate_result + '\n')
output_inter.close()

print('Duration:', str(time.time()-startTime))



