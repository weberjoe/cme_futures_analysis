# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:49:27 2019

@author: you_c
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import community

import matplotlib.pyplot as plt
import seaborn as sns; 
sns.set()


# this function has the returns dataset input. It calculates the cosine similarity between the variables
def cos_similarities(df):
    df_filtered_futures=df.copy() # creating a copy of the clean data to avoid codes below to change original dataset
    df_filtered_futures=df_filtered_futures.replace(np.inf, float("NaN")) # setting infinite values to nas
    df_filtered_futures= df_filtered_futures.replace(-np.inf, np.nan)
    df_filtered_futures=df_filtered_futures.fillna(0) # replacing all nas with 0s
    np.isinf(df_filtered_futures).sum().sum()
    similarities = pd.DataFrame(cosine_similarity(df_filtered_futures.transpose()))
    
    return similarities


# plotting cosine similarities
def plot_cos_similarity(similarities):
    similarities_plot=similarities.copy()
    similarities_plot=np.asarray(similarities_plot)
    similarities_plot=similarities_plot.ravel()
    #display(similarities_plot)
    plt.figure(figsize=(16,10))
    plt.hist(similarities_plot,bins=50,color='b')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    
# Louvain Algorithm with no threshold on similarity
def louvain_without_threshold(similarities):
    adjacency=similarities.values
    node_degrees = np.count_nonzero(adjacency, axis=1)      # this part of the function removes nodes ...
    nodes_to_keep = np.nonzero(node_degrees)[0]             # in the graph that are not connected to anything
    adjacency = adjacency[nodes_to_keep,:][:,nodes_to_keep]
    G=nx.from_numpy_matrix(adjacency)
    partition=community.best_partition(graph=G)#, random_state=1)
    """size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0.
    plt.figure(2,figsize=(10,8))
    for com in set(partition.values()):
        count += 1.
        list_nodes = [nodes for nodes in partition.keys()
                            if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()"""
    
# Louvain Algorithm - setting similarities less than 0.5 = 0
def louvain_with_threshold(similarities):
    threshold=0.5
    adjacency=similarities.values 
    adjacency[adjacency<threshold]=0
    node_degrees = np.count_nonzero(adjacency, axis=1)      # this part of the function removes nodes ...
    nodes_to_keep = np.nonzero(node_degrees)[0]             # in the graph that are not connected to anything
    adjacency = adjacency[nodes_to_keep,:][:,nodes_to_keep]
    G=nx.from_numpy_matrix(adjacency)
    partition=community.best_partition(graph=G)#, random_state=1)

    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0.
    plt.figure(2,figsize=(15,12))
    for com in set(partition.values()):
        count += 1.
        list_nodes = [nodes for nodes in partition.keys()
                            if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    
    
# This function's sole purpose is to feed the partition variable into the function "portfolio" in 
# the constructport file
def louvain_ret_part(similarities):
    threshold=0.5
    adjacency=similarities.values 
    adjacency[adjacency<threshold]=0
    node_degrees = np.count_nonzero(adjacency, axis=1)      # this part of the function removes nodes ...
    nodes_to_keep = np.nonzero(node_degrees)[0]             # in the graph that are not connected to anything
    adjacency = adjacency[nodes_to_keep,:][:,nodes_to_keep]
    G=nx.from_numpy_matrix(adjacency)
    partition=community.best_partition(graph=G) # random_state=1)
    return partition

