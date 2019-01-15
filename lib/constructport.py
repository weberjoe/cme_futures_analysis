# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 17:56:13 2019

@author: you_c
"""

# The aim of this section is to construct our portfolio through dimension reduction - making use of Louvain's Algorithm output


# As a first step, the function below groups all futures belonging to the same communities together
def regroup_communities(similarities):
    import lib.clustering as cluster
    partition=cluster.louvain_ret_part(similarities)
    selections = dict()

    for futureindex, community in partition.items():

      if community not in selections.keys():
        selections[community] = list()


      selections[community].append(futureindex)
    return selections


"""To construct the portfolio, the following methodology was employed: 

- For each futures in a community, sum over the futures contract cosine similarities with each of the other futures contracts
- For each community, the futures contract that has the highest cosine similiarity with the other futures is the one that is 'most popular' and is therefore the one that is chosen to represent the respective community in the portfolio"""

def portfolio(selections,similarities,df): 
    import numpy as np
    import pandas as pd
    representatives = []
    df_filtered_futures=df.copy()
    adjacency=similarities.values
    node_degrees = np.count_nonzero(adjacency, axis=1)      # this part of the function removes nodes ...
    nodes_to_keep = np.nonzero(node_degrees)[0]             # ... in the graph that are not connected to anything
    adjacency = adjacency[nodes_to_keep,:][:,nodes_to_keep]
    selections=regroup_communities(similarities)
    for community, futureindices in selections.items():
      adjacency_matrix = adjacency[:, futureindices]       # subselection of adjacency with columns that I want
      s = adjacency_matrix.sum(axis=0)
      representatives.append(futureindices[np.argmax(s)])  # which index has the highest sum = argmax(s)

    portfolio=df_filtered_futures.iloc[:,representatives]
    portfolio=pd.DataFrame(portfolio)
    
    return portfolio

def port_ret_reps(selections,similarities,df): 
    import numpy as np
    import pandas as pd
    representatives = []
    df_filtered_futures=df.copy()
    adjacency=similarities.values
    node_degrees = np.count_nonzero(adjacency, axis=1)      # this part of the function removes nodes ...
    nodes_to_keep = np.nonzero(node_degrees)[0]             # ... in the graph that are not connected to anything
    adjacency = adjacency[nodes_to_keep,:][:,nodes_to_keep]
    selections=regroup_communities(similarities)
    for community, futureindices in selections.items():
      adjacency_matrix = adjacency[:, futureindices]       # subselection of adjacency with columns that I want
      s = adjacency_matrix.sum(axis=0)
      representatives.append(futureindices[np.argmax(s)])  # which index has the highest sum = argmax(s)

    return representatives
