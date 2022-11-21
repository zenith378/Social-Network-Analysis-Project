#!/usr/bin/env python
# coding: utf-8

# # Community discovery con grafo direzionato pesato ed unica componente connessa gigante

# In[1]:


import warnings
from collections import Counter
import numpy as np
warnings.filterwarnings('ignore')
import networkx as nx
import pandas as pd
from cdlib import algorithms, ensemble, evaluation, viz, NodeClustering
from networkx.generators.community import LFR_benchmark_graph


# In[2]:


def community_evaluation(method):
    print('Evaluating: ' + str(method.method_name))
    print("Node coverage: ", method.node_coverage)
    print("Overlapping: ", method.overlap)
    print("N_communities: ", len(method.communities))
    print("Number of nodes: ", method.size())
    print("Average internal degree: ", method.average_internal_degree())
    print("Internal edge density: ", method.internal_edge_density())
    print("Conductance: ", method.conductance())
    print("Cut ratio: ", method.cut_ratio())
    print("Newman Girvan modularity: ", method.newman_girvan_modularity())
    return
    
def community_comparator_AMI(method, list_of_methods):
    for methods in list_of_methods:
        eval = evaluation.normalized_mutual_information(method, methods)
        print('Comparison between ' + str(method.method_name) +' and ' + str(methods.method_name) + ' using Adjusted Mutual Information \n' + str(eval) +'\n')
    return


# In[5]:


G = nx.read_gexf("../../data_collection/data/RW_Graphs/SCC_weighted_graph_di.gexf")


# In[6]:


len(G)


# ## Community Discovery algorithms

# ### Rb_pots

# In[10]:


resolution_parameter = ensemble.Parameter(name="resolution_parameter", start=0.1, end=1, step=0.1)
w = ensemble.Parameter(name="weights", start='weight')
coms_rb, scoring_rb = ensemble.grid_search(graph=G, method = algorithms.rb_pots,
                                                       parameters = [resolution_parameter, w],
                                                       quality_score = evaluation.newman_girvan_modularity,
                                                       aggregate = max)

print("Configuration: %s \nScoring: %s" %(coms_rb.method_parameters, scoring_rb))


# In[11]:


rb_coms = algorithms.rb_pots(G, weights='weight', resolution_parameter=0.9)


# In[12]:


rb_coms.method_parameters # Clustering parameters


# ## Surprise Communities

# In[13]:


sc_coms = algorithms.surprise_communities(G, weights='weight')


# In[14]:


sc_coms.method_parameters # Clustering parameters


# <a id="fitness"></a>
# ## Clustering Evaluation

# In[16]:


community_evaluation(rb_coms)


# In[15]:


community_evaluation(sc_coms)


# <a id="comparison"></a>
# ## Clustering Evaluation: Valutazione delle somiglianze tra metodi diversi

# Ho deciso di utilizzare l' Adjusted Mutual Information perchè la MI soffre di bias nel momento in cui si hanno clustering con un numero elevato di cluster (come nel nostro caso). Infatti, anche se esse non condividono maggiore informazioni, è probabile avere comunque una misura che sovrastima la loro somiglianza.

# In[17]:


community_comparator_AMI(rb_coms, [sc_coms])


# In[18]:


viz.plot_sim_matrix([rb_coms, sc_coms],evaluation.adjusted_mutual_information)


# 
# ## Community/Statistics Visualization
# 

# ### Graph visualization

# In[19]:


from cdlib import viz

pos = nx.spring_layout(G)


# In[27]:


viz.plot_network_clusters(G, rb_coms, pos, figsize=(30, 30), plot_labels=False)


# In[26]:


viz.plot_network_clusters(G, sc_coms, pos, figsize=(30, 30), plot_labels=False)


# In[28]:


viz.plot_community_graph(G, rb_coms, figsize=(10, 10))


# In[29]:


viz.plot_community_graph(G, sc_coms, figsize=(10, 10))


# ## Community fitness/comparison visualization
# 

# In[30]:


viz.plot_com_stat([rb_coms, sc_coms], evaluation.conductance)


# In[31]:


viz.plot_com_stat([rb_coms, sc_coms], evaluation.internal_edge_density)


# In[32]:


viz.plot_com_properties_relation([rb_coms, sc_coms], evaluation.size, evaluation.conductance)
viz.plot_com_properties_relation([rb_coms, sc_coms], evaluation.size, evaluation.internal_edge_density)


# In[ ]:




