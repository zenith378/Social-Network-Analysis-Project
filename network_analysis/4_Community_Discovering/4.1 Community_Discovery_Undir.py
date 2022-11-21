#!/usr/bin/env python
# coding: utf-8

# # Community discovery con grafo non direzionato pesato ed unica componente connessa gigante

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
from collections import Counter
import numpy as np
warnings.filterwarnings('ignore')
import networkx as nx
import pandas as pd
from cdlib import algorithms, ensemble, evaluation, viz, NodeClustering
from networkx.generators.community import LFR_benchmark_graph
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


# In[2]:


def clustering_evaluation(method):
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
    
def clustering_comparator_AMI(method, list_of_methods): #Adjusted_Mutual_Information
    for methods in list_of_methods:
        eval = evaluation.normalized_mutual_information(method, methods)
        print('Comparison between ' + str(method.method_name) +' and ' + str(methods.method_name) + ' using Adjusted Mutual Information \n' + str(eval) +'\n')
    return

def clustering_comparator_ONMI(method, list_of_methods): #Overlapping_Normalized_Mutual_Information_LFK
    for methods in list_of_methods:
        eval = evaluation.overlapping_normalized_mutual_information_LFK(method, methods)
        print('Comparison between ' + str(method.method_name) +' and ' + str(methods.method_name) + ' using Overlapping Normalized Mutual Information LFK \n' + str(eval) +'\n')
    return

def modularity_vs_conductance(graph, lower_range, upper_range, algorithm):
    n_com = []
    condu = []
    modu = []
    for i in range(lower_range, upper_range):
        coms = algorithm(graph, i)
        n_com.append(i)
        condu.append(coms.conductance()[2])
        modu.append(coms.newman_girvan_modularity()[2])

    plt.plot(n_com, condu, '-g', label='conductance')
    plt.plot(n_com, modu, '-b', label='modularity')
    plt.legend();
    
    return


# In[3]:


G = nx.read_gexf("../../data_collection/data/RW_Graphs/SCC_weighted_graph_un.gexf")


# In[4]:


len(G.nodes())


# ## Community Discovery algorithms

# ### Label propagation

# In[5]:


lp_coms = algorithms.label_propagation(G)


# In[6]:


lp_coms.method_parameters # Clustering parameters


# ### Leiden

# https://www.nature.com/articles/s41598-019-41695-z

# In[7]:


leiden_coms = algorithms.leiden(G, weights = 'weight')


# In[8]:


leiden_coms.method_parameters # Clustering parameters


# ### Louvain

# In[9]:


resolution_mod = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
randomize_mod = ensemble.BoolParameter(name="randomize")

coms_mod, scoring_mod = ensemble.grid_search(graph=G, method=algorithms.louvain,
                                                     parameters=[resolution_mod, randomize_mod],
                                                     quality_score=evaluation.newman_girvan_modularity,
                                                     aggregate=max)

print("Configuration: %s \nScoring: %s" %(coms_mod.method_parameters, scoring_mod))


# In[9]:


louvain_coms = algorithms.louvain(G, weight= 'weight', resolution= 0.9, randomize=True)


# In[10]:


louvain_coms.method_parameters # Clustering parameters


# ### Demon

# In[13]:


eps = ensemble.Parameter(name="epsilon", start=0.1, end=1, step=0.1)
size = ensemble.Parameter(name="min_com_size", start=3, end=8, step=1)

coms_demon, scoring_demon = ensemble.random_search(graph=G, method = algorithms.demon,
                                                       parameters = [eps, size],
                                                       quality_score = evaluation.conductance,
                                                       aggregate = min)

print("Configuration: %s \nScoring: %s" %(coms_demon.method_parameters, scoring_demon))


# In[11]:


demon_coms = algorithms.demon(G, min_com_size = 3, epsilon = 0.1)


# In[12]:


demon_coms.method_parameters # Clustering parameters


# ### Angel

# In[17]:


threshold = ensemble.Parameter(name="threshold", start=0, end=1, step=0.1)
size_ang = ensemble.Parameter(name="min_community_size", start=3, end=8, step=1)

coms_angel, scoring_angel = ensemble.random_search(graph=G, method = algorithms.angel,
                                                       parameters = [threshold, size_ang],
                                                       quality_score = evaluation.conductance,
                                                        aggregate = min)

print("Configuration: %s \nScoring: %s" %(coms_angel.method_parameters, scoring_angel))


# In[13]:


angel_coms = algorithms.angel(G, min_community_size = 6, threshold = 0.1)


# In[14]:


angel_coms.method_parameters # Clustering parameters


# ### K-Cliques

# In[16]:


modularity_vs_conductance(G, 3, 5, algorithms.kclique)


# In[15]:


kclique_coms = algorithms.kclique(G, k=3)


# In[16]:


kclique_coms.method_parameters # Clustering parameters


# ### Principled Clustering

# In[19]:


modularity_vs_conductance(G, 2, 15, algorithms.principled_clustering)


# In[17]:


principled_coms = algorithms.principled_clustering(G, 4)


# In[18]:


principled_coms.method_parameters # Clustering parameters


# <a id="fitness"></a>
# ## Clustering Evaluation

# In[19]:


clustering_evaluation(lp_coms)


# In[20]:


clustering_evaluation(leiden_coms)


# In[21]:


clustering_evaluation(louvain_coms)


# In[22]:


clustering_evaluation(demon_coms)


# In[23]:


clustering_evaluation(angel_coms)


# In[24]:


clustering_evaluation(kclique_coms)


# In[25]:


clustering_evaluation(principled_coms)


# ## Clustering Evaluation: Valutazione delle somiglianze tra metodi diversi

# Ho deciso di utilizzare l' Adjusted Mutual Information perchè la MI soffre di bias nel momento in cui si hanno clustering con un numero elevato di cluster (come nel nostro caso). Infatti, anche se esse non condividono maggiore informazioni, è probabile avere comunque una misura che sovrastima la loro somiglianza.
# Per gli algoritmi che vanno a creare dei cluster con una copertura dei nodi del grafo non completa ho deciso invece di utilizzare l'overlapping normalized mutual information.
# 
# In ogni caso tale comparazioni soffrono il fatto che stiamo comparando algoritmi di famiglie diverse e che quindi utilizzano anche funzioni di qualità, o principi, diversi l'uno dall'altro nella costruzione delle comunità.

# In[26]:


clustering_comparator_AMI(lp_coms, [louvain_coms, leiden_coms, principled_coms])


# In[27]:


clustering_comparator_AMI(louvain_coms, [leiden_coms, principled_coms])


# In[ ]:


clustering_comparator_AMI(leiden_coms, [principled_coms])


# In[ ]:


clustering_comparator_ONMI(demon_coms, [angel_coms, kclique_coms])


# In[ ]:


clustering_comparator_ONMI(angel_coms, [kclique_coms])


# <a id="visualization"></a>
# ## Community/Statistics Visualization
# 

# ### Graph visualization
# 

# In[34]:


pos = nx.spring_layout(G)


# In[36]:


viz.plot_network_clusters(G, louvain_coms, pos, figsize=(30, 30), plot_labels=False)


# In[39]:


viz.plot_network_clusters(G, leiden_coms, pos, figsize=(30, 30), plot_labels=False)


# In[40]:


viz.plot_network_clusters(G, principled_coms, pos, figsize=(30, 30), plot_labels=False)


# In[41]:


viz.plot_community_graph(G, louvain_coms, figsize=(20, 20))


# In[42]:


viz.plot_community_graph(G, leiden_coms, figsize=(20, 20))


# In[43]:


viz.plot_community_graph(G, principled_coms, figsize=(20, 20))


# #### Community fitness/comparison visualization

# In[1]:


#viz.plot_com_stat([lp_coms, louvain_coms, leiden_coms, principled_coms], evaluation.conductance)


# In[45]:


viz.plot_com_stat([demon_coms, angel_coms, kclique_coms], evaluation.conductance)


# In[39]:


viz.plot_com_properties_relation( [lp_coms, louvain_coms, leiden_coms, principled_coms], evaluation.size, evaluation.conductance)


# In[41]:


viz.plot_com_properties_relation( [demon_coms, angel_coms, kclique_coms], evaluation.size, evaluation.conductance)


# In[44]:


viz.plot_sim_matrix([lp_coms, louvain_coms, leiden_coms, principled_coms],evaluation.adjusted_mutual_information)


# In[43]:


viz.plot_sim_matrix([demon_coms, angel_coms, kclique_coms],evaluation.overlapping_normalized_mutual_information_LFK)


# In[49]:


viz.plot_sim_matrix([demon_coms, angel_coms, kclique_coms],evaluation.overlapping_normalized_mutual_information_LFK)


# In[ ]:




