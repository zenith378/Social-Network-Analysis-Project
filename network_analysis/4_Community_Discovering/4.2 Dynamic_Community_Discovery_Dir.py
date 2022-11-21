#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
from collections import Counter
import numpy as np
warnings.filterwarnings('ignore')
import networkx as nx
import pandas as pd
from cdlib import algorithms, ensemble, evaluation, viz, NodeClustering, TemporalClustering
from networkx.generators.community import LFR_benchmark_graph
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


# In[3]:


def w_graph_creation(csv):
    df = pd.read_csv(csv)
    
    #creo grafo
    G = nx.DiGraph()
    
    #creo nodi e info
    for i in df.index:
        if df['User'][i] not in G.nodes():
            G.add_node(df['User'][i], follower = (df['Follows_Count'][i]), 
                    friends = (df['Friends_Count'][i]), u_label = (df['user_label'][i]))
    
    #creo link pesati
    df_w = df.groupby(['User','In_reply_To'],as_index=False).size().sort_values('size',ascending=False)
    for i in df_w.index:
        G.add_edge(df_w['User'][i], df_w['In_reply_To'][i], weight= df_w['size'][i])
    
    return G

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

def modularity_vs_conductance(graph, algorithm):
    n_com = []
    condu = []
    modu = []
    for param in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        coms = algorithm(graph, weights='weight', resolution_parameter = param)
        n_com.append(param)
        condu.append(coms.conductance()[2])
        modu.append(coms.newman_girvan_modularity()[2])

    plt.plot(n_com, condu, '-g', label='conductance')
    plt.plot(n_com, modu, '-b', label='modularity')
    plt.legend();
    
    return


# Vado a crerae i semestri

# In[4]:


df_def= pd.read_csv("../../data_collection/data/df_analisi_sem.csv")


# In[5]:


df_def['Semester'] = np.nan


# In[6]:


for i in df_def.index:
    if df_def.Year[i] == 2019 or df_def.Year[i] == 2020:
        df_def['Semester'][i] = 1
    elif df_def.Year[i] == 2021 and (df_def.Month[i] == 1 or df_def.Month[i] == 2):
        df_def['Semester'][i] = 1
    elif df_def.Year[i] == 2021 and (df_def.Month[i] == 3 or df_def.Month[i] == 4 or
                                     df_def.Month[i] == 5 or df_def.Month[i] == 6 or
                                     df_def.Month[i] == 7 or df_def.Month[i] == 8):
        df_def['Semester'][i] = 2
    elif df_def.Year[i] == 2021 and (df_def.Month[i] == 9 or df_def.Month[i] == 10 or
                                     df_def.Month[i] == 11 or df_def.Month[i] == 12):
        df_def['Semester'][i] = 3
    elif df_def.Year[i] == 2022 and (df_def.Month[i] == 1 or df_def.Month[i] == 2):
        df_def['Semester'][i] = 3
    else:
        df_def['Semester'][i] = 4


# In[7]:


option=[2019, 2020, 2021]
df_S1 = df_def[(df_def['Year'].isin(option)) & 
          (df_def['Semester']== 1)]
df_S2 = df_def[(df_def['Year'] == 2021) & 
          (df_def['Semester'] == 2)]


# In[8]:


option=[2021, 2022]
df_S3 = df_def[(df_def['Year'].isin(option)) & 
          (df_def['Semester']== 3)]
df_S4 = df_def[(df_def['Year'] == 2022) & 
          (df_def['Semester'] == 4)]


# In[9]:


df_S1.reset_index(inplace=True)


# In[10]:


del(df_S1['index'])
df_S1


# In[11]:


df_S2.reset_index(inplace=True)
del(df_S2['index'])
df_S2


# In[12]:


df_S3.reset_index(inplace=True)
del(df_S3['index'])
df_S3


# In[13]:


df_S4.reset_index(inplace=True)
del(df_S4['index'])
df_S4


# ### Pulizia dei vari df semestrali in modo che contengano tutti i tweet delle conversazioni iniziate in quel semestre

# Creo set dei conversation ID per ogni semestre

# In[14]:


conv_id_S1 = set(df_S1.Conversation_Id)
conv_id_S2 = set(df_S2.Conversation_Id)
conv_id_S3 = set(df_S3.Conversation_Id)
conv_id_S4 = set(df_S4.Conversation_Id)


# Vado a calcolarmi i conversation ID comuni nei vari semestri

# In[15]:


conv_S1_S2 = []
for conv in conv_id_S1:
    if conv in conv_id_S2:
        conv_S1_S2.append(conv)
conv_S1_S2


# Creo pd dataframe dove copio i tweet che si riferiscono ad un conversation ID presente già nel semestre uno cercando negli altri semestri in base alle intersezioni calcolate prima

# In[16]:


df1 = df_S2[df_S2.Conversation_Id == 1365558136053133313].copy()
df2 = df_S2[df_S2.Conversation_Id == 1366012886016028673].copy()
df3 = df_S2[df_S2.Conversation_Id == 1365380873001041922].copy()
df4 = df_S2[df_S2.Conversation_Id == 1321392541447692288].copy()
df5 = df_S2[df_S2.Conversation_Id == 1301159186164854784].copy()
df6 = df_S2[df_S2.Conversation_Id == 1363752358761672705].copy()
df7 = df_S2[df_S2.Conversation_Id == 1363155385364922368].copy()
df8 = df_S2[df_S2.Conversation_Id == 1334572811306881029].copy()
df9 = df_S2[df_S2.Conversation_Id == 1335886083964497921].copy()
df10 = df_S2[df_S2.Conversation_Id == 1362683301845164032].copy()
df11 = df_S2[df_S2.Conversation_Id == 1145123250554511362].copy()
df12 = df_S2[df_S2.Conversation_Id == 1289998032864346113].copy()
df13 = df_S2[df_S2.Conversation_Id == 1310100391489679360].copy()
df14 = df_S2[df_S2.Conversation_Id == 1300447253115678731].copy()
df15 = df_S2[df_S2.Conversation_Id == 1301206982502694912].copy()


# elimino i tweet che si riferiscono ai conversation ID di interesse dai semstri successivi a quello in cui è partita la discussione

# In[17]:


for i in df_S2.index:
    if df_S2.Conversation_Id[i] in conv_S1_S2:
        df_S2 = df_S2.drop(labels=i, axis=0)


# In[18]:


conv_S1_S3 = []
for conv in conv_id_S1:
    if conv in conv_id_S3:
        conv_S1_S3.append(conv)
conv_S1_S3


# In[19]:


df16 = df_S3[df_S3.Conversation_Id == 1349440336859443201].copy()
df17 = df_S3[df_S3.Conversation_Id == 1289998032864346113].copy()


# In[20]:


for i in df_S3.index:
    if df_S3.Conversation_Id[i] in conv_S1_S3:
        df_S3 = df_S3.drop(labels=i, axis=0)


# In[21]:


conv_S1_S4 = []
for conv in conv_id_S1:
    if conv in conv_id_S4:
        conv_S1_S4.append(conv)
conv_S1_S4 


# In[22]:


df18 = df_S4[df_S4.Conversation_Id == 1313242941431918605].copy()
df19 = df_S4[df_S4.Conversation_Id == 1289998032864346113].copy()


# In[23]:


for i in df_S4.index:
    if df_S4.Conversation_Id[i] in conv_S1_S4:
        df_S4 = df_S4.drop(labels=i, axis=0)


# Alla fine faccio il merge e ripristino gli indici

# In[24]:


dfList = [df_S1, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19]
df_S1 = pd.concat(dfList)


# In[25]:


df_S1.reset_index(inplace=True)
del(df_S1['index'])


# In[26]:


df_S2.reset_index(inplace=True)
del(df_S2['index'])
df_S3.reset_index(inplace=True)
del(df_S3['index'])
df_S4.reset_index(inplace=True)
del(df_S4['index'])


# Solita procedura di prima ripetuta

# In[27]:


conv_S2_S3 = []
for conv in conv_id_S2:
    if conv in conv_id_S3:
        conv_S2_S3.append(conv)
conv_S2_S3


# In[28]:


df1 = df_S3[df_S3.Conversation_Id == 1432287234351304705].copy()
df2 = df_S3[df_S3.Conversation_Id == 1289998032864346113].copy()
df3 = df_S3[df_S3.Conversation_Id == 1432268270430789638].copy()
df4 = df_S3[df_S3.Conversation_Id == 1427553575354388481].copy()
df5 = df_S3[df_S3.Conversation_Id == 1426571792047476742].copy()
df6 = df_S3[df_S3.Conversation_Id == 1427019131711279112].copy()
df7 = df_S3[df_S3.Conversation_Id == 1371088876622778375].copy()
df8 = df_S3[df_S3.Conversation_Id == 1422472956257710102].copy()
df9 = df_S3[df_S3.Conversation_Id == 1430500213072953360].copy()
df10 = df_S3[df_S3.Conversation_Id == 1401161080890662912].copy()
df11 = df_S3[df_S3.Conversation_Id == 1395339299814584324].copy()
df12 = df_S3[df_S3.Conversation_Id == 1422193503820464130].copy()
df13 = df_S3[df_S3.Conversation_Id == 1427229382175776769].copy()
df14 = df_S3[df_S3.Conversation_Id == 1414003035227967489].copy()
df15 = df_S3[df_S3.Conversation_Id == 1413931119469289475].copy()
df16 = df_S3[df_S3.Conversation_Id == 1406547470163795974].copy()
df17 = df_S3[df_S3.Conversation_Id == 1383440571923005441].copy()
df18 = df_S3[df_S3.Conversation_Id == 1385553299554934786].copy()
df19 = df_S3[df_S3.Conversation_Id == 1410695956899667972].copy()


# In[29]:


for i in df_S3.index:
    if df_S3.Conversation_Id[i] in conv_S2_S3:
        df_S3 = df_S3.drop(labels=i, axis=0)


# In[30]:


conv_S2_S4 = []
for conv in conv_id_S2:
    if conv in conv_id_S4:
        conv_S2_S4.append(conv)
conv_S2_S4


# In[31]:


df20 = df_S4[df_S4.Conversation_Id == 1289998032864346113].copy()
df21 = df_S4[df_S4.Conversation_Id == 1426571792047476742].copy()
df22 = df_S4[df_S4.Conversation_Id == 1371088876622778375].copy()
df23 = df_S4[df_S4.Conversation_Id == 1405250465873416207].copy()


# In[32]:


for i in df_S4.index:
    if df_S4.Conversation_Id[i] in conv_S2_S4:
        df_S4 = df_S4.drop(labels=i, axis=0)


# In[33]:


dfList = [df_S2, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23]
df_S2 = pd.concat(dfList)


# In[34]:


df_S2.reset_index(inplace=True)
del(df_S2['index'])
df_S3.reset_index(inplace=True)
del(df_S3['index'])
df_S4.reset_index(inplace=True)
del(df_S4['index'])


# In[35]:


conv_S3_S4 = []
for conv in conv_id_S3:
    if conv in conv_id_S4:
        conv_S3_S4.append(conv)
conv_S3_S4


# In[36]:


df1 = df_S4[df_S4.Conversation_Id == 1496004135896125440].copy()
df2 = df_S4[df_S4.Conversation_Id == 1497314981225762820].copy()
df3 = df_S4[df_S4.Conversation_Id == 1498332266178793482].copy()
df4 = df_S4[df_S4.Conversation_Id == 1497167628900708441].copy()
df5 = df_S4[df_S4.Conversation_Id == 1464992345510842368].copy()
df6 = df_S4[df_S4.Conversation_Id == 1426571792047476742].copy()
df7 = df_S4[df_S4.Conversation_Id == 1371088876622778375].copy()
df8 = df_S4[df_S4.Conversation_Id == 1467385194575446019].copy()
df9 = df_S4[df_S4.Conversation_Id == 1495874870294585356].copy()
df10 = df_S4[df_S4.Conversation_Id == 1289998032864346113].copy()
df11 = df_S4[df_S4.Conversation_Id == 1475795835204120576].copy()
df12 = df_S4[df_S4.Conversation_Id == 1461376513878765574].copy()
df13 = df_S4[df_S4.Conversation_Id == 1490436401271394308].copy()
df14 = df_S4[df_S4.Conversation_Id == 1498281836342034432].copy()


# In[37]:


for i in df_S4.index:
    if df_S4.Conversation_Id[i] in conv_S3_S4:
        df_S4 = df_S4.drop(labels=i, axis=0)


# In[38]:


dfList = [df_S3, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14]
df_S3 = pd.concat(dfList)


# In[39]:


df_S3.reset_index(inplace=True)
del(df_S3['index'])
df_S4.reset_index(inplace=True)
del(df_S4['index'])


# In[41]:


df_S1.to_csv('df_S1.csv', sep=',', index=False)
df_S2.to_csv('df_S2.csv', sep=',', index=False)
df_S3.to_csv('df_S3.csv', sep=',', index=False)
df_S4.to_csv('df_S4.csv', sep=',', index=False)


# ### Creo i vari grafi

# In[42]:


G1 = w_graph_creation('../../data_collection/data/RW_Graphs/df_S1.csv')
G2 = w_graph_creation('../../data_collection/data/RW_Graphs/df_S2.csv')
G3 = w_graph_creation('../../data_collection/data/RW_Graphs/df_S3.csv')
G4 = w_graph_creation('../../data_collection/data/RW_Graphs/df_S4.csv')


# In[43]:


g = [G1, G2, G3, G4]
for graph in g:
    conn_comps = sorted(nx.strongly_connected_components(graph), key=len, reverse=True) 
    graph.remove_nodes_from([n for n in graph if n not in set(conn_comps[0])])


# ## Temporal Clustering

# In[44]:


modularity_vs_conductance(G1, algorithms.rb_pots)


# In[45]:


modularity_vs_conductance(G2, algorithms.rb_pots)


# In[46]:


modularity_vs_conductance(G3, algorithms.rb_pots)


# In[47]:


modularity_vs_conductance(G4, algorithms.rb_pots)


# Best selection resolution_parameter:       
# t1 = 0,8            
# t2 = 0,6       
# t3 = 0,6       
# t4 = 0,5    

# In[48]:


tc = TemporalClustering()
t = 1

for graph in g:
    resol_param = input("Give me resolution_parameter please: ")
    coms = algorithms.rb_pots(graph, weights='weight', resolution_parameter = float(resol_param))
    tc.add_clustering(coms, t)
    print(len(tc.get_clustering_at(t).communities))   
    t = t+1


# In[49]:


trend = tc.clustering_stability_trend(evaluation.nf1)
trend


# In[50]:


tc.has_explicit_match()


# In[51]:


jaccard = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))


# In[52]:


matches = tc.community_matching(jaccard, two_sided=True)


# In[53]:


f"Example of community match: {matches[0]}"


# In[54]:


com_polytree = tc.lifecycle_polytree(jaccard, two_sided=False)
print(com_polytree)


# In[55]:


weak_comp = list(nx.weakly_connected_components(com_polytree))[0]
x = com_polytree.copy()
x.remove_nodes_from([n for n in com_polytree.nodes() if n not in weak_comp])


# In[56]:


plt.figure(3,figsize=(12,12)) 
nx.draw(x, pos=nx.spring_layout(x), with_labels=True, node_color = "purple") 


# In[57]:


nx.info(x)


# In[58]:


for nodes in nx.weakly_connected_components(com_polytree):
    sub =  com_polytree.subgraph(nodes)    
    plt.figure(3,figsize=(12,12)) 
    nx.draw(sub, pos=nx.spring_layout(sub), with_labels=True) 


# In[59]:


tc.get_explicit_community_match()


# In[60]:


list(nx.weakly_connected_components(com_polytree))


# In[ ]:




