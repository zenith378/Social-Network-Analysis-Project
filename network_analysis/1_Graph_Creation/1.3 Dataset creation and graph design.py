#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt

import itertools
import networkx as nx
import datetime
import numpy as np


# In[2]:


df_class = pd.read_csv("../../data_collection/data/RW_Graphs/dataset_classified.csv", sep=',')


# In[3]:


del df_class['Unnamed: 0']


# In[29]:


df_class.shape


# In[30]:


# pro = []
# contro = []
# for i in df_class.index:
#     if (df_class['tweet_label'][i] == 0) & (df_class['User'][i] not in contro):
#         contro.append(df_class['User'][i])
#     elif (df_class['tweet_label'][i] == 1) & (df_class['User'][i] not in pro):
#         pro.append(df_class['User'][i])


# In[31]:


len(pro)


# In[32]:


len(contro)


# In[33]:


#numero utenti "neutri"
neutro = set(set(pro)&set(contro))
len(neutro)


# In[34]:


#numero utenti solo "pro"
pro = (set(set(pro)-set(contro)))
len(pro)


# In[35]:


#numero utenti solo "contro"
contro = set(set(contro)-set(pro))
len(contro)


# In[5]:


#Associate a continuous value to all users based on the number of positive tweets they have out of the total number of tweets 
d = dict(df_class.groupby(['User'])['tweet_label'].mean())


# In[48]:


d[df_class['User'][96341]]


# In[49]:


df_class[df_class['User']=='Agenzia_Ansa']


# In[6]:


df_class['user_label'] = None
for i in df_class.index:
    df_class['user_label'][i] = d[df_class['User'][i]]


# In[7]:


df = df_class.copy()


# In[51]:


df.isna().sum()


# In[52]:


df.shape


# # Directed Weighted Graph

# In[53]:


G = nx.DiGraph()


# ### Nodes

# In[54]:


for i in df.index:
    if df['User'][i] not in G.nodes():
        G.add_node(df['User'][i], follower = (df['Follows_Count'][i]), 
                    friends = (df['Friends_Count'][i]), u_label = (df['user_label'][i]))


# In[55]:


#G.nodes().data()


# ### Weighted Edges

# In[56]:


#weighted dataframe
df_w = df.groupby(['User','In_reply_To'],as_index=False).size().sort_values('size',ascending=False)


# In[57]:


#archi pesati 
for i in df_w.index:
    G.add_edge(df_w['User'][i], df_w['In_reply_To'][i], weight= df_w['size'][i])


# In[58]:


G.edges().data()


# In[59]:


nx.write_gexf(G, "../../data_collection/data/RW_Graphs/weighted_graph.gexf")


# In[60]:


#strongly connected component directed weighted graph
Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
G0 = G.subgraph(Gcc[0])


# In[61]:


len(G0.nodes())


# In[62]:


nx.write_gexf(G0, "../../data_collection/data/RW_Graphs/SCC_weighted_graph_di.gexf")


# In[63]:


#connected component und weighted graph
Gu = G.to_undirected()


# In[64]:


Gcc = sorted(nx.connected_components(Gu), key=len, reverse=True)
G0 = Gu.subgraph(Gcc[0])


# In[65]:


len(G0.nodes())


# In[66]:


nx.write_gexf(G0, "../../data_collection/data/RW_Graphs/SCC_weighted_graph_un.gexf")


# ### Edgelist weighted graph

# In[67]:


tupla = []
for i in df_w.index:
    tupla.append((df_w['User'][i], df_w["In_reply_To"][i], df_w['size'][i]))
        
df_def_und = pd.DataFrame(tupla, columns = ["source","target", "weight"])


# In[68]:


df_def_und.to_csv("../../data_collection/data/RW_Graphs/edgelist_extended.csv", index=False)


# # Multi Directed Graph

# In[8]:


MG = nx.MultiDiGraph()


# ### Nodes
# I nodi sono gli utenti, con attributi : 
# * numero di follower
# * numero di amici

# In[9]:


for i in df.index:
    if df['User'][i] not in MG.nodes():
        MG.add_node(df['User'][i], follower = (df['Follows_Count'][i]), 
                    friends = (df['Friends_Count'][i]), u_label = (df['user_label'][i]))


# In[71]:


#MG.nodes.data()


# ### Links
# I link sono le menzioni e retweet, con attributi: 
# * Utente
# * Utente a cui il tweet risponde (se esistente)
# * il testo del tweet
# * gli hastag contenuti
# * Il numero di like ottenuto
# * conversation_id

# In[4]:


def graph_creation(df): 
    MG = nx.MultiDiGraph()
    #archi non pesati
    for i in df.index:
        if df['User'][i] not in MG.nodes():
            MG.add_node(df['User'][i], follower = (df['Follows_Count'][i]), 
                    friends = (df['Friends_Count'][i]), u_label = (df['user_label'][i]))
            
            
        if df['In_reply_To'][i] == 'None':
            MG.add_edge(df['User'][i], df['User'][i], tweet = df['Tweet'][i], hashtag = df['Hashtags'][i],
                        likes = (df['Number_of_Likes'][i]), coversation_id = (df['Conversation_Id'][i]), 
                        mentions = (df['mention'][i]), tweet_label = (df['tweet_label'][i]))
        else:
            MG.add_edge(df['User'][i], df['In_reply_To'][i], tweet = df['Tweet'][i], hashtag = df['Hashtags'][i],
                        likes = (df['Number_of_Likes'][i]), coversation_id = (df['Conversation_Id'][i]),
                        mentions = df['mention'][i], tweet_label = (df['tweet_label'][i]))
    return MG


# In[5]:


split_date = datetime.date(2022,2,24)


# In[31]:


df_class['Year_Month'] = pd.to_datetime(df_class['Year_Month']) # simplify, does not require Py datetime

# use pd.Timestamps for comparisons
#start_date = pd.Timestamp('2022-02-24T01:00:00.000000+0100')

#mask_before = (df['Year_Month'] > split_date)

df_before = df_class.loc[df_class['Year_Month'].dt.date<split_date]

df_after = df_class.loc[df_class['Year_Month'].dt.date>=split_date]



# In[32]:


d = dict(df_before.groupby(['User'])['tweet_label'].mean())


# In[33]:


df_before['user_label'] = None
for i in df_before.index:
    df_before['user_label'][i] = d[df_before['User'][i]]


# In[16]:


d = dict(df_after.groupby(['User'])['tweet_label'].mean())


# In[17]:


df_after['user_label'] = None
for i in df_after.index:
    df_after['user_label'][i] = d[df_after['User'][i]]


# In[27]:


val=4*df_after.user_label
val2=val.astype(float)
df_after['user_label']=val2.round()/4


# In[34]:


val=4*df_before.user_label
val2=val.astype(float)
df_before['user_label']=val2.round()/4


# In[35]:


MG_before=graph_creation(df_before)


# In[52]:


MG_after=graph_creation(df_after)
MG=graph_creation(df)


# In[74]:


len(MG.nodes())


# In[75]:


len(MG.edges())


# In[76]:


df['Conversation_Id'].nunique()


# In[77]:


df['Number_of_Likes'].max()


# In[53]:


nx.write_gexf(MG, "../../data_collection/data/RW_Graphs/extended_graph.gexf")


# In[36]:


nx.write_gexf(MG_before, "../../data_collection/data/RW_Graphs/extended_graph_before.gexf")
nx.write_gexf(MG_after, "../../data_collection/data/RW_Graphs/extended_graph_after.gexf")


# ## Edge csv 

# In[79]:


tupla = []
for i in df.index:
    tupla.append((df['User'][i], df["In_reply_To"][i], df['Tweet'][i], df['Hashtags'][i], df['Number_of_Likes'][i],
                  df['Conversation_Id'][i], df['mention'][i], df['tweet_label'][i], df['user_label'][i]))
        
df_def = pd.DataFrame(tupla, columns = ["source","target", "tweet", "hashtag", "likes",
                                        "conv_id", "mentions", "tweet_label", "user_label"])


# In[80]:


df_def.to_csv("../../data_collection/data/RW_Graphs/edgelist_extended.csv", index=False)


# In[ ]:




