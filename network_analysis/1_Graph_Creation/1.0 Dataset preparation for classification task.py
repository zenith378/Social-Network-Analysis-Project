#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

import itertools
import networkx as nx


# ## Dataset "porti_20_22" - utenti contro

# In[2]:


df_c = pd.read_csv("../../data_collection/data/porti_18_22.csv", sep=',')


# In[3]:


df_c.shape


# In[4]:


df_c = df_c.drop(df_c[df_c.Language != "it"].index)
df_c['tweet_label'] = 0


# In[5]:


contro = []
for i in df_c.index:
    if df_c['User'][i] not in contro:
        contro.append(df_c['User'][i])


# ## Dataset scapp_20_22 - utenti pro

# In[6]:


df_p = pd.read_csv("../../data_collection/data/scapp_18_22.csv", sep=',')


# In[7]:


df_p = df_p.drop(df_p[df_p.Language != "it"].index)
df_p['tweet_label'] = 1


# In[8]:


df_p.shape


# In[9]:


pro = []
for i in df_p.index:
    if df_p['User'][i] not in pro:
        pro.append(df_p['User'][i])


# In[10]:


df_conc2= pd.concat([df_p, df_c])


# In[11]:


df_conc2.to_csv("../../data_collection/data/dataset_conc_iniziale.csv", index=False)


# ## Dataset tweet generali

# In[12]:


df = pd.read_csv("../../data_collection/data/df_def.csv", sep=',')


# In[13]:


df.info()


# In[14]:


df = df.drop(df[df.Language != "it"].index)


# In[15]:


df.shape


# In[16]:


df['tweet_label'] = -100


# In[17]:


df['Hashtags'].fillna("None", inplace=True)
df['In_reply_To'].fillna(df['User'], inplace=True)


# In[18]:


df['mention'] = (df.Tweet.str.findall(r'(?<![@\w])@(\w{1,25})').apply(','.join))


# ## Tweet context
# 
# ### Ucrain

# In[20]:


import re
import numpy as np

ukr = df[df['Tweet'].str.contains(r'ucrain', flags = re.IGNORECASE)]

ukr.reset_index(inplace = True)

print("\n dataset shape: ", ukr.shape)

print("\n Number of unique conversation ID: ", ukr['Conversation_Id'].nunique())

#New column in orginal dataframe
df['ukr'] = np.where(df['Tweet'].str.contains(r'ucrain', case=False, na=False, flags = re.IGNORECASE), 1, 0)


# ### ucraina/o & immigra-zione immigra-to/ta/te/ti

# In[21]:


ukr_imm = ukr[ukr['Tweet'].str.contains(r'immigra', flags = re.IGNORECASE)]

ukr_imm.reset_index(inplace = True)

print("\n dataset shape: ", ukr_imm.shape)
print("\n Number of unique conversation ID: ", ukr_imm['Conversation_Id'].nunique())


#New column
df['imm'] = np.where(df['Tweet'].str.contains(r'immigra', flags = re.IGNORECASE), 1, 0)


# ### ucraina/o & profugo/a/hi/he

# In[22]:


ukr_pro = ukr[ukr['Tweet'].str.contains(r'profug', flags = re.IGNORECASE)]

ukr_pro.reset_index(inplace = True)

print("\n dataset shape: ", ukr_pro.shape)
print("\n Number of unique conversation ID: ", ukr_pro['Conversation_Id'].nunique())


#New column
df['prof'] = np.where(df['Tweet'].str.contains(r'profug', flags = re.IGNORECASE), 1, 0)


# ### ucraina/o & migrant/e/i

# In[25]:


ukr_mig = ukr[ukr['Tweet'].str.contains(r'migrant', flags = re.IGNORECASE)]
ukr_mig.reset_index(inplace = True)

print("\n dataset shape: ", ukr_mig.shape)
print("\n Number of unique conversation ID: ", ukr_mig['Conversation_Id'].nunique())

#new column 
df['migr'] = np.where(df['Tweet'].str.contains(r'migrant', flags = re.IGNORECASE), 1, 0)


# ### ucraina/o & profughi veri

# In[26]:


ukr_pv = ukr[ukr['Tweet'].str.contains(r"profughi veri", flags = re.IGNORECASE)]
ukr_pv.reset_index(inplace = True)

print("\n dataset shape: ", ukr_pv.shape)
print("\n Number of unique conversation ID: ", ukr_pv['Conversation_Id'].nunique())

#new column 
df['prof_v'] = np.where(df['Tweet'].str.contains(r'profughi veri', flags = re.IGNORECASE), 1, 0)


# In[ ]:





# In[27]:


df[df['User']=="matteosalvinimi"]


# In[28]:


df.to_csv("../../data_collection/data/dataset_all.csv", index=False)


# In[ ]:




