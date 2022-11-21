#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import snscrape.modules.twitter as sntwitter
import time
import itertools


# In[2]:


df = pd.read_csv("./data/RW_Graphs/df.csv")


# In[4]:


df


# In[4]:


df.info()


# Numero di users diversi

# In[5]:


df["User"].nunique()


# Elaboriamo qualche statistica

# In[6]:


df.describe()


# Mi creo la colonna Year_Month per poter fare alcune analisi, come ad esempio il numero di tweet che ho in ogni mese

# In[3]:


df['Data_Tweet'] = pd.to_datetime(df['Date_Tweet']).dt.date
df.drop(["Date_Tweet"], axis=1, inplace=True)
df['Month'] = pd.to_datetime(df['Data_Tweet']).dt.month
df['Year'] = pd.to_datetime(df['Data_Tweet']).dt.year
df = df.sort_values(['Year','Month'], ascending=True)
df["Year_Month"] = df['Year'].apply(str) + "-" + df['Month'].apply(str)


# In[8]:


df


# Elimino tweet che allunga serie temporale senza particolare risultanza per l'analisi

# In[4]:


for i in df.index:
    if df['Conversation_Id'][i] == 859497031458488320:
        df = df.drop(labels=i, axis=0)


# In[5]:


df_count = df['Year_Month'].value_counts(sort=False).rename_axis('Year_Month').reset_index(name='Tweets_Counts')
df_count.head(60)


# Faccio plot time series

# In[7]:


df_count.plot(x='Year_Month', y='Tweets_Counts',figsize=(15, 6),xlabel="Time [year-month]", ylabel="Tweets Count");

plt.xticks(rotation=35,fontsize=14)
plt.savefig('./plots/tweets_time.pdf',format='pdf')


# In[11]:


sns.lineplot(data = df_count, x='Year_Month', y='Tweets_Counts')
sns.set_theme(rc={'figure.figsize':(50,10)}, font_scale=1.5)
plt.xticks(rotation=35)


# In[12]:


df = df.sort_values("Data_Tweet", ascending=False)


# In[13]:


#df.drop(["Year"], axis=1, inplace=True)
#df.drop(["Month"], axis=1, inplace=True)
#df.drop(["Year_Month"], axis=1, inplace=True)


# In[14]:


usertweets = df.groupby('User')

#Taking the top 25 tweeting users

top_users = usertweets.count()['Tweet'].sort_values(ascending = False)[:25]
top_users_dict = top_users.to_dict()
user_ordered_dict =sorted(top_users_dict.items(), key=lambda x:x[1])
user_ordered_dict = user_ordered_dict[::-1]
#Now, like in the previous hashtags and mention cases, going to make #two lists, one with the username and one with the value
dict_values = []
dict_keys = []
for item in user_ordered_dict[0:25]:
    dict_keys.append(item[0])
    dict_values.append(item[1])

fig = plt.figure(figsize = (15,15))
index = np.arange(25)
plt.bar(index, dict_values, edgecolor = 'black', linewidth=1)
plt.xlabel('Most active Users', fontsize = 18)
plt.ylabel('Nº of Tweets', fontsize=20)
plt.xticks(index,dict_keys, fontsize=15, rotation=90)
plt.title('Number of tweets for the most active users', fontsize = 20)
#plt.savefig('Tweets_of_active_users.jpg')
plt.show()


# Analizzando il grafo corrispondente al dataset su Gephi ci siamo accorti che mancavano alcuni tweet source che ci rendevano impossibile avere informazioni su utenti che si comportavano da hub nel nostro netwrok. Per questo andiamo a recuperare tali tweet

# In[15]:


lost_id=[]
for id in set(df.Conversation_Id):
    if id not in set(df.Tweet_Id):
        lost_id.append(id)


# In[16]:


#inizializzo lista
attributes_container = []
max_tweet = 100000

for j, id in enumerate(lost_id):
    min_id = id-1
#faccio scraping e uso enumerate per tenere conto dei tweet
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('since_id:'+ str(min_id) + ' max_id:' + str(id) +' (filter:safe OR -filter:safe').get_items()):
        #print(tweet.content)
        #print('')
        if i>max_tweet:
            break
        attributes_container.append([tweet.user.username,
                                 tweet.user.verified,
                                 tweet.user.created,
                                 tweet.user.followersCount, #persone che seguono l'utente
                                 tweet.user.friendsCount, #persone seguite dall'utente
                                 tweet.retweetCount,
                                 tweet.lang,
                                 tweet.date,
                                 tweet.likeCount,
                                 tweet.sourceLabel,
                                 tweet.id,
                                 tweet.content,
                                 tweet.hashtags,
                                 tweet.conversationId,
                                 tweet.inReplyToUser,
                                 tweet.coordinates,
                                 tweet.place])
    
#Creo df
lostest_id = pd.DataFrame(attributes_container, columns=["User",
                                                "verified",
                                                "Date_Created",
                                                "Follows_Count",
                                                "Friends_Count",
                                                "Retweet_Count",
                                                "Language",
                                                "Date_Tweet",
                                                "Number_of_Likes",
                                                "Source_of_Tweet",
                                                "Tweet_Id",
                                                "Tweet",
                                                "Hashtags",
                                                "Conversation_Id",
                                                "In_reply_To",
                                                "Coordinates",
                                                "Place"])


# In[17]:


lostest_id


# In[18]:


lostest_id['Data_Tweet'] = pd.to_datetime(lostest_id['Date_Tweet']).dt.date
lostest_id.drop(["Date_Tweet"], axis=1, inplace=True)
lostest_id['Month'] = pd.to_datetime(lostest_id['Data_Tweet']).dt.month
lostest_id['Year'] = pd.to_datetime(lostest_id['Data_Tweet']).dt.year
lostest_id = lostest_id.sort_values(['Year','Month'], ascending=True)
lostest_id["Year_Month"] = lostest_id['Year'].apply(str) + "-" + lostest_id['Month'].apply(str)


# In[19]:


lostest_id


# In[20]:


lostest_id.to_csv('lostest_id.csv', sep=',', index=False)


# In[38]:


df_def=pd.concat([df, lostest_id])


# In[39]:


df_def


# In[40]:


df_def.reset_index(inplace=True)
del(df_def['index'])
df_def


# In[25]:


#df_def.to_csv('df_def.csv', sep=',')


# Si va adesso ad eliminare quegli utenti di cui non abbiamo informazioni (visualizzati su gephi). Tale problema è dato da utenti i quali figurano in risposta ad alcuni tweet ma che non presentano il proprio tweet. Per prima cosa mi cerco gli username che compaiono in reply ma che non compaiono invece il User. Poi vado ad eliminare i None value e rimuovo il primo valore che è Nan. A quel punto mi cerco tutte le righe del dataset che contengono quei username in reply e gli elimino

# In[41]:


no_info = []
for user in set(df_def.In_reply_To):
    if user not in set(df_def.User):
        no_info.append(user)


# In[42]:


len(no_info)


# In[43]:


for i, val in enumerate(no_info):
    if val == None:
        print(i)


# In[44]:


no_info_new=[]
for i, elem in enumerate(no_info):
    if i == 0 or i == 850:
        continue
    else:
        no_info_new.append(elem)


# In[45]:


r_to_be_del = []
for i in df_def.index:
    if df_def['In_reply_To'][i] in no_info_new:
        r_to_be_del.append(i)
df_def.drop(labels=r_to_be_del, axis=0, inplace=True)


# In[46]:


df_def


# Controllo nuovamente il numero dei dati mancanti ed esso è nettamente minore.

# In[49]:


#df_def.to_csv('df_def.csv', sep=',', index=False)

