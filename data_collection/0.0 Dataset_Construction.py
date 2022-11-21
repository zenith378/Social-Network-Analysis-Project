#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tweepy
import configparser
import pandas as pd
import snscrape.modules.twitter as sntwitter
import time
import itertools


# In[2]:


def tweet_scraper(query, n_tweet):
    
    attributes_container = []
    max_tweet = n_tweet
    
    #faccio scraping e uso enumerate per tenere conto dei tweet
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):

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
    return pd.DataFrame(attributes_container, columns=["User",
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


# ## Ricerca hashtag singoli

# ### rifugiato/rifugiati

# In[ ]:


rif_20_22 = tweet_scraper('(#rifugiati OR #rifugiato) since:2020-09-01 until:2022-09-01', 500000)


# In[ ]:


rif_20_22


# In[ ]:


#rif_20_22.to_csv("rif_20_22.csv", sep=",", index=False)


# ### migrante/migranti

# In[ ]:


migr_20_22 = tweet_scraper('(#migranti OR #migrante) since:2020-09-01 until:2022-09-01', 500000)


# In[ ]:


migr_20_22 


# In[ ]:


#migr_20_22.to_csv("migr_20_22.csv", sep=",", index=False)


# ## profugo/profughi

# In[ ]:


prof_20_22 = tweet_scraper('(#profughi OR #profugo) since:2020-09-01 until:2022-09-01', 500000)


# In[ ]:


prof_20_22


# In[ ]:


#prof_20_22.to_csv("prof_20_22.csv", sep=",", index=False)


# ## Creo primo Dataset

# In[103]:


migranti = pd.read_csv("./data/migr_20_22.csv")
profughi = pd.read_csv("./data/prof_20_22.csv")
rifugiati = pd.read_csv("./data/rif_20_22.csv")


# In[104]:


df = pd.concat([migranti, profughi, rifugiati])


# In[105]:


df.reset_index(inplace=True)
df


# In[106]:


del(df['index'])
df


# In[107]:


df.info()


# In[108]:


df["User"].nunique()


# In[109]:


df['Conversation_Id'].nunique()


# In[110]:


df['Tweet'].nunique()


# Elimino tweet doppi soggeti a spam

# In[111]:


df.drop_duplicates(subset=['Tweet', 'User'], inplace=True)


# In[112]:


df["In_reply_To"] = df["In_reply_To"].str.replace("https://twitter.com/", "")


# In[113]:


df["In_reply_To"][10:20]


# Voglio prendere solo i tweet che sono in risposta a qualcuno ma non cancellando quelli source. Da verificare se il conversation ID del source esiste.

# In[114]:


df = df[df.duplicated(subset=['Conversation_Id'], keep=False)] 


# In[115]:


df


# In[116]:


df['Conversation_Id'].nunique()


# In[117]:


df['User'].nunique()


# In[118]:


df1 = df.drop_duplicates(subset='Conversation_Id')
df1.Conversation_Id


# ### Recupero tutte le conversazioni che riguardano i tweet trovati precedentemente

# In[ ]:


#inizializzo lista
attributes_container = []
max_tweet = 100000

for j, id in enumerate(df1.Conversation_Id):
#faccio scraping e uso enumerate per tenere conto dei tweet
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('conversation_id:' + str(id) +' (filter:safe OR -filter:safe)').get_items()):
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
conversation_20_22 = pd.DataFrame(attributes_container, columns=["User",
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


# In[ ]:


conversation_20_22


# In[ ]:


conversation_20_22.to_csv("./data/conversation_20_22.csv", sep=",", index=False)


# In[119]:


conversazioni = pd.read_csv("./data/conversation_20_22.csv")


# ## Creo secondo dataset con tutte le conversazioni

# In[120]:


df = pd.concat([df, conversazioni])


# In[121]:


df


# In[122]:


df.reset_index(inplace=True)
del(df['index'])
df


# In[123]:


df.info()


# In[124]:


df["In_reply_To"] = df["In_reply_To"].str.replace("https://twitter.com/", "")


# In[125]:


df.In_reply_To


# In[126]:


df.info()


# In[127]:


df['User'].nunique()


# In[128]:


df.groupby(['User', 'Tweet']).ngroups


# Tengo tweet unici (per evitare spam)

# In[129]:


df = df.drop_duplicates(['User', 'Tweet'])


# In[130]:


df


# Elimino tutti gli utenti che si rispondono da soli o che hanno scritto tweet in lingua non italiana

# In[31]:


for i in df.index:
    if df['User'][i] == df['In_reply_To'][i] or df['Language'][i] != 'it':
        df = df.drop(labels=i, axis=0)


# In[32]:


df


# In[33]:


df.reset_index(inplace=True)
del(df['index'])
df


# In[34]:


df['User'].nunique()


# In[24]:


df


# In[ ]:


#df.to_csv("df.csv", sep=",", index=False)

