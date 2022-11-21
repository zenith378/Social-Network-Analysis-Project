#!/usr/bin/env python
# coding: utf-8

# # Tweet cleaning
# * Import libraries and load dataset
# * Clean tweet in preparation to classification task
# 

# In[1]:


#pip install demoji


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Text cleaning
import re, string
from stop_words import get_stop_words
import demoji

#nltk
import nltk
from nltk import ngrams
from nltk.stem import WordNetLemmatizer,SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator

#import spacy


# In[3]:


df = pd.read_csv('../../data_collection/data/dataset_conc_iniziale.csv')


# In[4]:


df = df.drop_duplicates()


# In[5]:


unique= df["tweet_label"].unique()
freq = df["tweet_label"].value_counts()
sns.set(font_scale = 1)

ax = sns.countplot(df["tweet_label"], 
                   order = df["tweet_label"].value_counts().index)
plt.title("Target variable counts in dataset")
plt.ylabel('Number of tweets')
plt.xlabel('Tweet Type')

# adding the text labels
rects = ax.patches
for rect, label in zip(rects, freq):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
plt.show()


# ## Tweet Cleaning

# In[6]:


#DETERMINE ALL STOP WORDS
stop_words = get_stop_words('italian')
#stp_wrd = ['','r', 're', 'rt', 'bc', 'etc']

#stop_words.extend(stp_wrd)

#Remove punctuation, links, stopwords, mentions and \r\n new line characters
def strip_all_entities(text):
    text = re.sub(r"twit", "tweet", text)
    text = re.sub(r"http\S+", "", text) #remove links
    text = re.sub(r"#\S+","", text) #remove hashtags
    text = re.sub(r"(@\w+)", "" , text) #remove mentions
    text = text.replace('/', ' ') #remove "/"
    text = text.replace('\r', '').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"[^\w\d'\s]+",' ',text) #remove punctuaction
#     banned_list= string.punctuation #remove punctuation
#     table = str.maketrans('', '', banned_list)
#     text = text.translate(table)
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)    
#     text =' '.join(word for word in text.split() if len(word) < 20) # remove words longer than 20 characters
    return text


#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the "#" symbol
def clean_hashtags(text):
    text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text)) #remove last hashtags
    text = " ".join(word.strip() for word in re.split('#|_', text)) #remove hashtags symbol from words in the middle of the sentence
    return text

#Filter special characters such as "&" and "$" present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

#Remove multiple sequential spaces
def remove_mult_spaces(text):
    return re.sub("\s\s+" , " ", text)

#Clean emoticons from text
def emoticon_remove(text):
    dem = demoji.findall(text)
    for i in dem.keys():
        dem[i]= re.sub(r' ', "_", dem[i])
        text =  text.replace(i, dem[i])
    return text


# In[7]:


#Then we apply all the defined functions in the following order
def preprocess(text):
    #text = re.sub(r"'", "_ ", text)
    text = strip_all_entities(text)
    text = filter_chars(text)
    text = re.sub(r'\W*\b\w{1,1}\b', '', text)
    text = remove_mult_spaces(text)
    text = emoticon_remove(text)
    return text


# In[8]:


df['text_clean'] = df['Tweet'].apply(lambda x: preprocess(x))


# In[9]:


#TOKENIZATION
def tokenization(text):
    text = re.split('\W+', text)
    return text

df['Tweet_tokenized'] = df['text_clean'].apply(lambda x: tokenization(x))
df['tokenized_bigram'] = df['Tweet_tokenized'].apply(lambda words: list(nltk.ngrams(words, 2)))


# In[10]:


df


# In[11]:


#WORDS FREQUENCY ALL
text = list()
for tweet in df['Tweet_tokenized']:
    for el in tweet:
        if el not in stop_words:
            text.append(el)
    
freq_dist = nltk.FreqDist(text)

freq_dist.most_common(20)


# In[12]:


#WORDS FREQUENCY LABEL 1-PRO
text = list()
subset_pro = df[df['tweet_label']==1]

for tweet in subset_pro.Tweet_tokenized:
    for el in tweet:
        if el not in stop_words:
            text.append(el)
    
freq_dist = nltk.FreqDist(text)

freq_dist.most_common(20)


# In[13]:


#WORDS FREQUENCY _ 1Gram e 2Gram

bigram_pro = list()
for tweet in subset_pro['Tweet_tokenized']:
    for i in range (0, len(tweet)-1):
        if (tweet[i] not in stop_words) & (tweet[i+1] not in stop_words):
            bigram_pro.append((tweet[i], tweet[i+1]))
    
big_freq_pro = nltk.FreqDist(bigram_pro)

big_freq_pro.most_common(20)


# In[14]:


#WORDS FREQUENCY LABEL -1 contro
text = list()
subset_contro = df[df['tweet_label']==0]

for tweet in subset_contro.Tweet_tokenized:
    for el in tweet:
        if el not in stop_words:
            text.append(el)
            
freq_dist = nltk.FreqDist(text)

freq_dist.most_common(20)


# In[15]:


#WORDS FREQUENCY _ 1Gram e 2Gram

bigram_contro = list()
for tweet in subset_contro['Tweet_tokenized']:
    for i in range (0, len(tweet)-1):
        if (tweet[i] not in stop_words) & (tweet[i+1] not in stop_words):
            bigram_contro.append((tweet[i], tweet[i+1]))
    
big_freq_contro = nltk.FreqDist(bigram_contro)

big_freq_contro.most_common(20)


# In[16]:


text_len = []
for text in df.text_clean:
    tweet_len = len(text.split())
    text_len.append(tweet_len)


# In[17]:


df['text_len'] = text_len


# In[18]:


df


# In[19]:


plt.figure(figsize=(20,5))
ax = sns.countplot(x=df['text_len'], hue=df['tweet_label'], palette='mako')
plt.title('Count of tweets with words distribution', fontsize=20)
plt.legend(['contro', 'pro'])
plt.yticks([1000, 2000, 4000], ['1000', '2000', '4000'],rotation=45)
plt.ylabel('count')
plt.xlabel('number of words')
plt.show()


# In[20]:


df.to_csv("../../data_collection/data/cleaned_tweet.csv")


# In[21]:


#da rivedere distribuzione tweet

