#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[ ]:


from matplotlib import pyplot as plt


# In[4]:


f1 = open(r"C:\Users\you10\OneDrive\바탕 화면\소종\2020_09_12_topic_model_공유\data\nq\nq-dev-nosub-vocab.txt", 'rt', encoding='UTF8')
worddata = []
while True:
    line = f1.readline()
    if not line: break
    worddata.append(line)


# In[5]:


word = []
a=0
for i in worddata:
    word.append(worddata[a].split())
    a = a+1


# In[6]:


word


# In[4]:


f2 = open(r"C:\Users\you10\OneDrive\바탕 화면\소종\2020_09_12_topic_model_공유\data\nq\nq-dev-nosub-idf.txt", 'rt', encoding='UTF8')
idfdata = []
while True:
    line = f2.readline()
    if not line: break
    idfdata.append(line)


# In[5]:


idf = []
a=0
for i in idfdata:
    idf.append(idfdata[a].split())
    a = a+1


# In[6]:


bm25_1 = "where do red ear slider turtles lay eggs what parts make up the peripheral nervous system what does it mean groundhog sees his shadow"


# In[7]:


bm25_2 = "103 163 348 7714 20328 3463 3708 3686 166 599 212 79 1 7346 4135 141 166 263 23 1654 20372 2532 29 4671"


# In[8]:


bert_1 = "where are the spore containing sori of a fern found when did india win their first cricket match is aluminium a ferrous or non ferrous metal"


# In[9]:


bert_2 = "103 26 1 2034 4 8 20499 172 55 142 258 453 36 41 4090 877 10 10119 8 31 9848 1541"


# In[10]:


list_bm25_1 = bm25_1.split()
list_bm25_2 = bm25_2.split()
list_bert_1 = bert_1.split()
list_bert_2 = bert_2.split()


# In[11]:


temp =[]
a=0
for i in  list_bm25_2:
    temp.append(int(list_bm25_2[a]))
    a=a+1
list_bm25_2=temp


# In[12]:


temp =[]
a=0
for i in  list_bert_2:
    temp.append(int(list_bert_2[a]))
    a=a+1
list_bert_2=temp


# In[13]:


list_bm25_1.sort()
list_bm25_2.sort()
list_bert_1.sort()
list_bert_2.sort()


# In[14]:


bm25_idf =[]
for i in list_bm25_2:
    bm25_idf.append(idf[int(i)][1])


# In[15]:


bert_idf =[]
for i in list_bert_2:
    bert_idf.append(idf[int(i)][1])


# In[21]:


plt.scatter(list_bm25_2,bm25_idf)


# In[22]:


plt.scatter(list_bert_2,bert_idf)


# In[ ]:




