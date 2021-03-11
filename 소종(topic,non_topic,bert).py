#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[1]:


f1 = open(r"C:\\Users\you10\OneDrive\바탕 화면\소종\2020_09_12_topic_model_공유\비교데이터\bert.txt", 'r')
bertdata = []
while True:
    line = f1.readline()
    if not line: break
    bertdata.append(line)


# In[3]:


f2 = open(r"C:\\Users\you10\OneDrive\바탕 화면\소종\2020_09_12_topic_model_공유\비교데이터\bm25.txt", 'r')
bm25data = []
while True:
    line = f2.readline()
    if not line: break
    bm25data.append(line)


# In[4]:


f3 = open(r"C:\\Users\you10\OneDrive\바탕 화면\소종\2020_09_12_topic_model_공유\비교데이터\not_topic_bm25.txt", 'r')
not_bertdata = []
while True:
    line = f3.readline()
    if not line: break
    not_bertdata.append(line)


# In[5]:


bertnum = []
a=0
for i in bertdata:
    b= bertdata[a].index(',')
    bertnum.append(bertdata[a][1:b])
    a = a+1


# In[6]:


bm25num = []
a=0
for i in bm25data:
    b= bm25data[a].index(',')
    bm25num.append(bm25data[a][1:b])
    a = a+1


# In[7]:


not_bertnum = []
a=0
for i in not_bertdata:
    b= not_bertdata[a].index(',')
    not_bertnum.append(not_bertdata[a][1:b])
    a = a+1


# In[8]:


len(bertnum)


# In[9]:


bertvsbm25 =[]
p=np.array(bertnum)
o=np.array(bm25num)
a=0
b=0
for i in bertnum:
    for j in bm25num:
        if p[a] == o[b] :
            bertvsbm25.append(bertnum[a])
        b=b+1
    b=0
    a=a+1


# In[10]:


len(bertvsbm25)


# In[11]:


total =[]
p=np.array(bertvsbm25)
o=np.array(not_bertnum)
a=0
b=0
for i in bertvsbm25:
    for j in not_bertnum:
        if p[a] == o[b] :
            total.append(bertvsbm25[a])
        b=b+1
    b=0
    a=a+1


# In[12]:


total


# In[18]:


x_1=[]
x=0
count=0
total=0
while x != 'q':
    x = input('Enter a name (q to quit): ')
    x_1.append(x)

for i in x_1:
    y = x_1[count].count('A')
    z = x_1[count].count('a')
    w = y+z
    total =total+w
    count = count+1
    
print('Appearance of letter \'a\':',total)


# In[ ]:




