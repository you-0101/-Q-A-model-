#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


from matplotlib import pyplot as plt


# In[3]:


f1 = open(r"C:\Users\you10\OneDrive\바탕 화면\소종\2020_09_12_topic_model_공유\data\nq\nq-dev-nosub-dataset-ids.txt", 'r')
total = []
while True:
    line = f1.readline()
    if not line: break
    total.append(line)


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


f3 = open(r"C:\Users\you10\OneDrive\바탕 화면\소종\2020_09_12_topic_model_공유\비교데이터\bm25_than_bert.txt", 'rt', encoding='UTF8')
compdata = []
while True:
    line = f3.readline()
    if not line: break
    compdata.append(line)


# In[7]:


compnum = []
temp = []
a=0
for i in compdata:
    temp.append(compdata[a].split())
    compnum.append(temp[a][2])
    a = a+1


# In[8]:


compnum


# In[9]:


temp =[]
qurry_result =[]
for i in range(int(len(total)/2)):
    temp.append(total[i*2].split())
    for j in range(len(temp)):
        for k in range(len(temp[j])):
            qurry_result.append(int(temp[j][k]))
    temp =[]


# In[10]:


temp =[]
bm25_result =[]
for i in range(len(compnum)):
    a = int(compnum[i])*2
    temp.append(total[a].split())
    for j in range(len(temp)):
        for k in range(len(temp[j])):
            bm25_result.append(int(temp[j][k]))
    temp =[]


# In[11]:


temp =[]
bert_result =[]
for i in range(int(len(total)/2)):
    boola =1
    for t in range(len(compnum)):
        if i == int(compnum[t]):
            boola =0
    if boola ==1:
        temp.append(total[i*2].split())
        for j in range(len(temp)):
            for k in range(len(temp[j])):
                bert_result.append(int(temp[j][k]))
    temp =[]


# In[12]:


bert_result.sort()
bm25_result.sort()


# In[13]:


bert_idf =[]
for i in bert_result:
    bert_idf.append(float(idf[int(i)][1]))


# In[14]:


len(bert_idf)


# In[15]:


bm25_idf =[]
for i in bm25_result:
    bm25_idf.append(float(idf[int(i)][1]))


# In[16]:


len(bm25_idf)


# In[17]:


count = {}
for i in bm25_idf:
    try: count[i] += 1
    except: count[i]=1
print(count)


# In[18]:


bm25_idf


# In[19]:


def  zero(x):
      return x==0
def  one(x):
      return x<1
def  two(x):
      return x<2
def  three(x):
      return x<3
def  four(x):
      return x<4
def  five(x):
      return x<5
def  six(x):
      return x<6
def  seven(x):
      return x<7
def  eight(x):
      return x<8
def  nine(x):
      return x<9
def  ten(x):
      return x<10


# In[20]:


filteredList0 = list(filter(zero, bm25_idf))
filteredList1 = list(filter(one, bm25_idf))
filteredList2 = list(filter(two, bm25_idf))
filteredList3 = list(filter(three, bm25_idf))
filteredList4 = list(filter(four, bm25_idf))
filteredList5 = list(filter(five, bm25_idf))
filteredList6 = list(filter(six, bm25_idf))
filteredList7 = list(filter(seven, bm25_idf))
filteredList8 = list(filter(eight, bm25_idf))
filteredList9 = list(filter(nine, bm25_idf))
filteredList10 = list(filter(ten, bm25_idf))


# In[21]:


bm25_num = []
bm25_num.append(len(filteredList0))
bm25_num.append(len(filteredList1)-len(filteredList0))
bm25_num.append(len(filteredList2)-len(filteredList1))
bm25_num.append(len(filteredList3)-len(filteredList2))
bm25_num.append(len(filteredList4)-len(filteredList3))
bm25_num.append(len(filteredList5)-len(filteredList4))
bm25_num.append(len(filteredList6)-len(filteredList5))
bm25_num.append(len(filteredList7)-len(filteredList6))
bm25_num.append(len(filteredList8)-len(filteredList7))
bm25_num.append(len(filteredList9)-len(filteredList8))
bm25_num.append(len(filteredList10)-len(filteredList9))


# In[22]:


bm25_num


# In[23]:


filteredList0 = list(filter(zero, bert_idf))
filteredList1 = list(filter(one, bert_idf))
filteredList2 = list(filter(two, bert_idf))
filteredList3 = list(filter(three, bert_idf))
filteredList4 = list(filter(four, bert_idf))
filteredList5 = list(filter(five, bert_idf))
filteredList6 = list(filter(six, bert_idf))
filteredList7 = list(filter(seven, bert_idf))
filteredList8 = list(filter(eight, bert_idf))
filteredList9 = list(filter(nine, bert_idf))
filteredList10 = list(filter(ten, bert_idf))


# In[24]:


bert_num = []
bert_num.append(len(filteredList0))
bert_num.append(len(filteredList1)-len(filteredList0))
bert_num.append(len(filteredList2)-len(filteredList1))
bert_num.append(len(filteredList3)-len(filteredList2))
bert_num.append(len(filteredList4)-len(filteredList3))
bert_num.append(len(filteredList5)-len(filteredList4))
bert_num.append(len(filteredList6)-len(filteredList5))
bert_num.append(len(filteredList7)-len(filteredList6))
bert_num.append(len(filteredList8)-len(filteredList7))
bert_num.append(len(filteredList9)-len(filteredList8))
bert_num.append(len(filteredList10)-len(filteredList9))


# In[25]:


bert_percent = []
for i in range(len(bert_num)):
    bert_percent.append(bert_num[i]/len(bert_idf))
bm25_percent = []
for i in range(len(bm25_num)):
    bm25_percent.append(bm25_num[i]/len(bm25_idf))


# In[26]:


bert_percent


# In[27]:


bm25_percent


# In[28]:


index = [0,1,2,3,4,5,6,7,8,9,10]


# In[29]:


index1 = ['0','~1','~2','~3','~4','~5','~6','~7','~8','~9','~10']


# In[30]:


bar_width = 0.35
alpha = 0.5
x = np.arange(11)
plt.bar(x, bert_percent,bar_width,color='b',label='BERT')
plt.xticks(x, index1)
plt.bar(x+bar_width, bm25_percent,bar_width,color='r',label='BM25')
plt.xticks(x, index1)

plt.title('Percent of token(Idf)', fontsize=20)
plt.ylabel('Percent', fontsize=18)
plt.xlabel('Idf value', fontsize=18)

plt.legend(('BERT', 'BM25'), fontsize=15)

plt.show()


# In[31]:


x = np.arange(11)


# In[32]:


plt.plot(index,bert_percent,label ='bert')
plt.plot(index,bm25_percent,label ='bm25')
plt.legend(loc='upper left')


# In[33]:


plt.scatter(bm25_result,bm25_idf, alpha = 0.1)


# In[34]:


plt.scatter(bert_result,bert_idf, alpha = 0.1)


# In[ ]:




