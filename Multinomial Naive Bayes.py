#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
data.target_names


# In[2]:


# the above are categories assigned so now we need to define the all categories
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)


# In[3]:


#printing out the training data
print(train.data[5])


# In[10]:


#how many articles are in trainign DATA so here in above we look though article no.5
print(len(train.data))


# ## Model based on Multinomial Naive Bayes 

# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
#TF-IDF is an abbreviation for Term Frequency Inverse Document Frequency. 
#This is very common algorithm to transform text into a meaningful 
#representation of numbers which is used to fit machine algorithm for prediction.
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())


# In[9]:


model.fit(train.data, train.target)
labels = model.predict(test.data)


# ## Creating confusion_matrix and heat_map

# In[7]:


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
xticklabels=train.target_names, yticklabels=train.target_names)
#Plotting the heatmap of confusion amtrix
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[11]:


# based on the trained model predicting category on new data
# s as string and pypline is model, so it is going to push any string through the model pipline 2
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# In[19]:


predict_category('President of Germany')


# In[13]:


predict_category('Sending load to International Space Station ISS')


# In[14]:


predict_category('Suzuki Hayabusa is a very fast motorcycle')


# In[15]:


predict_category('Audi is better than BMW')


# In[ ]:


# able to correctly classify texts into different group
#based on which category they belong to using Naive Bayes Classifier

