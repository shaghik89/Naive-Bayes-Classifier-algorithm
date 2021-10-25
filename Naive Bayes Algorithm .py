#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn import datasets
wine = datasets.load_wine()
print(wine)


# In[ ]:


# class 0, 1, 2 determines the three samples of wine


# In[6]:


# in order to know the features
print(wine.feature_names)


# In[7]:


print(wine.target_names)


# In[9]:


X=pd.DataFrame(wine['data'])
print(X.head())


# In[10]:


y= print(wine.target)


# ## Model & Training

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.30, random_state=100)


# In[16]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train, y_train)
y_predict=gnb.predict(X_test)
print(y_predict)


# ## Accuracy

# In[18]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_predict))


# In[29]:


from sklearn.metrics import confusion_matrix
confm=np.array(confusion_matrix(y_test, y_predict))

confm

