#!/usr/bin/env python
# coding: utf-8

# In[7]:


from pyforest import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.datasets import load_iris


# In[8]:


data= load_iris()


# In[16]:


data.feature_names


# In[18]:


df= pd.DataFrame(data.data)


# In[19]:


df.head()


# In[20]:


df.columns= data.feature_names


# In[21]:


df.head()


# In[23]:


data.target


# In[24]:


df["target"]= data.target


# In[25]:


df.head()


# In[26]:


data.target_names


# <h6> here we can see that the target 0 is setosa, 1 is versicolor and 2 is virginica </h6>

# In[29]:


X= df.drop("target", axis=1)
y= df.target


# In[30]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42)


# In[31]:


model= SVC()


# In[32]:


model.fit(X_train,y_train)


# In[34]:


y_pred= model.predict(X_test)


# In[38]:


joblib.dump(model, 'model.pkl')


# In[39]:


c= [3.4,3,4,5]


# In[57]:


d= np.array(c)


# In[47]:


model.predict([c])


# In[48]:


from_joblib = joblib.load('model.pkl')


# In[58]:


from_joblib.predict([d])

