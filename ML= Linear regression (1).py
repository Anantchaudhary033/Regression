#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression


# In[2]:


df=pd.read_csv('Salary_Data.csv')
df


# In[3]:


# show top and bottom


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


# status summary


# In[7]:


df.describe()


# In[8]:


# distribution plot


# In[9]:


plt.title('salary Distribution plot')
sns.distplot(df['Salary'])
plt.show()


# In[10]:


# scatter plot


# In[11]:


plt.scatter(df['YearsExperience'],df['Salary'],color = 'lightcoral')
plt.title('salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False)
plt.show()


# In[12]:


# find out dependent and independent variable


# In[13]:


x = df.loc[:,['YearsExperience']]
x.head()


# In[14]:


y = df.loc[:,['Salary']]
y.head()


# In[15]:


# split the data set 


# In[16]:


x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.3,random_state=42)


# In[17]:


x_train.shape


# In[18]:


x_test.shape


# In[19]:


# train the dataset


# In[20]:


reg=LinearRegression()


# In[21]:


reg.fit(x_train,y_train)


# In[22]:


# predict


# In[23]:


y_pred_test = reg.predict(x_test)


# In[24]:


y_pred_train = reg.predict(x_train)


# In[25]:


y_pred_test


# In[26]:


y_pred_train


# In[27]:


# Predict


# In[28]:


reg.predict([[1.2]])


# In[29]:


# accuracy


# In[30]:


reg.score(x_test,y_test)


# In[31]:


reg.score(x_train,y_train)


# In[32]:


# validation(check by formula)


# In[33]:


intercept=reg.intercept_


# In[34]:


coefficient=reg.coef_


# In[35]:


#  y= mx+c


# In[41]:


def formula(x):
    salary= coefficient*x+intercept
    print (salary)
    
formula(1.2)


# In[37]:


# visualization(Best fit line)


# In[38]:


plt.scatter(x_train,y_train,color='lightcoral')
plt.plot(x_train,y_pred_train,color='firebrick')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['x_train/pred(y_test)','x_train/y_train'],title= 'Sal/Exp',loc='best')


# In[39]:


sns.lmplot(x='YearsExperience',y='Salary',data=df)


# In[ ]:




