#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/dropbox/Personal Notebooks'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[78]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
data.sex=le.fit_transform(data.sex)

data.smoker=le.fit_transform(data.smoker)

data_region=pd.get_dummies(data.region)

data=pd.concat([data,data_region],axis=1)
data=data.drop(['region'],axis=1)


data.head()


# In[83]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


f,ax=plt.subplots(figsize=(12,10))
corr=data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(230,10,as_cmap=True),
            square=False, ax=ax)


# In[86]:


plt.figure(figsize=(10,6))
ax = sns.scatterplot(x='bmi',y='charges',data=data,palette='magma',hue='smoker')
ax.set_title('Scatter plot of charges and bmi')

sns.lmplot(x="bmi", y="charges", hue="smoker", data=data, palette = 'magma', size = 8)


# In[89]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

### currently working on the model
