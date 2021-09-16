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

X = data.drop(['charges'], axis = 1)
Y = data.charges

quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)

plr = LinearRegression().fit(X_train,Y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)
orest = RandomForestRegressor(n_estimators = 100,
    criterion = 'mse',
    random_state = 1,
    n_jobs = -1)
forest.fit(x_train,y_train)
forest_train_pred = forest.predict(x_train)
forest_test_pred = forest.predict(x_test)

print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,forest_train_pred),
mean_squared_error(y_test,forest_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))

plt.figure(figsize=(10,6))

plt.scatter(forest_train_pred,forest_train_pred - y_train,
          c = 'black', marker = 'o', s = 35, alpha = 0.6,
          label = 'Trained')
plt.scatter(forest_test_pred,forest_test_pred - y_test,
          c = 'c', marker = 'o', s = 25, alpha = 0.9,
          label = 'Tested')
plt.xlabel('Predicted values')
plt.ylabel('Tailings')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 0, xmax = 4000, lw = 2, color = 'red')


from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

reg_features = ['cluster', 'age', 'smoker', 'bmi', 'children']

X_train = np.array(data_train[reg_features])
poly = PolynomialFeatures(3)
X_train = poly.fit_transform(X_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = np.array(data_train[['charges']])

X_cv = np.array(data_cv[reg_features])
X_cv = scaler.transform(poly.transform(X_cv))
y_cv = np.array(data_cv[['charges']])

X_test = np.array(data_test[reg_features])
X_test = scaler.transform(poly.transform(X_test))
y_test = np.array(data_test[['charges']])

scores = np.array([])
scores = scores.reshape(-1, 2)
for alpha in np.arange(0, 1, .1):
    reg = Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    scores = np.append(scores, np.array([[reg.score(X_cv, y_cv), alpha]]), axis=0)
    print('Alpha: ' + str(alpha) + '\tacc: ' + str(round(scores[-1, 0], 5)))

ind = np.argmax(scores, axis=0)[0]
alpha = scores[ind, 1]
reg = Ridge(alpha=alpha)
reg.fit(X_train, y_train)
print('\nChosen alpha: ' + str(alpha) + '\tfor accuracy: ' + str(reg.score(X_cv, y_cv)))


print('\nAccuracy of the model:')
print(reg.score(X_test, y_test))
