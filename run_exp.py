
# coding: utf-8

# In[1]:


from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

import numpy as np


# In[2]:


from adaboost_exp import *


# In[3]:


# Data path

"""three datasets: a9a, breast-cancer, gisette"""

data_dir = "/home/tati/Documents/data/"

a9 = data_dir + "a9/a9a.txt"
a9_test = data_dir + "a9/a9a.t"

breast = data_dir + "breast-cancer/breast-cancer.txt"

gisette = data_dir + "gisette/gisette_scale"
gisette_test = data_dir + "gisette/gisette_scale.t"


# In[4]:


# Load breast-cancer dataset and split into train/test by 0.8/0.2
breast = load_svmlight_file(breast, n_features=10)
breast_X, breast_X_test, breast_y, breast_y_test = train_test_split(
    breast[0], breast[1], train_size=0.8, test_size=0.2)


# In[5]:


clf = example_DV(breast_X, breast_y, breast_X_test, breast_y_test,
           dv_interval=100,
           n_estimators=100,
           dv_max_iter=200,
           dv_loss="log",
           dv_penalty="none")


# In[6]:


clf.estimator_weights_


# In[7]:


clf = example_BINARY(breast_X, breast_y, breast_X, breast_y,
           n_estimators=50)


# In[8]:


coef_init = np.array([clf.estimator_weights_])
coef_init.shape


# In[9]:


from sklearn.linear_model import LogisticRegression


# In[10]:


vote_model = LogisticRegression(fit_intercept=False,
                                max_iter=100,
                                penalty='l2',
                                C=1,
                                warm_start=True)
X_map = np.array([est.predict(breast_X) for est in clf.estimators_]).T
vote_model.coef_ = coef_init
vote_model.fit(X_map, breast_y)


# In[14]:


# clf.estimator_weights_ = vote_model.coef_[0]
(clf.predict(breast_X) != breast_y).sum() / len(breast_y)


# In[12]:


(vote_model.predict(X_map) != breast_y).sum() /  len(breast_y)


# In[13]:


log_model = LogisticRegression(fit_intercept=True,
                                max_iter=20,
                                penalty='l1',
                                C=10)
log_model.fit(breast_X, breast_y)
print(log_model.n_iter_)
(log_model.predict(breast_X) != breast_y).sum() /  len(breast_y)


# In[22]:


(clf.estimators_[0].predict(breast_X)== clf.n_classes_).T.shape


# In[29]:


pred = sum((estimator.predict(breast_X) == clf.classes_[:, np.newaxis]).T * w for estimator, w in
zip(clf.estimators_, clf.estimator_weights_)) / clf.estimator_weights_.sum()


# In[39]:


pred[:,0] *= -1


# In[45]:


pred = pred.sum(axis=1)


# In[48]:


clf.classes_.take(pred>0, axis=0)

