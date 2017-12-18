# coding: utf-8

# In[1]:

from time import time

from adaboost_impl import AdaBoostClassifier

# In[2]:


def example_SAMME_R(X, y, X_test, y_test, n_estimators=50):
    clf = AdaBoostClassifier(
            n_estimators=n_estimators,
            algorithm='SAMME.R')
    start_time = time()
    clf.fit(X, y)
    elapsed = time() - start_time
    print("SAMME.R time=%s" % elapsed)

    idx = 0
    for p in clf.staged_predict(X_test):
        error = (p != y_test).sum() / len(y_test)
        print("\t[Iter %s]: error=%s" % (idx, error))
        idx += 1

    return clf


# In[3]:


def example_SAMME(X, y, X_test, y_test, n_estimators=50):
    clf = AdaBoostClassifier(
            n_estimators=n_estimators,
            algorithm='SAMME')
    start_time = time()
    clf.fit(X, y)
    elapsed = time() - start_time
    print("SAMME time=%s" % elapsed)

    idx = 0
    for p in clf.staged_predict(X_test):
        error = (p != y_test).sum() / len(y_test)
        print("\t[Iter %s]: error=%s" % (idx, error))
        idx += 1

    return clf


# In[4]:


def example_BINARY(X, y, X_test, y_test, n_estimators=50):
    clf = AdaBoostClassifier(
            n_estimators=n_estimators,
            algorithm='BINARY')

    start_time = time()
    clf.fit(X, y)
    elapsed = time() - start_time
    print("BINARY time=%s" % elapsed)

    idx = 0
    for p in clf.staged_predict(X_test):
        error = (p != y_test).sum() / len(y_test)
        print("\t[Iter %s]: error=%s" % (idx, error))
        idx += 1

    return clf


# In[6]:


def example_DV(
    X,
    y,
    X_test,
    y_test,
    n_estimators=50,
    dv_interval=50,
    dv_solver=None
):
    clf = AdaBoostClassifier(
        n_estimators=n_estimators,
        algorithm='DYNAMIC_VOTES',
        dv_interval=dv_interval,
        dv_solver=dv_solver
    )
    start_time = time()
    clf.fit_test(X, y, X_test, y_test)
    elapsed = time() - start_time
    error = (clf.predict(X_test) != y_test).sum() / len(y_test)
    print("DV error=%s, time=%s" % (error, elapsed))

    return clf
