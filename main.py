import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


data = pd.read_csv('crime.csv', delimiter=',')

n_samples = data['Count'].count()
print(n_samples)
outliers_fraction = 0.20
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

day = np.array(data['DATE'])
X = np.array(data['Count']).reshape(-1,1)

#Local Outlier Factor
clf = LocalOutlierFactor(n_neighbors=5, contamination=outliers_fraction)
pred = clf.fit_predict(X)
print('Local Outlier Factor Outliers: ', pred)

#One-Class SVM
oc_svm = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1).fit(X)
oc_svm_pred = oc_svm.predict(X)
print('One-Class SVM Outliers: ', oc_svm_pred)

#Robust Coveriance
rc = EllipticEnvelope(contamination=outliers_fraction)
rc_pred = rc.fit(X).predict(X)
print('Robust Coverience Outliers: ', rc_pred)

#Isolation Forest
is_fo = IsolationForest(contamination=outliers_fraction, random_state=42)
is_pred = is_fo.fit_predict(X)
print('Isolation Forest Outliers: ', is_pred)

# #Outlier days
# index = []
# for i in range(len(pred)):
#     if pred[i]!=1:
#         index.append(i)
# print('Outliers: ')
# total_outliers = 0
# for i in index:
#     print('DATE',day[i])
#     total_outliers+=1
# print('Total number of outliers: ',total_outliers)


