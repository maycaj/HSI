import scipy.io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import re

filepath = '/Users/maycaj/Documents/HSI_III/spectrums-11-15-24_EdemaTF.csv'


X = np.genfromtxt(filepath, delimiter=',')
X = X[1:,1:] 
y = pd.read_csv(filepath)
y = y.iloc[:,0]
# selectedIdx = np.random.randint(0,X.shape[0],size=20)
# X = X[selectedIdx, :]
# y = y[selectedIdx]

n_neighbors = 3
random_state = np.random.randint(0,4294967295) # generate random integer for random state

selectedNum = 3000
testSize = 1 - (selectedNum / X.shape[0])
X, _, y, _ = train_test_split(X, y, test_size=testSize, stratify=y, random_state=random_state)
y_int = [0 if item == 'EdemaFalse' else 1 for item in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, stratify=y, random_state=random_state)
n_examples = X_train.shape[0]

nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state))
knn = KNeighborsClassifier(n_neighbors=n_neighbors) # SVM and K-means also used commonly

nca.fit(X_train, y_train) # fit method's model
knn.fit(nca.transform(X_train), y_train) # compute the nearest neighbor classifier on the transformed training set
acc_knn = knn.score(nca.transform(X_test), y_test) # compute the nearest neighbor accuracy on the transformed test set=
knn.fit(nca.transform(X_train), y_train) # compute the nearest neighbor classifier on the transformed training set
plt.figure()
X_embedded = nca.transform(X) # transform (embed) the entire dataset 
scatter = plt.scatter(X_embedded[:,0],X_embedded[:,1],c=y_int,s=15,cmap="Set1", alpha=0.4, marker='.')     # (x, y, color, size, colormap) c=targets if we want a different color for each target
plt.title("n_train={}, KNN (k={}\nTest accuracy = {:.2f})".format(n_examples, n_neighbors, acc_knn))
plt.xlabel('Component 1')
plt.ylabel('Compoenent 2')
plt.show()

print('All done!')