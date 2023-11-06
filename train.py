from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target variable

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

from joblib import dump

dump(knn, 'model.joblib')