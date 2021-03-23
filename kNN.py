import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv('kNN_dataset.csv')

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']]

y = df['custcat'].values

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train,  y_train)

yhat = neigh.predict(X_test)

print("Accuracy - Training Set : ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Accuracy - testing Set ", metrics.accuracy_score(y_test, yhat))