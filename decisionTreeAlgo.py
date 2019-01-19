import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv("E:/forth year project/final attribute table/final attribute table.csv")

print(dataset.shape)
print(dataset.head())

X = dataset.drop('Landslide_Occurance', axis=1)
y = dataset['Landslide_Occurance']

print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(class_weight='balanced')
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# To Get Automatic Confusion matrix accuracy result from below technique
from sklearn import metrics
print("Accuracy=")
print(metrics.accuracy_score(y_test,classifier.predict(X_test)))

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df)
prediction = pd.DataFrame(df, columns=['Actual','Predicted']).to_csv('E:/forth year project/final attribute table/test2LogisticRegression/dfdata.csv')


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))