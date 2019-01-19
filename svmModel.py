import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



bankdata  = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')

bankdata.shape

bankdata.head()

x = bankdata.iloc[:,3:9].values
y= bankdata.iloc[:,9].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear' ,probability=True)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# To Get Automatic Confusion matrix accuracy result from below technique
from sklearn import metrics
print("Accuracy=")
print(metrics.accuracy_score(y_test,svclassifier.predict(X_test)))

print(y_pred)

ynew = svclassifier.predict_proba(x)[0]
print("Predicted=%s" % (ynew))