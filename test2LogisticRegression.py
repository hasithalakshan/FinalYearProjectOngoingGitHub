import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOT


dataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')


print(dataset)
print(dataset.shape)


x = dataset.iloc[:,3:9].values
y= dataset.iloc[:,9].values


# Split the data into Training and Testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.34,random_state=0)

print("This is X_tain  and y_train")
print(x_train.shape)
print(y_train.shape)

# sm = SMOTE(random_state=2)
# X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())


#Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
classifier = LogisticRegression(penalty='l2', class_weight='balanced', C=1)
classifier.fit(x_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(x_test)


prediction = pd.DataFrame(y_pred, columns=['predictions2']).to_csv('E:/forth year project/final attribute table/test2LogisticRegression/prediction2.csv')


# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Confution Matrix=")
print(cm)


# To Get Automatic Confusion matrix accuracy result from below technique
from sklearn import metrics
print("Accuracy=")
print(metrics.accuracy_score(y_test,classifier.predict(x_test)))

from sklearn.metrics import recall_score
rs=recall_score(y_test,y_pred)
print(rs)


Xnew = dataset.iloc[:,3:9].values
# make a prediction
ynew0 = classifier.predict(Xnew)
ynew = classifier.predict_proba(Xnew)[:,1]
print("Predicted=%s" % (ynew0))
print("Predicted=%s" % (ynew))
print(ynew0)
print(ynew)

prediction = pd.DataFrame(ynew, columns=['ynew_predict_probability']).to_csv('E:/forth year project/final attribute table/test2LogisticRegression/ynew_predict_probability.csv')


