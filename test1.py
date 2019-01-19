import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


#import os
#print(os.listdir("../input"))
dataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')

# Any results you write to the current directory are saved as output.

print(dataset)
print(dataset.shape)
# print dataset information
dataset.info()
# //print dataset column details
print(dataset.columns)

x = dataset.iloc[:,3:9].values
y= dataset.iloc[:,9].values

print("this is y")
print(y.shape)
print(y)

# To write to a csv file
prediction = pd.DataFrame(y, columns=['predictions']).to_csv('E:/forth year project/final attribute table/prediction.csv')



# Split the data into Training and Testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.34,random_state=0)

print("This is X_tain  and y_train")
print(x_train.shape)
print(y_train.shape)


prediction = pd.DataFrame(y_train, columns=['Y train']).to_csv('E:/forth year project/final attribute table/y train.csv')
prediction = pd.DataFrame(y_test, columns=['X train']).to_csv('E:/forth year project/final attribute table/y test.csv')

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# X_train = sc.fit_transform(x_train)
# X_test = sc.transform(x_test)

print(x_train)
print(x_test)

# Feature scalling has done
# Next do the prediction

#Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
classifier = LogisticRegression(penalty='l2', class_weight='balanced', C=1)
classifier.fit(x_train,y_train)



# Predicting the Test set results
y_pred = classifier.predict(x_test)
# y_pred2=classifier.predict_proba(x_test)
# print(y_pred2)

print("Y prediction values=")
print(y_pred)
prediction = pd.DataFrame(y_pred, columns=['predictions2']).to_csv('E:/forth year project/final attribute table/prediction2.csv')


# Support Vector Machines
# from sklearn.svm import SVC
# svc = SVC()
# svc.fit(x_train, y_train)
# y_pred = svc.predict(x_test)
# acc_svc = round(svc.score(x_train, y_train) * 100, 2)
# print(acc_svc)




# Confusion matrix
# Track prediction accuracy results of True positive vs True Negative (Sensitivity vs Specificity)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Confution Matrix=")
print(cm)


# To Get Automatic Confusion matrix accuracy result from below technique
from sklearn import metrics
print("Accuracy=")
print(metrics.accuracy_score(y_test,classifier.predict(x_test)))

Xnew = dataset.iloc[:,3:9].values
# make a prediction
ynew0 = classifier.predict(Xnew)
ynew = classifier.predict_proba(Xnew)[:,0]
print("Predicted=%s" % (ynew0))
print("Predicted=%s" % (ynew))
print(ynew0)
print(ynew)

prediction = pd.DataFrame(ynew, columns=['ynew_predict_probability']).to_csv('E:/forth year project/final attribute table/ynew_predict_probability.csv')