import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report  #for confution matrx classification report
from sklearn.externals import joblib #This is used for save the model and load the model
# from imblearn.over_sampling import SMOTE


# read csv dataset for training
dataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv',nrows=100000)

print(dataset)
print(dataset.shape)


# Check the missing values
if dataset.isnull().values.any():
  print("has null values")
else:
  print("has not null values")


x = dataset.iloc[:, 3:9].values
y = dataset.iloc[:, 9].values

# Split the data into Training and Testing set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=42)

print("This is X_tain  and y_train")
print(x_train.shape)
print(y_train.shape)



# Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression

# classifier = LogisticRegression(random_state = 0)
classifier = LogisticRegression(penalty='l2', class_weight='balanced', C=1)
classifier.fit(x_train, y_train)

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(classifier, filename)

# some time later...

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(x_test, y_test)
print("result=")
print(result)



y_predP = loaded_model.predict_proba(x_test)

u = 0
for w in y_predP:

    if w[0] > w[1]:
        y_predP[u] = w[0]
        u = u + 1

    else:
        y_predP[u] = w[1]
        u = u + 1

# Predicting the Test set results
y_pred = loaded_model.predict(x_test)

prediction = pd.DataFrame(y_predP[:, 0], columns=['predictions2']).to_csv(
    'E:/forth year project/final attribute table/test3LogisticRegression/prediction2.csv')

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confution Matrix=")
print(cm)

#for confution matrx classification report
report = classification_report(y_test, y_pred)
print(report)

# To Get Automatic Confusion matrix accuracy result from below technique
from sklearn import metrics
print("Accuracy=")
print(metrics.accuracy_score(y_test, loaded_model.predict(x_test)))

# from sklearn.metrics import recall_score
# rs = recall_score(y_test, y_pred)
# print(rs)


fulldataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')
Xalldata = fulldataset.iloc[:, 3:9].values
# make a prediction
yalldata = loaded_model.predict(Xalldata)
yalldataproba = loaded_model.predict_proba(Xalldata)[:, 1]
print("Predicted=%s" % (yalldata))
print("Predicted=%s" % (yalldataproba))


prediction = pd.DataFrame(yalldataproba, columns=['ynew_predict_probability']).to_csv(
    'E:/forth year project/final attribute table/test3LogisticRegression/ynew_predict_probability.csv')


