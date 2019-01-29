import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report  #for confution matrx classification report
from sklearn.externals import joblib #This is used for save the model and load the model
from matplotlib import pyplot as plt


# read csv dataset for training
dataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')

print(dataset.columns)
print(dataset.shape)


# Check the missing values
if dataset.isnull().values.any():
  print("has null values")
else:
  print("has not null values")


print(dataset.isnull().sum())
print("")


#######################################################################################

# Class count
count_class_0, count_class_1 = dataset.Landslide_Occurance.value_counts()

# Divide by class
dataset_class_0 = dataset[dataset['Landslide_Occurance'] == 0]
dataset_class_1 = dataset[dataset['Landslide_Occurance'] == 1]

#Random Oversampling
dataset_class_1_over = dataset_class_1.sample(count_class_0, replace=True)
dataset_over = pd.concat([dataset_class_0, dataset_class_1_over], axis=0)

print('Random over-sampling:')
print(dataset_over.Landslide_Occurance.value_counts())

#plot oversampling values
dataset_over.Landslide_Occurance.value_counts().plot(kind='bar', title='Count (Landslide_Occurance)');


#######################################################################################

x = dataset_over.iloc[:, 3:9].values
y = dataset_over.iloc[:, 9].values

# Split the data into Training and Testing set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=42)


# Fitting logistic regression to the training set
classifier = LogisticRegression(penalty='l2', class_weight='balanced', C=1)
classifier.fit(x_train, y_train)

# save the model to disk
filename = 'LogisticRegression_model.sav'
joblib.dump(classifier, filename)

#######################################################################################
# some time later...

# load the model from disk
loaded_LogisticReg_model = joblib.load("C:/Users/user/PycharmProjects/testpanda5/LogisticRegression_model.sav")
result = loaded_LogisticReg_model.score(x_test, y_test)
print(" ")
print("Model Accuracy result= %.2f%%" % (result * 100.0))
print(" ")

#######################################################################################

y_predP = loaded_LogisticReg_model.predict_proba(x_test)

u = 0
for w in y_predP:

    if w[0] > w[1]:
        y_predP[u] = w[0]
        u = u + 1

    else:
        y_predP[u] = w[1]
        u = u + 1

# Predicting the Test set results
y_pred = loaded_LogisticReg_model.predict(x_test)


# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confution Matrix=")
print(cm)

#for confution matrx classification report
report = classification_report(y_test, y_pred)
print(report)


#######################################################################################


#roc
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

#plot roc curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange',marker='.', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


########################################################################


#Load all data
fulldataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')
Xalldata = fulldataset.iloc[:, 3:9].values
# make a prediction
yalldata = loaded_LogisticReg_model.predict(Xalldata)
yalldataproba = loaded_LogisticReg_model.predict_proba(Xalldata)[:, 1]
print("Predicted=%s" % (yalldata))
print("Predicted=%s" % (yalldataproba))


#Save the predicted probabilities values to the csv file
prediction = pd.DataFrame(yalldataproba, columns=['ynew_predict_probability']).to_csv(
    'E:/forth year project/final attribute table/LogisticRegressionAlgo/predict_probability_LogisticReg.csv')


