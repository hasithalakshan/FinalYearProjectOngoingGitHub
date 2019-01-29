import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report  # for confusion matrix classification report
from sklearn.externals import joblib # This is used for save the model and load the model
from matplotlib import pyplot as plt



# read csv dataset for training
dataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv',nrows=20000)

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

# Random Oversampling
dataset_class_1_over = dataset_class_1.sample(count_class_0, replace=True)
dataset_over = pd.concat([dataset_class_0, dataset_class_1_over], axis=0)

print('Random over-sampling:')
print(dataset_over.Landslide_Occurance.value_counts())

dataset_over.Landslide_Occurance.value_counts().plot(kind='bar', title='Count (Landslide_Occurance)');


#######################################################################################


x = dataset_over.iloc[:, 3:9].values
y = dataset_over.iloc[:, 9].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=1)


# Create SCM Model
svclassifier = SVC(kernel='linear' ,probability=True)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)


# This is not model accuracy here is done value comparison
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Create Confision Matrix
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

# for confution matrx classification report
report = classification_report(y_test, y_pred)
print(report)


#######################################################################################

# save the model to disk
filename = 'SVM_model.sav'
joblib.dump(svclassifier, filename)


#######################################################################################
# some time later...

# load the model from disk
loaded_RandomForest_model = joblib.load("C:/Users/user/PycharmProjects/testpanda5/SVM_model.sav")
result = loaded_RandomForest_model.score(X_test, y_test)
print(" ")
print("Model Accuracy result= %.2f%%" % (result * 100.0))
print(" ")


#######################################################################################

#
# # roc
# from sklearn.metrics import roc_curve, auc
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)
#
# # plot roc curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange',marker='.', label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.show()
#
#
# #######################################################################################
#
#
# # Here load the full dataset
# fulldataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')
# Xalldata = fulldataset.iloc[:, 3:9].values
# # make a prediction
# yalldata = loaded_RandomForest_model.predict(Xalldata)
# yalldataproba = loaded_RandomForest_model.predict_proba(Xalldata)[:, 1]
# print("Predicted=%s" % (yalldata))
# print("Predicted=%s" % (yalldataproba))
#
#
# # Save the predicted probabilities values to the csv file
# prediction = pd.DataFrame(yalldataproba, columns=['ynew_predict_probability']).to_csv(
#     'E:/forth year project/final attribute table/SVMAlgo/predict_probability_SVM.csv')