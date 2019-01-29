import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report  #for confution matrx classification report
from sklearn.externals import joblib #This is used for save the model and load the model
from matplotlib import pyplot as plt
import seaborn as sns

# read csv dataset for training
dataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')

print(dataset.columns)


# Check the missing values
if dataset.isnull().values.any():
  print("has null values")
else:
  print("has not null values")

print(dataset.isnull().sum())
print("")


#print landslide occurance count no of 0 and 1 values
target_count = dataset.Landslide_Occurance.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar', title='Count (target)');


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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=1)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# print(X_train)

#create XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#This is not model accuracy here is done value comparison
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score: %.2f%%" % (accuracy * 100.0))

conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

#for confution matrx classification report
report = classification_report(y_test, y_pred)
print(report)

#print the confusion matrix in the plot
labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()


#######################################################################################

# save the model to disk
filename ='XGBoost_model.sav'
joblib.dump(model, filename)

#######################################################################################
# some time later...

# load the model from disk
loaded_XGBoost_model = joblib.load("C:/Users/user/PycharmProjects/testpanda5/XGBoost_model.sav")
result = loaded_XGBoost_model.score(X_test, y_test)
print(" ")
print("Model Accuracy result= %.2f%%" % (result * 100.0))
print(" ")

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
plt.show()



#Show most importance feature
from matplotlib import pyplot
print("Feature Importence")
print(loaded_XGBoost_model.feature_importances_)
# plot
plt.bar(range(len(loaded_XGBoost_model.feature_importances_)), loaded_XGBoost_model.feature_importances_,color='rgbkymc')
plt.xticks(range(len(loaded_XGBoost_model.feature_importances_)),dataset.iloc[:, 3:9].columns ,rotation='vertical')
plt.xlabel('Features')
plt.ylabel('Importences')
plt.title('Feature Importance Comparison')
plt.show()


########################################################################


#Here load the full dataset
fulldataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')
Xalldata = fulldataset.iloc[:, 3:9].values
# make a prediction
yalldata = loaded_XGBoost_model.predict(Xalldata)
yalldataproba = loaded_XGBoost_model.predict_proba(Xalldata)[:, 1]
print("Predicted=%s" % (yalldata))
print("Predicted=%s" % (yalldataproba))


#Save the predicted probabilities values to the csv file
prediction = pd.DataFrame(yalldataproba, columns=['ynew_predict_probability']).to_csv(
    'E:/forth year project/final attribute table/XGBoostAlgoTest/predict_probability_XGBoost.csv')