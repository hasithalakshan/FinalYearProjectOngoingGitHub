import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report  # for confusion matrix classification report
from sklearn.externals import joblib # This is used for save the model and load the model
from matplotlib import pyplot as plt


# read csv dataset for training
dataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')


# Check the missing values
if dataset.isnull().values.any():
  print("has null values")
else:
  print("has not null values")


print(dataset.isnull().sum())
print("")


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


# Feature Scaling
from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# print(X_train)

# from sklearn.ensemble import RandomForestRegressor


# Create RandomForest Model
model = RandomForestClassifier(n_estimators=20, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# This is not model accuracy here is done value comparison
accuracy = accuracy_score(y_test, y_pred.round())
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Create Confision Matrix
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred.round())
print('Confusion matrix:\n', conf_mat)

# for confution matrx classification report
report = classification_report(y_test, y_pred.round())
print(report)

# print the confusion matrix in the plot
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
filename = 'RandomForest_model.sav'
joblib.dump(model, filename)


#######################################################################################
# some time later...

# load the model from disk
loaded_RandomForest_model = joblib.load("C:/Users/user/PycharmProjects/testpanda5/RandomForest_model.sav")
result = loaded_RandomForest_model.score(X_test, y_test)
print(" ")
print("Model Accuracy result= %.2f%%" % (result * 100.0))
print(" ")


#######################################################################################


# roc
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# plot roc curve
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


# feature importance
from matplotlib import pyplot
print("Feature Importances :",loaded_RandomForest_model.feature_importances_)
# Draw The plot
pyplot.bar(range(len(loaded_RandomForest_model.feature_importances_)), loaded_RandomForest_model.feature_importances_)
plt.xticks(range(6),["Distance_to_Stream", "Distance_to_Road", "Plane_Curvature", "Profile_Curvature", "Slope_Angle", "Slope_Aspect"] ,rotation='vertical')
pyplot.show()


#######################################################################################


# Here load the full dataset
fulldataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')
Xalldata = fulldataset.iloc[:, 3:9].values
# make a prediction
yalldata = loaded_RandomForest_model.predict(Xalldata)
yalldataproba = loaded_RandomForest_model.predict_proba(Xalldata)[:, 1]
print("Predicted=%s" % (yalldata))
print("Predicted=%s" % (yalldataproba))


# Save the predicted probabilities values to the csv file
prediction = pd.DataFrame(yalldataproba, columns=['ynew_predict_probability']).to_csv(
    'E:/forth year project/final attribute table/RandomForestAlgoTest/predict_probability_RandomForest.csv')