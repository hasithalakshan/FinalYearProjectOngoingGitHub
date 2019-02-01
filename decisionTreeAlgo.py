import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
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


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components =1)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_.cumsum()
print(pca.explained_variance_ratio_)

#######################################################################################

# Fitting Decision Tree Classification to the Training set
model=DecisionTreeClassifier(criterion = 'entropy', random_state = 0,class_weight='balanced')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# save the model to disk
filename ='DecisionTree_model.sav'
joblib.dump(model, filename)


#######################################################################################
# some time later...

# load the model from disk
loaded_DecisionTree_model = joblib.load("C:/Users/user/PycharmProjects/testpanda5/DecisionTree_model.sav")
result = loaded_DecisionTree_model.score(X_test, y_test)
print(" ")
print("Model Accuracy result= %.2f%%" % (result * 100.0))
print(" ")

#Confision Acuuracy Score
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy Score: %.2f%%" % (accuracy * 100.0))

# Making the confusion matrix
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

#for confution matrx classification report
report = classification_report(y_test, y_pred)
print(report)

#######################################################################################


#ploting roc curve
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure()
plt.plot(false_positive_rate, true_positive_rate, color='darkorange',marker='.', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


#show mean errors
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


########################################################################

# #Here load the full dataset
# fulldataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')
# Xalldata = fulldataset.iloc[:, 3:9].values
# # make a prediction
# yalldata = loaded_DecisionTree_model.predict(Xalldata.reshape(-1, 1))
# yalldataproba = loaded_DecisionTree_model.predict_proba(Xalldata.reshape(-1, 1))[:, 1]
# print("Predicted=%s" % (yalldata))
# print("Predicted=%s" % (yalldataproba))
#
#
#
# prediction = pd.DataFrame(yalldataproba, columns=['ynew_predict_probability']).to_csv(
#     'E:/forth year project/final attribute table/DicisionTreeAlgo/predict_probability_DecisionTree.csv')