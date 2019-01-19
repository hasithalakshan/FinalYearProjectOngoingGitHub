import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report  #for confution matrx classification report
from sklearn.externals import joblib #This is used for save the model and load the model
from matplotlib import pyplot as plt

# read csv dataset for training
df_train = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv',nrows=100000)

# Check the missing values
if df_train.isnull().values.any():
  print("has null values")
else:
  print("has not null values")


target_count = df_train.Landslide_Occurance.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)');


#######################################################################################
# Class count
count_class_0, count_class_1 = df_train.Landslide_Occurance.value_counts()

# Divide by class
df_class_0 = df_train[df_train['Landslide_Occurance'] == 0]
df_class_1 = df_train[df_train['Landslide_Occurance'] == 1]

#Random Oversampling
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.Landslide_Occurance.value_counts())

df_test_over.Landslide_Occurance.value_counts().plot(kind='bar', title='Count (Landslide_Occurance)');



#########################################################

x = df_test_over.iloc[:, 3:9].values
y = df_test_over.iloc[:, 9].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=1)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# print(X_train)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=20, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred.round())
print("Accuracy: %.2f%%" % (accuracy * 100.0))

conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred.round())
print('Confusion matrix:\n', conf_mat)

#for confution matrx classification report
report = classification_report(y_test, y_pred.round())
print(report)

# save the model to disk
filename = 'RandomForestTest_model.sav'
joblib.dump(model, filename)

# some time later...

# load the model from disk
loaded_RandomForest_model = joblib.load(filename)
result = loaded_RandomForest_model.score(X_test, y_test)
print("result=")
print(result)

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

########################################################################

fulldataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')
Xalldata = fulldataset.iloc[:, 3:9].values
# make a prediction
yalldata = loaded_RandomForest_model.predict(Xalldata)
yalldataproba = loaded_RandomForest_model.predict_proba(Xalldata)[:, 1]
print("Predicted=%s" % (yalldata))
print("Predicted=%s" % (yalldataproba))


prediction = pd.DataFrame(yalldataproba, columns=['ynew_predict_probability']).to_csv(
    'E:/forth year project/final attribute table/RandomForestAlgoTest/ynew_predict_probability_XGBoost.csv')