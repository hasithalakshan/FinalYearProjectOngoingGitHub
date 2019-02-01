import pandas as pd
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from flask import Flask, request, redirect, url_for, flash, jsonify
import pickle as p
import numpy as np
import json


#######################################################################################


app = Flask(__name__)

# load the model from disk
loaded_RandomForest_Trainned_model = joblib.load("C:/Users/user/PycharmProjects/testpanda5/LogisticRegression_model.sav")

# Here load the full dataset
fulldataset = pd.read_csv('E:/forth year project/final attribute table/one record.csv')
print(fulldataset)
Xalldata = fulldataset.iloc[:, 3:9].values


@app.route('/risk', methods=['POST'])
def makecalc():
    # data = request.get_json()
    # data = request.data
    # print(data)
    prediction = np.float(loaded_RandomForest_Trainned_model.predict_proba(Xalldata)[:, 1])

    print(prediction)
    # prediction = int(prediction)



    if prediction > 0.8:
        risk ="Very High"
    elif prediction > 0.6:
        risk = "High"
    elif prediction > 0.4:
        risk = "Moderate"
    elif prediction > 0.2:
        risk = "Low"
    else:
        risk = "Very Low"


    # return jsonify({'prediction': str(risk)})
    return "Landslide Susceptability = %s %.2f%%." % (risk , prediction*100)

if __name__ == '__main__':

    print('Model columns loaded')
    app.run()




#######################################################################################


# Here load the full dataset
# fulldataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')
# Xalldata = fulldataset.iloc[:, 3:9].values
# # make a prediction
# yalldata = loaded_RandomForest_Trainned_model.predict(Xalldata)
# yalldataproba = loaded_RandomForest_Trainned_model.predict_proba(Xalldata)[:, 1]
# print("Predicted=%s" % (yalldata))
# print("Predicted=%s" % (yalldataproba))


# prediction = [map(int, x) for x in prediction]