import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle
from sklearn import preprocessing
import pandas as pd


def load_models(model_path):      
    model = pickle.load(open(model_path,'rb'))
    return model

app = Flask(__name__)
@app.route('/predict', methods=['GET'])
def predict():
    # stub input features
    request_json = request.get_json()
    x = request_json['input']
    #transforming X
    x_in = np.array(x).reshape(1,-1)
    #getting the training data
    train_data=pd.read_csv('train_data',index_col=False)
    train_data_length = len(train_data)
    train_data.loc[train_data_length] = x_in[0] #adding the input to the 
                                                #bottom of the table
   #Normalizing data 
    stand_scaler = preprocessing.StandardScaler()
    
    train_scaled = stand_scaler.fit_transform(train_data)
    train_scaled = pd.DataFrame(train_scaled)
    train_scaled.columns=train_data.columns
    train_scaled.index=train_data.index

    # load models
    model2 = load_models('models/lassoCV_reg.sav')
    model3 = load_models('models/linear_model_reg.sav')
    model4 = load_models('models/random_forest_reg.sav')
    
    #estimations
    # The input is in the bottom of the dataframe, that's why we used tail(1)
    prediction_2 = model2.predict(train_scaled.tail(1).values)[0]
    prediction_3 = model3.predict(train_scaled.tail(1).values)[0]
    prediction_4 = model4.predict(train_scaled.tail(1).values)[0]
    
    #mean of estimations
    prediction = (prediction_2+prediction_3+prediction_4)/3
    
    #returning the response 
    response = json.dumps({'response': prediction})
    return response, 200

if __name__ == '__main__':
    application.run(debug=True)