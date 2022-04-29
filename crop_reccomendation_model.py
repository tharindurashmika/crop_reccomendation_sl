"""Created at Sat Apr 09 20:42:00 2022

@author: tharindumatharaarachchi
"""
import joblib
from flask import Flask, Response
import json
import numpy as np
from flask import request
import requests

app= Flask(__name__)

model = joblib.load(r'crop_reccomendation_model')

@app.route('/predict',methods=['POST'])

def predict():
    event = json.loads(request.data)
    values= event['values']
    
    values= list(map(np.float,values))
    pre= np.array(values)
    pre =pre.reshape(1,-1)
    res= model.predict(pre)
    print(res)
    return str(res[0])
    #return "1"

if __name__ == '__main__':
    app.run(debug=True)