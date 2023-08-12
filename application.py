import pickle as p
from flask import Flask , request , jsonify , render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #beacuse of Standard Scaler


app = Flask(__name__)

#Import Ridge  Regressor model and Standard Scale pickel
standard_scaler = p.load(open('models/Scaler.pkl' , 'rb'))
ridge_model = p.load(open('models/Ridge.pkl','rb'))

#Route for Home Page
#for every route we have to write a function
@app.route("/")

def index():
    return render_template('index.html')

@app.route('/Prediction' , methods = ['GET' , 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        #we have taken the input from the POST request
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        #Now From this we will predict the new data that we have input

        new_data_scaled = standard_scaler.transform([[Temperature ,RH ,WS ,Rain ,FFMC, DMC , ISI , Classes ,Region]])
        result = ridge_model.predict(new_data_scaled) #In Result we get a list with only one element so we can access that element by index- > 0

        return render_template('home.html' , result = result[0])

    else:
        return render_template('home.html')



if __name__ == "__main__":
    app.run(host = '0.0.0.0')