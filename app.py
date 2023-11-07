import traceback
from flask import Flask, request, jsonify
from sklearn.externals import joblib
# import joblib

import numpy as np
from flask_cors import CORS

# from flask import Flask
# app = Flask(__name__)


app = Flask(__name__)
CORS(app)

clf = joblib.load('./model/model.pkl')

def getParameters():
    age = request.args.get('age')
    gender = request.args.get('gender')
    chest_pain= request.args.get('chest_pain')
    sugar = request.args.get('sugar')
    restecg = request.args.get('restecg')
    exang = request.args.get('exang')
    slope = request.args.get('slope')
    CA = request.args.get('CA')
    thal = request.args.get('thal')
    bp = request.args.get('bp')
    cholesterol = request.args.get('cholesterol')
    thalach = request.args.get('thalach')

    # cigsPerDay = request.args.get('cigs')
    # totChol = request.args.get('chol')
    # sysBP = request.args.get('sBP')
    # diabetes = request.args.get('dia')
    # diaBP = request.args.get('dBP')
    # glucose = request.args.get('gluc')
    # heartRate = request.args.get('hRate')

    params = {
        'age':age,
        'gender': gender,
        'chest_pain': chest_pain,
        'sugar': sugar,
        'restecg': restecg,
        'exang': exang,
        'slope': slope,
        'CA': CA,
        'thal': thal,
        'bp':bp,
        'cholesterol': cholesterol,
        'thalach': thalach
    }
    return (params)

@app.route('/predict', methods=['GET'])
def predict():
    global clf
    if clf:
        try:

            params = getParameters()
            input = np.array(
                [[
                int(params['age']),
                int(params['gender']),
                int(params['chest_pain']),
                float(params['sugar']),
                float(params['restecg']),
                float(params['exang']),
                float(params['slope']),
                float(params['CA']),
                float(params['thal']),
                float(params['bp']),
                float(params['cholesterol']),
                float(params['thalach'])
                ]]
            ) 
            prediction = (clf.predict(input)).tolist()
            probability = (clf.predict_proba(input)).tolist()
            return jsonify(
                {
                    # 'probability': probability,
                    'prediction': prediction[0],
                    # 'data': params
                }
            )
        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        return("no model")


@app.route('/model')
def model():
    global clf
    coefficients = clf.coef_.tolist()
    intercept = clf.intercept_.tolist()
    return jsonify(
        {
            'model': 'Logistic Regression',
            'coefficients': coefficients,
            'intercept': intercept
        }
    )

@app.route('/')
def index():
    return('<h1> JSON API for predicting Coronary Heart Disease in a patient. </h1>'+
            '<h2> An example query is <a href="https://heartapi.herokuapp.com/predict?age=31&sex=1&cigs=5&chol=230&sBP=280&dia=0&dBP=90&gluc=87&hRate=84">this</a></h2>'+
            '<h3>It uses parameters such as Age, Sex, Blood Pressure, Diabetes, Glucose and Heart rate for prediction. </h3>'+
            '<h3> This api uses scikit-learn for training the ML model and Flask for serving the webservice. It is trained on a Logistic Regression model with ~88% accuracy</h3>'+
            '<p>For more details view <a href="https://github.com/agoel00/HeartDiseasePredictionAPI">source</a></p>' +
            '<p>Made with 💙 by <a href="https://github.com/agoel00">Anmol Goel</a></p>')

if __name__ == '__main__':
    print('Model loaded successfully!')
    app.run()