from flask import Flask,request,render_template,jsonify
from flask_cors import CORS,cross_origin
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template("index.html")

@app.route('/predict',methods = ['POST' , 'GET'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            pregnancies  = float(request.form['pregnancies'])
            glucose = float(request.form['glucose'])
            bloodpressure = float(request.form['bloodpressure'])
            skinthickness = float(request.form['skinthickness'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            diabetespedigreefunction = float(request.form['diabetespedigreefunction'])
            age = float(request.form['age'])

            filename = "DiabetesPickle.pickle"
            loaded_model = pickle.load(open(filename, 'rb'))
            scaler=pickle.load(open('DiabetesPickleScaled.pickle', 'rb'))
            prediction = loaded_model.predict(scaler.transform([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]]))
            print("Prediction is", prediction[0])

            return render_template('result.html', prediction=prediction[0]*100)
        except Exception as e:
            print('The Exception message is:', e)
            return ("Something is wrong", e)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug = True)