from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import numpy as np
import pandas as pd
import math

app = Flask(__name__)
model1 = pickle.load(open("model1.pkl", "rb"))
model2 = pickle.load(open("model2.pkl", "rb"))
model3 = pickle.load(open("model3.pkl", "rb"))
model4 = pickle.load(open("model4.pkl", "rb"))
model5 = pickle.load(open("model5.pkl", "rb"))
standard = pickle.load(open("standard.pkl", "rb"))


@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        
        age=int(request.form["age"])
        sex=int(request.form["sex"])
        cp=int(request.form["cp"])
        trestbps=int(request.form["trestbps"])
        chol=int(request.form["chol"])
        fbs=int(request.form["fbs"])
        restecg=int(request.form["restecg"])
        thalach=int(request.form["thalach"])
        exang=int(request.form["exang"])
        oldpeak=float(request.form["oldpeak"])
        slope=int(request.form["slope"])
        ca=int(request.form["ca"])
        thal=int(request.form["thal"])


    final_features = np.array([[age,sex ,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    std_features= standard.transform(final_features)

    prediction1 = model1.predict(std_features)
    prediction2 = model2.predict(std_features)
    prediction3 = model3.predict(final_features)
    prediction4 = model4.predict(final_features)
    prediction5 = model5.predict(final_features)

    output = (prediction1[0]+prediction2[0]+prediction3[0]+prediction4[0]+prediction5[0])/0.05
    

    

    
    return render_template('index.html', prediction_text="The probability of having heart disease is {} %".format(output))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
