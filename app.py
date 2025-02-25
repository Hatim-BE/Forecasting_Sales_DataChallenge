from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Unnamed = 0,
            id_produit =  request.form.get("id_produit"),
            date =  request.form.get("date"),
            categorie =  request.form.get("categorie"),
            marque =  request.form.get("marque"),
            prix_unitaire =  float(request.form.get("prix_unitaire", 0)),  # Convert to float, default to 0 if missing
            promotion =  float(request.form.get("promotion", 0)),  # Convert to float
            jour_ferie =  request.form.get("jour_ferie"),
            weekend =  request.form.get("weekend"),
            stock_disponible =  int(request.form.get("stock_disponible", 0)),  # Convert to int
            condition_meteo =  request.form.get("condition_meteo"),
            region =  request.form.get("region"),
            moment_journee =  request.form.get("moment_journee"),
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('index.html',results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)  


