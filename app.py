from flask import Flask,request, url_for, redirect, render_template, json
import pickle
import pandas as pd
from flask import jsonify
import numpy as np

app = Flask(__name__)

model_diabetes = pickle.load(open("Diabetes.pkl", "rb"))
model_heart = pickle.load(open("heart1.pkl", "rb"))
model_kidney = pickle.load(open("kidney.pkl", "rb"))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
    text8 = request.form['8']
 
    row_df_diabetes = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8])])
    prediction=model_diabetes.predict_proba(row_df_diabetes)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    output = str(float(output)*100)+'%'
    if output>str(0.5):
        return render_template('result.html',pred=f'You have chance of having diabetes.\nProbability of having Diabetes is {output}')
    else:
        return render_template('result.html',pred=f'You are safe.\n Probability of having diabetes is {output}')


#my logic
@app.route('/predict_disease',methods=['POST','GET'])
def predict_disease():
    
    data = request.get_json()
    userid = data['userId'] 
    #logistic regression 84%
    if(userid == 101):
        row_df_diabetes = pd.DataFrame([pd.Series([6,148,80,35,0,33.6,0.627,50])])
        prediction_diabetes=model_diabetes.predict_proba(row_df_diabetes)
        output='{0:.{1}f}'.format(prediction_diabetes[0][1], 2)
        output_diabetes = str(float(output)*100)+'%'
        course_diabetes = {'output_diabetes' : output_diabetes,'Course_number' : ["diabetes","taking-your-medication","yoga"] , 'article_diabetes' : ["1005","1006","1008"] } 
        
        
        #for heart 0 no or 1 yes RandomForestClassifier 84%
        values = np.asarray([46,1,2,150,231,0,1,147,0,3.6,1,0,2])
        output_heart = str(model_heart.predict(values.reshape(1, -1))[0])
        course_heart = {'output_heart' : output_heart,'Course_number' : ["heart-failure","anxiety-mini-course","yoga"], 'article_heart' : ["1001","1007","1004"]}
        
        
       
        
        #for kidney 0 no or 1 yes RandomForestClassifier 100% on test data
        values = np.asarray([40,80,0,0,0,0,0,0,140,10,1.2,5,10400,0,0,0,0,0])
        output_kidney = str(model_kidney.predict(values.reshape(1, -1))[0])
        course_kidney = {'output_kidney' : output_kidney,'Course_number' : ["kidney","msk","taking-your-medication"], 'article_kidney' : ["1003","1005","1007"]}
        
        data = {"clinical_data" : {"blood_pressure" : 80, "hypertension" : 0, "BMI" : 33.6, "Insulin" : 0,"blood_sugar" : 280 , "heart_rate" : 125}}
        
        total_response = {"Heart" : course_heart, "Diabetes" : course_diabetes, "Kidney" : course_kidney, "data" : data}
        
        
    return total_response
        
        
#    return diabetes_response



    

if __name__ == '__main__':
    app.run(debug=True)
