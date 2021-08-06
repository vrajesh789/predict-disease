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
        row_df_diabetes = pd.DataFrame([pd.Series([4,153,115,28,180,37.8,1.7,62])])
        prediction_diabetes=model_diabetes.predict_proba(row_df_diabetes)
        output='{0:.{1}f}'.format(prediction_diabetes[0][1], 2)
        output_diabetes = str(float(output)*100)+'%'
        course_diabetes = {'output_diabetes' : output_diabetes,'Course_number' : ["diabetes","taking-your-medication","yoga"] , 'article_diabetes' : ["1005","1006","1008"] } 
        
        
        #for heart 0 no or 1 yes RandomForestClassifier 87% on test data
        values = np.asarray([62,1,2,150,231,0,1,147,0,3.6,1,0,2])
        output_heart = str(model_heart.predict(values.reshape(1, -1))[0])
        course_heart = {'output_heart' : output_heart,'Course_number' : ["heart-failure","anxiety-mini-course","yoga"], 'article_heart' : ["1001","1007","1004"]}
        
    
        
        #for kidney 0 no or 1 yes RandomForestClassifier 100% on test data
        values = np.asarray([40,80,0,0,0,0,0,0,140,10,1.2,5,10400,0,0,0,0,0])
        output_kidney = str(model_kidney.predict(values.reshape(1, -1))[0])
        course_kidney = {'output_kidney' : output_kidney,'Course_number' : ["kidney","msk","taking-your-medication"], 'article_kidney' : ["1003","1005","1007"]}
        
        data = {"clinical_data" : {"BMI" : 37.8 , "blood_pressure" : 115, "blood_sugar" : 155}}
        hist_data = {"hist_data" : {"BMI" : [29.3, 31.1, 32.8, 34.7, 35.5, 36.6, 37.8], "blood_pressure" : [110, 116, 120, 115, 109, 112, 115], "blood_sugar" : [156, 148, 150, 158, 160, 151, 155]}}
        total_response = {"Heart" : course_heart, "Diabetes" : course_diabetes, "Kidney" : course_kidney, "data" : data, "hist_data" : hist_data}
        
    elif(userid == 102):
        row_df_diabetes = pd.DataFrame([pd.Series([2,120,115,28,150,30,0.2,46])])
        prediction_diabetes=model_diabetes.predict_proba(row_df_diabetes)
        output='{0:.{1}f}'.format(prediction_diabetes[0][1], 2)
        output_diabetes = str(float(output)*100)+'%'
        course_diabetes = {'output_diabetes' : output_diabetes,'Course_number' : ["diabetes","taking-your-medication","yoga"] , 'article_diabetes' : ["1005","1006","1008"] } 
        
        
        #for heart 0 no or 1 yes RandomForestClassifier 87% on test data
        values = np.asarray([46,1,2,142,177,0,0,160,1,1.4,0,0,2])
        output_heart = str(model_heart.predict(values.reshape(1, -1))[0])
        course_heart = {'output_heart' : output_heart,'Course_number' : ["heart-failure","anxiety-mini-course","yoga"], 'article_heart' : ["1001","1007","1004"]}
        
    
        
        #for kidney 0 no or 1 yes RandomForestClassifier 100% on test data
        values = np.asarray([40,80,0,0,0,0,0,0,140,10,1.2,5,10400,0,0,0,0,0])
        output_kidney = str(model_kidney.predict(values.reshape(1, -1))[0])
        course_kidney = {'output_kidney' : output_kidney,'Course_number' : ["kidney","msk","taking-your-medication"], 'article_kidney' : ["1003","1005","1007"]}
        
        data = {"clinical_data" : {"BMI" : 32 , "blood_pressure" : 139, "blood_sugar" : 151}}
       
        hist_data = {"hist_data" : {"BMI" : [31.8, 32.3, 32.8, 32.6, 32.0, 32.3, 32.0], "blood_pressure" : [132, 136, 140, 143, 138, 140, 139], "blood_sugar" : [165, 159, 160, 158, 155, 153, 151]}}
        
        total_response = {"Heart" : course_heart, "Diabetes" : course_diabetes, "Kidney" : course_kidney, "data" : data, "hist_data" : hist_data}
        
    elif(userid == 103):
        row_df_diabetes = pd.DataFrame([pd.Series([0,115,110,23,130,25,0.167,25])])
        prediction_diabetes=model_diabetes.predict_proba(row_df_diabetes)
        output='{0:.{1}f}'.format(prediction_diabetes[0][1], 2)
        output_diabetes = str(float(output)*100)+'%'
        course_diabetes = {'output_diabetes' : output_diabetes,'Course_number' : ["diabetes","taking-your-medication","yoga"] , 'article_diabetes' : ["1005","1006","1008"] } 
        
        
        #for heart 0 no or 1 yes RandomForestClassifier 87% on test data
        values = np.asarray([25,0,1,110,134,0,1,147,0,3.6,1,0,2])
        output_heart = str(model_heart.predict(values.reshape(1, -1))[0])
        course_heart = {'output_heart' : output_heart,'Course_number' : ["heart-failure","anxiety-mini-course","yoga"], 'article_heart' : ["1001","1007","1004"]}
        
    
        
        #for kidney 0 no or 1 yes RandomForestClassifier 100% on test data
        values = np.asarray([40,80,0,0,0,0,0,0,140,10,1.2,5,10400,0,0,0,0,0])
        output_kidney = str(model_kidney.predict(values.reshape(1, -1))[0])
        course_kidney = {'output_kidney' : output_kidney,'Course_number' : ["kidney","msk","taking-your-medication"], 'article_kidney' : ["1003","1005","1007"]}
        
        data = {"clinical_data" : {"BMI" : 23.5 , "blood_pressure" : 112, "blood_sugar" : 110}}
        hist_data = {"hist_data" : {"BMI" : [23.0, 23.2, 23.1, 23.3, 23.6, 23.4, 23.5], "blood_pressure" : [115, 117, 114, 115, 117, 113, 112], "blood_sugar" : [115, 118, 114, 111, 109, 108, 110]}}
        
        course_fitness = {'Course_number' : ["yoga","sleep","wellbeing"], 'article_fitness' : ["1009","1008","1003"]}
        total_response = {"Heart" : course_heart, "Diabetes" : course_diabetes, "Kidney" : course_kidney, "data" : data , "course_fitness" : course_fitness, "hist_data" : hist_data}
    return total_response
        
        
#    return diabetes_response



    

if __name__ == '__main__':
    app.run(debug=True)
