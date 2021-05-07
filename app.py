import math
import numpy as np
import datetime
from flask import Flask, request, jsonify, render_template
import pickle
from model import Model

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recheck',methods=['POST','GET'])
def recheck():
    return render_template('index.html')
	
@app.route('/loginacc',methods=['POST','GET'])
def loginacc():
    return render_template('login.html')

@app.route('/filedata',methods=['POST','GET'])
def filedata():
    return render_template('files.html')

@app.route('/cvr.ac.in',methods=['POST','GET'])
def cvrweb():
    return redirect(url_for('http://cvr.ac.in/home4/'))
	

	
@app.route('/predict',methods=['POST','GET'])
def predicts():  
  #df = __model.dataset_preprocessed
  
  
  
  
  
  
  
  selector_city=request.form['city']
  #Zone=request.form['zone']
  propertyType=request.form['type']
  size=request.form['size']
  rooms=request.form['room']
  bathrooms=request.form['bathroom']
  statusOutput=request.form['status']
  
  #parking_price_df =(df.loc[df['parkingSpacePrice']>1000].groupby('district').mean()['parkingSpacePrice'])
  
  value_model =model.predict( 
            size,
            propertyType,
            (selector_city),
            statusOutput,
            roomsCat,
            bathroomsCat,
            )
	
  #parking_space = int(parking_price_df.loc[parking_price_df.index==(selector_city+district)].values[0])
  return render_template('conform.html',value='{} %'.format(value))
  



  
if __name__ == "__main__":
     #app.run(debug=True)
     app.run(host="0.0.0.0",port=6442) 
  
