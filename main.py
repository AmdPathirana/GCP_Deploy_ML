import imp
import numpy as np
import pickle
from flask import Flask , request
from sklearn.linear_model import LogisticRegression


app=Flask(__name__)
# load the model when the application execute
model_pkl = pickle.load(open('flowerV1.pkl','rb' ))
@app.route('/api_predict', methods=['GET', 'POST'])
def api_predict():
    if request.method=='GET':
        return "Please send a POST request"
    elif request.method == 'POST':
        
        data = request.get_json()
        
        sepal_length = data["sepal_length"]
        sepal_width = data["sepal_width"]
        petal_length = data["petal_length"]
        petal_width = data["petal_width"]
    
        data = np.array([[sepal_length, sepal_width, 
                          petal_length, petal_width]])
           
        prediction = model_pkl.predict(data)
        return str(prediction)
app.run()