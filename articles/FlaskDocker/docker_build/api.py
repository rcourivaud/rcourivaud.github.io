from flask import Flask, request
import pandas as pd
import json
from sklearn.externals import joblib
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'MAchine Learning API !'

@app.route('/predict_species' , methods=['POST'])
def predict_species():
    if request.method == 'POST':
        print(request)
        print(type(request.get_json()))
        dataframe = pd.DataFrame(request.get_json()["data"])
        dataframe = dataframe[request.get_json()["columns"]]
        print(dataframe.head())
        model = joblib.load("./logisticregression.pkl")
        print(list(model.predict(dataframe)))
        return json.dumps({"result":[int(elt) for elt in list(model.predict(dataframe))]})


if __name__ =="__main__":
    app.run(debug=True, host='0.0.0.0')
