import os
import pickle
from flask import Flask, render_template, jsonify, request, make_response


class Config(object):
    SECRET_KEY = b'_5#y2L"F4Q8z\n\xec]/'
    APP_PATH = os.path.dirname(__file__)
    print(APP_PATH)

app = Flask(__name__)
app.config.from_object(Config)

@app.route('/')
def index():
    return "Hello World?"

@app.route('/predict', methods=['GET'])
def predict():
    data = request.get_json()
    gender = 1 if data['gender'] == 'Male' else 0
    smoker = 1 if data['smoker'] else 0
    diabetic = 1 if data['diabetic'] else 0
    model_file = os.path.join(app.config['APP_PATH'], 'model.pkl')
    bmi = calculate_bmi(data['weight'], data['height'])
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
        X_predict = [[gender, data['age'], smoker, data['cig_count'], diabetic, data['bp'], bmi]]
        value = model.predict(X_predict)
        result = {'status':'done', 'value':int(value[0])}
    data['result'] = result
    return make_response(jsonify(data), 200)

def calculate_bmi(weight, height):
    return round(float(weight) / (float(height / 100) * float(height / 100)), 2)


if __name__ == '__main__':
    app.run(debug=True)