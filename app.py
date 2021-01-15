import os
import pickle
from flask import Flask, render_template, jsonify, request, make_response
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import re
import string


class Config(object):
    SECRET_KEY = b'_5#y2L"F4Q8z\n\xec]/'
    APP_PATH = os.path.dirname(__file__)
    print(APP_PATH)

app = Flask(__name__)
app.config.from_object(Config)

def preprocess(title, sw_remover, stemmer):

    ## Case Folding (sentence lowering, remove punctuation & numbers)
    title = title.lower()
    title = re.sub(r'\d+', '', title)
    title = title.translate(str.maketrans('', '', string.punctuation))
    title = title.strip()
    
    ## Filtering (Stopwords removal)
    title = sw_remover.remove(title)

    ## Stemming (Change to bsaic form)
    title = stemmer.stem(title)

    return title

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

@app.route('/predictNewsTitle', methods=['GET'])
def predict_news_title():
    data = request.get_json()
    title_args = request.args.get('q')
    print(data)
    title_json = data['title'] if data is not None else None
    title_raw = title_args if title_args is not None else title_json
    sw_remover = StopWordRemoverFactory().create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    model = pickle.load(open("final_model.pickle", "rb"))
    title = preprocess(title_raw, sw_remover, stemmer)
    title = vectorizer.transform([title])
    predicted_label = model.predict(title)[0]
    result = {
        'title': title_raw,
        'predicted_label': predicted_label 
    }
    return make_response(jsonify(result), 200)

if __name__ == '__main__':
    app.run(debug=True)