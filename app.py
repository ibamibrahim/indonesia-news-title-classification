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
    return "Hello World!"

@app.route('/predictNewsTitle', methods=['GET'])
def predict_news_title():
    title_args = request.args.get('q')
    sw_remover = StopWordRemoverFactory().create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    model = pickle.load(open("final_model.pickle", "rb"))
    title_preprocessed = preprocess(title_args, sw_remover, stemmer)
    title = vectorizer.transform([title_preprocessed])
    predicted_label = model.predict(title)[0]
    result = {
        'title': title_args,
        'title_cleaned': title_preprocessed,
        'predicted_label': predicted_label 
    }
    return make_response(jsonify(result), 200)

if __name__ == '__main__':
    app.run(debug=True)