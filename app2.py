from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import re
import string

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

sw_remover = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()
vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
model = pickle.load(open("final_model.pickle", "rb"))
print(vectorizer)

titles = ['Raffi Ahmad Ikut Pesta Usai Divaksin Tuai Pro dan Kontra', 
          '3 Resep Omelet Sayuran yang Praktis Benutrisi Buat Sarapan',
          'Chelsea berhasil mengalahkan liverpool',
          'Di Tengah Viral Raffi Ahmad Pesta Tanpa Masker, dr Reisa Nge-tweet Begini',
          'Resort Cakep Berkonsep Pedesaan di Cirebon, Manusia Ikan Pecahkan Rekor',
          'Real Madrid Bakal Lawan Atalanta yang Retak',
          ' Ekonomi Rontok Kena Dampak Corona, Bagaimana Bisnis Budi Daya Ikan?']

for t in titles:
    title = preprocess(t, sw_remover, stemmer)
    title = vectorizer.transform([title])
    print(t, model.predict(title)[0])