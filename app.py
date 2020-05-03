from tensorflow.keras.preprocessing.sequence import pad_sequences
import re#regular expression
import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from tensorflow.keras.models import load_model
import pickle
import trstop
from string import digits
import flask

model = load_model('my_model.h5')
replace_espaco = re.compile("[/(){}\[\]\|@,.;!#'?₺*]")

def pre_processamento(text):
    text = text.lower()
    text = replace_espaco.sub(' ', text)
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    text = ' '.join(word for word in text.split() if trstop.is_stop_word(word)==False)
    return text

def getSikayetSinif(complaint):
    complaint_series = pd.Series(complaint)
    complaint_series = complaint_series.apply(pre_processamento)
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    tamanho_maximo_sent = 250
    complaint_series = tokenizer.texts_to_sequences(complaint_series.values)
    complaint_series = pad_sequences(complaint_series,maxlen=tamanho_maximo_sent)
    class_name = np.load('class_name.npy')
    yhat = model.predict_classes(complaint_series[0:1], verbose=0)
    return class_name[yhat]

app = flask.Flask(__name__)
app.config["DEBUG"] = False

@app.route('/sikayet/<string:sikayet>')
def sonuc(sikayet):
    return str(getSikayetSinif(sikayet))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)