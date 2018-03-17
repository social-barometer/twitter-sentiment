import sys
import traceback

import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from flask import Flask, request, json, jsonify

from data.preprocess import tokenize

app = Flask(__name__)

model = load_model('./models/lstm-19-0.396-0.839-0.392-0.837.hdf5')

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 280

def analyze(texts):
    """Analyses the emotional content of the texts given."""

    encoding_to_label = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'joy',
        4: 'sadness',
        5: 'surprise',
    }
    word_index, seqs = tokenize(texts, MAX_NB_WORDS)
    padded = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(padded, batch_size=128, verbose=1)

    analyses = []
    for pred in predictions:
        analysis = {}
        for i, prob in enumerate(pred):
            label = encoding_to_label[i]
            percentage = round(prob * 100, 3)
            analysis[label] = percentage

        analyses.append(analysis)

    return analyses

@app.errorhandler(415)
def error_415(error=None):
    message = {
        'status': 415,
        'message': 'Unsupported media type. Please use header Content-Type: application/json'
    }
    resp = jsonify(message)
    resp.status_code = 415

    return resp

@app.route('/emotion-analysis', methods=['GET', 'POST'])
def emotion_analysis():
    if request.headers['Content-Type'] == 'application/json':
        try:
            analysis = analyze(request.json['tweets'])
            return "Analysis: " + json.dumps(analysis)
        except:
            print("Error:", sys.exc_info()[3])
            return "500"
    else:
        return error_415()

if __name__ == '__main__':
    app.run()

