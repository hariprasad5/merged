import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
from keras.models import load_model

import pickle
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import json
import random
from tensorflow.python.keras.backend import set_session


global session
sess = tf.Session()
set_session(sess)
#remember to add a data.pkl here
data = pickle.load( open( "ahad-data.pkl", "rb" ) )
words = data['words']
classes = data['classes']

data1 = pickle.load( open( "opus-data.pkl", "rb" ) )
words1 = data1['words']
classes1 = data1['classes']


with open('trained1.json') as json_data:
    intents = json.load(json_data)
lisIndex = []
for indexIntent in intents['intents']:
    lisIndex.append(indexIntent['tag'])
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

with open('opus1.json') as json_data:
    intents1 = json.load(json_data)
lisIndex1 = []
for indexIntent in intents1['intents']:
    lisIndex1.append(indexIntent['tag'])
def clean_up_sentence1(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in classify_local(sentence, intents)bag: %s" % w)

    return(np.array(bag))


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow1(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence1(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in classify_local(sentence, intents)bag: %s" % w)

    return(np.array(bag))


session = None

global graph
graph = tf.get_default_graph()

#remember to model.pkl here
with open(f'ahad-model.pkl', 'rb') as f:
    model = pickle.load(f)
model._make_predict_function()

#remember to model.pkl here
with open(f'opus-model.pkl', 'rb') as f:
    model1 = pickle.load(f)
model1._make_predict_function()

def predict(input):
    with session.graph.as_default():
        set_session(sess)
        result = model.predict([input])[0]
        return result

def predict1(input):
    with session.graph.as_default():
        set_session(sess)
        result = model1.predict([input])[0]
        return result

def classify_local(sentence, intents):
    ERROR_THRESHOLD = 0.45
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = predict(input_data)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    return_resp = []
    if not results:
        return {
      "tag": "noanswer",
      "patterns": [],
      "responses": [ "Sorry, can't understand you <br>For further details, please email us at sales@ahadbuilders.com or call at +91-7676150150. <br>We would be glad to assist you." ],
      "action": "link",
      "resources": "https://ahad's website",
      "nextpattern": "",
      "buttons": [ "About Ahad", "Brochure", "About AhadExcellencia", "Floor plans/details", "Location" ],
      "context": [ "" ]
    }
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    choice = random.randint(0, len(intents['intents'][lisIndex.index(return_list[0][0])]['responses'])-1)
    print("result ----", return_list)
    resp = {
        "intent": return_list[0],
        #here add rest of the feilds
        "response": intents['intents'][lisIndex.index(return_list[0][0])]['responses'][choice],
        "buttons": [intents['intents'][lisIndex.index(return_list[0][0])]['buttons']],
        "nextpattern":intents['intents'][lisIndex.index(return_list[0][0])]['nextpattern'],
        "action":intents['intents'][lisIndex.index(return_list[0][0])]['action'],
        "resources":intents['intents'][lisIndex.index(return_list[0][0])]['resources']
    }

    return resp

def classify_local1(sentence, intents):
    ERROR_THRESHOLD = 0.40
    input_data = pd.DataFrame([bow1(sentence, words)], dtype=float, index=['input'])
    results = predict1(input_data)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    return_resp = []
    if not results:
        return {
            "tag": "noanswer",
            "responses": "Sorry, I can't understand you. I can speak about Ahad Builders.<br> Please type 'Help' <br>For further details, please email us at sales@ahadbuilders.com or call at +91-7676150150. <br>We would be glad to assist you.",
            "action": "link",
            "resources": "https://ahad's website",
            "nextpattern": "",
            "buttons": ["About Ahad", "Brochure", "About AhadOpus", "Floor plans", "Location"],
            "context": [""]
        }
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    choice = random.randint(0, len(intents1['intents'][lisIndex1.index(return_list[0][0])]['responses'])-1)
    print("result ----", return_list)
    resp = {
        "intent": return_list[0],
        #here add rest of the feilds
        "response": intents1['intents'][lisIndex1.index(return_list[0][0])]['responses'][choice],
        "buttons": [intents1['intents'][lisIndex1.index(return_list[0][0])]['buttons']],
        "nextpattern":intents1['intents'][lisIndex1.index(return_list[0][0])]['nextpattern'],
        "action":intents1['intents'][lisIndex1.index(return_list[0][0])]['action'],
        "resources":intents1['intents'][lisIndex1.index(return_list[0][0])]['resources']
    }

    return resp


def chat(sentence, intents):
    ERROR_THRESHOLD = 0.45
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = predict(input_data)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    return_resp = []
    if not results:
        return "Sorry, can't understand you <br>For further details, please email us at sales@ahadbuilders.com or call at +91-7676150150. <br>We would be glad to assist you."
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    choice = random.randint(0, len(intents['intents'][results[0][0]]['responses'])-1)
    print(intents['intents'][results[0][0]]['responses'][choice])
    print("return-list\n----\n", lisIndex.index(return_list[0][0]), "\n----\n")

    response_back = intents['intents'][lisIndex.index(return_list[0][0])]['responses'][choice]
    return response_back


def chat1(sentence, intents):
    ERROR_THRESHOLD = 0.40
    input_data = pd.DataFrame([bow1(sentence, words)], dtype=float, index=['input'])
    results = predict1(input_data)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    return_resp = []
    if not results:
        return "Sorry, I can't understand you. I can speak about Ahad Builders.<br> Please type 'Help' <br>For further details, please email us at sales@ahadbuilders.com or call at +91-7676150150. <br>We would be glad to assist you."
    for r in results:
        return_list.append((classes1[r[0]], str(r[1])))
    choice = random.randint(0, len(intents1['intents'][results[0][0]]['responses'])-1)
    print(intents1['intents'][results[0][0]]['responses'][choice])
    print("return-list\n----\n", lisIndex1.index(return_list[0][0]), "\n----\n")

    response_back = intents1['intents'][lisIndex1.index(return_list[0][0])]['responses'][choice]
    return response_back




app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "index page"

@app.route("/conv/<apikey>/<botid>", methods=['POST'])
def conv(apikey, botid):
    content = request.json
    sentence = content['sentence']
    if apikey == 'ahad123' and botid == 'b1':
        return jsonify(classify_local(sentence, intents))
    elif apikey == 'ahad456' and botid == 'b1':
        return jsonify(classify_local1(sentence, intents))
    else:
        return "Not authorised"

@app.route("/converse/<api>/<bot>", methods=['POST'])
def postMeth(api, bot):
    if api=='ahad123' and bot=='b1':
        content = request.form['msg']
        print("ok")
        print(content)
        return chat(content, intents)
    elif api=='ahad456' and bot=='b1':
        content = request.form['msg']
        print("ok")
        print(content)
        return chat1(content, intents)
    else:
        return "Not authorised"

@app.route("/chat")
def chatapp():
    return render_template('index.html')

if __name__ == '__main__':
    config = tf.ConfigProto(
        device_count={'GPU': 1},
        intra_op_parallelism_threads=1,
        allow_soft_placement=True
    )

    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    session = tf.Session(config=config)

    app.run(debug=True, threaded=False, port= 8080, host="0.0.0.0")
