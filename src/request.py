#!/usr/dev/bin python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 13:54:40 2023

@author: guille
"""

import joblib
import pandas as pd
import re
import os

os.chdir('/Users/guille/Sentiment-Analysis')

from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords



from flask import Flask, request, jsonify

from flask import Flask,jsonify,render_template,Response
from flask_restful import Resource,Api, reqparse
from flask_basicauth import BasicAuth



model = SentenceTransformer('stsb-roberta-base')

model.load('src/tuned_SBERT')
pipeline_filename = 'src/trained_softmax_pipeline.joblib'

loaded_pipeline = joblib.load(pipeline_filename)


#Load and clean data, do some EDA
def remove_sw(phrases):
    def aux(phrase):
        phrase_tokens = phrase.split()
        filtered = [word for word in phrase_tokens if not word in stopwords.words('english')]
        return " ".join(filtered)
    phrases=phrases.apply(lambda x: aux(x))
    return phrases

def phrase_clean(phrases):
    #Takes a pandas column of phrases and cleans it :)
    def aux(phrase):
        split=phrase.split()
        if len(split)>0:
            if split[0] in set(["'ve","'s","'ll","'re","'d","'n","n't","'m"]):
                split[0]=""
                phrase=" ".join(split)
        return phrase
    #phrases = phrases.apply(lambda x: x.replace("-", " "))
    phrases = phrases.apply(lambda x: x.lower())
    phrases = phrases.apply(lambda x: re.sub(r"[^A-Za-z0-9' ]+", '', x))
    #Skip this step and let phrases begin with "'s" ect?
    phrases = phrases.apply(lambda x: x.replace("  ", " "))
    phrases = phrases.apply(lambda x: x.replace(" 've", "'ve "))
    phrases = phrases.apply(lambda x: x.replace(" 's", "'s "))
    phrases = phrases.apply(lambda x: x.replace(" 'll", "'ll "))
    phrases = phrases.apply(lambda x: x.replace(" 're", "'re "))
    phrases = phrases.apply(lambda x: x.replace(" 'd", "'d "))
    phrases = phrases.apply(lambda x: x.replace(" 'n", "'n "))
    phrases = phrases.apply(lambda x: x.replace(" 'm", "'m "))
    phrases = phrases.apply(lambda x: x.replace(" n't", "n't "))
    phrases = phrases.apply(lambda x: x.replace("  ", " "))
    phrases = phrases.apply(lambda x: x.replace("'''", ""))
    phrases = phrases.apply(lambda x: x.replace("''", ""))
    phrases = phrases.apply(lambda x: x.replace(" ' ", " "))
    phrases = phrases.apply(lambda x: x.replace(" '", " "))
    phrases = phrases.apply(lambda x: x.replace("' ", " "))
    phrases = phrases.apply(lambda x: aux(x))
    phrases = phrases.apply(lambda x: x.strip())
    return phrases



def predict(sentence):
    review_to_encode_clean=phrase_clean(remove_sw(pd.DataFrame({'phrase':sentence},index=[0])['phrase'])).iloc[0,]
    output = int(loaded_pipeline.predict(model.encode([review_to_encode_clean]))[0])

    return output






# Associate emojis with happiness levels
emoji_mapping = {
    1: "1/5 -> 😢",
    2: "2/5 -> 😔",
    3: "3/5 -> 😐",
    4: "4/5 -> 🙂",
    5: "5/5 -> 😄"
}


app = Flask(__name__)
api = Api(app)

app.config['BASIC_AUTH_USERNAME'] = 'guillermo'
app.config['BASIC_AUTH_PASSWORD'] = 'santander'
basic_auth = BasicAuth(app)


# Configura la ruta a la carpeta de plantillas
import os
template_dir = os.path.abspath('/Users/guille/Sentiment-Analysis/src')
app.template_folder = template_dir


from googletrans import Translator
translator = Translator()




@app.route('/', methods=['GET'])
@basic_auth.required
def home():
    with open('img/sentiment_analysis.txt', 'r') as txt_file:
        image = txt_file.read()
    return render_template('bert_front.html',image=image)


@app.route('/get_sentiment_score', methods=['GET'])
def get_sentiment_score():
    # Get the phrase from the user
    phrase = request.args.get('phrase')

    translation_needed = translator.detect(phrase).lang!='en' 
    

    if translation_needed:
        phrase_in_english = translator.translate(phrase).text
        happiness_level = predict(phrase_in_english)
    else:
        happiness_level = predict(phrase)

    # Get the associated emoji
    emoji = emoji_mapping[happiness_level]
    
    print(translation_needed)

    # Prepare the response
    response = {
        "phrase": phrase,
        "happiness_level": happiness_level,
        "emoji": emoji,
        'translation_needed' : translation_needed,
        "translation": "Translation: " + phrase_in_english if translation_needed else " "
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='192.168.1.29')
    #app.run()
