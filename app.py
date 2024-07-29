from flask import Flask, request, jsonify, render_template
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import random
from sentiment_analysis import analyze_sentiment, save_to_mysql
from datetime import datetime

app = Flask(__name__)

# Carga el modelo y otros archivos
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents_spanish.json', 'r', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Inicializar la hora de inicio de la sesión
session_start_time = datetime.now()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Lo siento, no pude encontrar una respuesta para tu pregunta."

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['response'])

    return "Lo siento, no tengo una respuesta para eso."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    try:
        user_input = request.args.get('msg')
        useruuid = request.args.get('useruuid', 'default_uuid')  # Obtener el UUID del usuario o usar un valor por defecto
        intents_list = predict_class(user_input)
        response = get_response(intents_list, intents)

        # Realizar análisis de sentimientos
        sentiment_score, sentiment_type, contains_bad_words = analyze_sentiment(user_input)

        # Guardar en MySQL
        save_to_mysql(user_input, sentiment_score, sentiment_type, contains_bad_words, useruuid, session_start_time)

        return jsonify({'response': response, 'contains_bad_words': contains_bad_words})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'response': "Lo siento, ha ocurrido un error."})

if __name__ == "__main__":
    app.run(debug=False)  # Desactivar el modo de depuración para evitar problemas con werkzeug
