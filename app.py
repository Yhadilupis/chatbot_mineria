from flask import Flask, request, jsonify, send_file
import json
import openai
from sentiment_analysis import analyze_sentiment, save_to_mysql
from datetime import datetime
import os

app = Flask(__name__)

# Cargar la clave de API de OpenAI desde las variables de entorno
openai.api_key = os.getenv('OPENAI_API_KEY')

# Inicializar la hora de inicio de la sesión
session_start_time = datetime.now()

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    try:
        data = request.get_json()
        user_input = data.get('msg')
        useruuid = data.get('useruuid', 'default_uuid')

        # Llamada a la API de OpenAI para obtener la respuesta
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Especificar el modelo GPT-4 mini si está disponible
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Extraer la respuesta generada
        generated_text = response.choices[0].message['content'].strip()

        # Análisis de sentimientos y palabras ofensivas
        sentiment_score, sentiment_type, contains_bad_words = analyze_sentiment(user_input)

        # Guardar los resultados en la base de datos MySQL
        save_to_mysql(user_input, sentiment_score, sentiment_type, contains_bad_words, useruuid, session_start_time)

        return jsonify({
            'response': generated_text,
            'contains_bad_words': contains_bad_words
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'response': "Lo siento, ha ocurrido un error."})

@app.route('/predicciones', methods=['GET'])
def obtener_predicciones():
    try:
        return send_file('predicciones.png', mimetype='image/png')
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'response': "Lo siento, ha ocurrido un error al recuperar las predicciones."})

if __name__ == "__main__":
    app.run(debug=False)

