from datetime import datetime
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

bad_words = ["mierda", "idiota", "estúpido", "imbécil", "maldito", "basura", "no sirve", "no me gusta"]

from transformers import pipeline
sentiment_analysis = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

def analyze_sentiment(text):
    result = sentiment_analysis(text)[0]
    sentiment_label = result['label']
    sentiment_score = result['score']
    
    if '1 star' in sentiment_label or '2 stars' in sentiment_label:
        sentiment_type = 'Negativo'
        sentiment_scale = 1
    elif '3 stars' in sentiment_label:
        sentiment_type = 'Neutral'
        sentiment_scale = 3
    else:
        sentiment_type = 'Positivo'
        sentiment_scale = 5
    
    contains_bad_words = any(bad_word in text.lower() for bad_word in bad_words)
    
    return sentiment_scale, sentiment_type, contains_bad_words

def save_to_mysql(user_input, sentiment_score, sentiment_type, contains_bad_words, useruuid, session_start_time):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    mysql_user = os.getenv('MYSQL_USER')
    mysql_password = os.getenv('MYSQL_PASSWORD')
    mysql_host = os.getenv('MYSQL_HOST')
    mysql_database = os.getenv('MYSQL_DATABASE')
    mysql_table = os.getenv('MYSQL_TABLE')
    mysql_port = os.getenv('MYSQL_PORT')
    
    cantidad_mensajes = len(user_input.split())  
    duracion_sesion = (datetime.now() - session_start_time).total_seconds() / 60.0 
    
    cnx = mysql.connector.connect(user=mysql_user, password=mysql_password,
                                  host=mysql_host, database=mysql_database,
                                  port=mysql_port)
    cursor = cnx.cursor()

    crear_tabla = f"""
    CREATE TABLE IF NOT EXISTS {mysql_table} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        fecha DATETIME,
        useruuid VARCHAR(255),
        cantidad_mensajes INT DEFAULT 1,
        duracion_sesion FLOAT DEFAULT 0.0,
        calificacion FLOAT,
        tipo_sentimiento VARCHAR(255),
        contiene_groserias BOOLEAN
    );
    """
    cursor.execute(crear_tabla)

    insertar_datos = f"""
    INSERT INTO {mysql_table} (fecha, useruuid, cantidad_mensajes, duracion_sesion, calificacion, tipo_sentimiento, contiene_groserias)
    VALUES (%s, %s, %s, %s, %s, %s, %s);
    """
    cursor.execute(insertar_datos, (date, useruuid, cantidad_mensajes, duracion_sesion, sentiment_score, sentiment_type, contains_bad_words))
    cnx.commit()

    cursor.close()
    cnx.close()
