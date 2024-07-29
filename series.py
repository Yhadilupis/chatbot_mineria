import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from statsmodels.tsa.stattools import adfuller
import numpy as np

load_dotenv()

user = os.getenv('MYSQL_USER')
password = os.getenv('MYSQL_PASSWORD')
host = os.getenv('MYSQL_HOST')
database = os.getenv('MYSQL_DATABASE')
table = os.getenv('MYSQL_TABLE')

connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}"
engine = create_engine(connection_string)

query = f"""
SELECT fecha, cantidad_mensajes, tipo_sentimiento, calificacion
FROM {table}
"""

# Cargar los datos en un DataFrame
data = pd.read_sql(query, engine)

data['fecha'] = pd.to_datetime(data['fecha'])

data.set_index('fecha', inplace=True)

# Ordenar los datos por la fecha
data = data.sort_index()

# Convertir el tipo de sentimiento a valores numéricos para análisis
sentiment_mapping = {'Positivo': 1, 'Neutral': 0, 'Negativo': -1}
data['sentimiento_numerico'] = data['tipo_sentimiento'].map(sentiment_mapping)

# Agrupar datos por día
data = data.resample('D').agg({
    'cantidad_mensajes': 'sum',
    'sentimiento_numerico': 'mean',
    'calificacion': 'mean'
})

#ajuste de frecuencia
data = data.asfreq('D')

#valors Nan
data = data.dropna()

# Dividir los datos en conjuntos de entrenamiento y prueba
if len(data) > 30:
    train = data.iloc[:-30]
    test = data.iloc[-30:]
else:
    train = data
    test = pd.DataFrame()

# Verificar estacionariedad
result_sentimiento = adfuller(train['sentimiento_numerico'].dropna())
result_calificacion = adfuller(train['calificacion'].dropna())
print('p-value Sentimiento Promedio:', result_sentimiento[1])
print('p-value Calificación:', result_calificacion[1])

try:
    if len(train) > 0:
        print("Aplicando suavizado exponencial simple para Sentimiento Promedio...")
        model_se_sentimiento = ExponentialSmoothing(train['sentimiento_numerico'], trend=None, seasonal=None)
        se_fit_sentimiento = model_se_sentimiento.fit()
        # Predicción para el conjunto de prueba
        se_forecast_sentimiento = se_fit_sentimiento.forecast(steps=len(test))
        # Predicción para los próximos 7 días
        se_forecast_sentimiento_7d = se_fit_sentimiento.forecast(steps=7)
        mse_se_sentimiento = mean_squared_error(test['sentimiento_numerico'], se_forecast_sentimiento) if not test.empty else None
        print(f'Mean Squared Error (Simple Exponential Smoothing - Sentimiento Promedio): {mse_se_sentimiento}')
    else:
        print("No hay suficientes datos para aplicar suavizado exponencial simple para Sentimiento Promedio.")
except Exception as e:
    print(f'Error en el modelo de suavizado exponencial simple para Sentimiento Promedio: {e}')

try:
    if len(train) > 0 and len(train) > 5:
        print("Aplicando ARIMA para Sentimiento Promedio...")
        model_arima_sentimiento = ARIMA(train['sentimiento_numerico'].dropna(), order=(1,1,1))
        arima_fit_sentimiento = model_arima_sentimiento.fit(method_kwargs={"warn_convergence": False})
        # Predicción para el conjunto de prueba
        arima_forecast_sentimiento = arima_fit_sentimiento.forecast(steps=len(test))
        # Predicción para los próximos 7 días
        arima_forecast_sentimiento_7d = arima_fit_sentimiento.forecast(steps=7)
        mse_arima_sentimiento = mean_squared_error(test['sentimiento_numerico'], arima_forecast_sentimiento) if not test.empty else None
        print(f'Mean Squared Error (ARIMA - Sentimiento Promedio): {mse_arima_sentimiento}')
    else:
        print("No hay suficientes datos para aplicar ARIMA para Sentimiento Promedio.")
except Exception as e:
    print(f'Error en el modelo ARIMA para Sentimiento Promedio: {e}')

try:
    if len(train) > 0:
        print("Aplicando suavizado exponencial simple para Calificación...")
        model_se_calificacion = ExponentialSmoothing(train['calificacion'].dropna(), trend=None, seasonal=None)
        se_fit_calificacion = model_se_calificacion.fit()
        # Predicción para el conjunto de prueba
        se_forecast_calificacion = se_fit_calificacion.forecast(steps=len(test))
        # Predicción para los próximos 7 días
        se_forecast_calificacion_7d = se_fit_calificacion.forecast(steps=7)
        mse_se_calificacion = mean_squared_error(test['calificacion'], se_forecast_calificacion) if not test.empty else None
        print(f'Mean Squared Error (Simple Exponential Smoothing - Calificación): {mse_se_calificacion}')
    else:
        print("No hay suficientes datos para aplicar suavizado exponencial simple para Calificación.")
except Exception as e:
    print(f'Error en el modelo de suavizado exponencial simple para Calificación: {e}')

try:
    if len(train) > 0 and len(train) > 5:
        print("Aplicando ARIMA para Calificación...")
        model_arima_calificacion = ARIMA(train['calificacion'].dropna(), order=(1,1,1))
        arima_fit_calificacion = model_arima_calificacion.fit(method_kwargs={"warn_convergence": False})
        # Predicción para el conjunto de prueba
        arima_forecast_calificacion = arima_fit_calificacion.forecast(steps=len(test))
        # Predicción para los próximos 7 días
        arima_forecast_calificacion_7d = arima_fit_calificacion.forecast(steps=7)
        mse_arima_calificacion = mean_squared_error(test['calificacion'], arima_forecast_calificacion) if not test.empty else None
        print(f'Mean Squared Error (ARIMA - Calificación): {mse_arima_calificacion}')
    else:
        print("No hay suficientes datos para aplicar ARIMA para Calificación.")
except Exception as e:
    print(f'Error en el modelo ARIMA para Calificación: {e}')

plt.figure(figsize=(14, 12))

plt.subplot(2, 1, 1)
plt.plot(data.index, data['sentimiento_numerico'], label='Sentimiento Promedio Original')
if not test.empty:
    plt.plot(test.index, se_forecast_sentimiento, label='Simple Exponential Smoothing Forecast (Sentimiento)', linestyle='--')
    plt.plot(test.index, arima_forecast_sentimiento, label='ARIMA Forecast (Sentimiento)', linestyle='--')

#predicciones de 7 dias 
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
plt.plot(future_dates, se_forecast_sentimiento_7d, label='Forecast 7 días (Simple Exponential Smoothing - Sentimiento)', linestyle='--', color='orange')
plt.plot(future_dates, arima_forecast_sentimiento_7d, label='Forecast 7 días (ARIMA - Sentimiento)', linestyle='--', color='red')
plt.legend(loc='best')
plt.xlabel('Fecha')
plt.ylabel('Sentimiento Promedio')
plt.title('Predicciones de Sentimiento Promedio')

#visualizar las predicciones
plt.subplot(2, 1, 2)
plt.plot(data.index, data['calificacion'], label='Calificación Original')
if not test.empty:
    plt.plot(test.index, se_forecast_calificacion, label='Simple Exponential Smoothing Forecast (Calificación)', linestyle='--')
    plt.plot(test.index, arima_forecast_calificacion, label='ARIMA Forecast (Calificación)', linestyle='--')

#predicciones de los próximos 7 días
plt.plot(future_dates, se_forecast_calificacion_7d, label='Forecast 7 días (Simple Exponential Smoothing - Calificación)', linestyle='--', color='orange')
plt.plot(future_dates, arima_forecast_calificacion_7d, label='Forecast 7 días (ARIMA - Calificación)', linestyle='--', color='red')
plt.legend(loc='best')
plt.xlabel('Fecha')
plt.ylabel('Calificación')
plt.title('Predicciones de Calificación')

plt.tight_layout()
plt.savefig('predicciones.png')
