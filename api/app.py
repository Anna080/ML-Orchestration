from fastapi import FastAPI
import redis
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import logging

app = FastAPI()

# Configurer Redis
cache = redis.Redis(host='redis', port=6379)

# Configurer les paramètres du modèle
num_layers = 1
size_layer = 128
timestamp = 10
dropout_rate = 0.8
learning_rate = 0.01

model_path = '/appli/models/stock_model.keras'
data_path = '/appli/data/data_normalized.csv'

def create_model(input_shape, output_size, learning_rate):
    model = tf.keras.Sequential()
    for _ in range(num_layers):
        model.add(tf.keras.layers.LSTM(size_layer, return_sequences=True, input_shape=input_shape))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.LSTM(size_layer))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(output_size))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
    return model

def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Prediction API"}

@app.get("/predict")
def get_predictions():
    # Vérifier si les prédictions sont déjà en cache
    cached_predictions = cache.get('predictions')
    if cached_predictions:
        # Si oui, retourner les prédictions en cache
        predictions = np.frombuffer(cached_predictions, dtype=np.float64)
        return {"predicted_close": predictions.tolist()}

    # Sinon, calculer les prédictions
    predictions = compute_predictions()

    if predictions:
        # Mettre en cache les prédictions
        cache.set('predictions', np.array(predictions).tobytes())

    return {"predicted_close": predictions}

def compute_predictions():
    if not os.path.exists(data_path):
        logging.error(f"File {data_path} does not exist.")
        return []
    if not os.path.exists(model_path):
        logging.error(f"Model file {model_path} does not exist.")
        return []

    df = pd.read_csv(data_path)
    df_train = pd.DataFrame(df.values)
    
    model = tf.keras.models.load_model(model_path)
    
    future_day = 30  # Nombre de jours pour lesquels faire des prédictions

    data = df_train.values
    output_predict = np.zeros((len(data) + future_day, df_train.shape[1]))
    output_predict[:len(data)] = data
    
    for i in range(len(data) - timestamp):
        output_predict[i + timestamp] = model.predict(data[i:i + timestamp].reshape(1, timestamp, -1))
    
    for i in range(future_day):
        output_predict[len(data) + i] = model.predict(output_predict[len(data) + i - timestamp:len(data) + i].reshape(1, timestamp, -1))
    
    minmax = MinMaxScaler().fit(df.values.astype('float32'))
    output_predict = minmax.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], 0.4)
    
    return deep_future