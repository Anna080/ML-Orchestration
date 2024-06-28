from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import logging
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 27),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

timestamp = 10
test_size = 30  # Nombre de jours pour lesquels faire des prédictions

def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

def make_predictions():
    file_path = '/opt/airflow/data/data_normalized.csv'
    model_path = '/opt/airflow/models/stock_model.keras'
    
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        return
    if not os.path.exists(model_path):
        logging.error(f"Model file {model_path} does not exist.")
        return
    
    logging.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    logging.info(f"Model loaded from {model_path}")
    
    df = pd.read_csv(file_path)
    df_train = pd.DataFrame(df.values)
    
    future_day = test_size

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
    
    np.savetxt('/opt/airflow/data/predictions.csv', deep_future, delimiter=',')

dag = DAG(
    'make_predictions',
    default_args=default_args,
    description='DAG for making stock predictions using the trained model',
    schedule_interval=timedelta(days=1),
)

predict_task = PythonOperator(
    task_id='make_predictions_task',
    python_callable=make_predictions,
    dag=dag,
)