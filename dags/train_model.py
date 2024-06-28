from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 27),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

num_layers = 1
size_layer = 128
timestamp = 10
epoch = 300
dropout_rate = 0.8
learning_rate = 0.01

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

def train():
    file_path = '/opt/airflow/data/data_normalized.csv'
    model_dir = '/opt/airflow/models/'
    model_save_path = os.path.join(model_dir, 'stock_model.keras')
    
    # Check if the model directory exists, if not create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logging.info(f"Created directory {model_dir}")
    
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        return
    
    df = pd.read_csv(file_path)
    df_train = pd.DataFrame(df.values)
    
    model = create_model((timestamp, df_train.shape[1]), df_train.shape[1], learning_rate)

    data = df_train.values
    X, Y = [], []
    for i in range(len(data) - timestamp):
        X.append(data[i:i + timestamp])
        Y.append(data[i + timestamp])
    
    X = np.array(X)
    Y = np.array(Y)

    model.fit(X, Y, epochs=epoch, batch_size=64, verbose=1)
    
    logging.info(f"Saving model to {model_save_path}")
    model.save(model_save_path)
    logging.info(f"Model saved to {model_save_path}")

dag = DAG(
    'train_model',
    default_args=default_args,
    description='DAG for training stock prediction model',
    schedule_interval=timedelta(days=1),
)

train_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train,
    dag=dag,
)