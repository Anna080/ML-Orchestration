from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pandas as pd
import os
import logging
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.operators.dummy_operator import DummyOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 27),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

def preprocess_data():
    file_path = '/opt/airflow/data/data.csv'
    if os.path.exists(file_path):
        logging.info(f"Reading data from {file_path}")
        df = pd.read_csv(file_path)
        minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32'))  # Assuming column index 4 is Close price
        df_log = minmax.transform(df.iloc[:, 4:5].astype('float32'))
        output_file = '/opt/airflow/data/data_normalized.csv'
        pd.DataFrame(df_log, columns=['Normalized_Close']).to_csv(output_file, index=False)
        logging.info(f"Normalized data saved to {output_file}")
    else:
        logging.error(f"File {file_path} does not exist.")

dag = DAG(
    'preprocess_data',
    default_args=default_args,
    description='DAG for preprocessing stock data',
    schedule_interval=timedelta(days=1),
)

preprocess_task = PythonOperator(
    task_id='preprocess_stock_data',
    python_callable=preprocess_data,
    dag=dag,
)

