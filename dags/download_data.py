from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import logging

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 27),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

def download_data(symbol, period):
    logging.info(f'Starting data download for {symbol} for period {period}')
    yf.pdr_override()
    df = pdr.get_data_yahoo(symbol, period=period, interval="1d")
    logging.info(f'Data download complete, number of rows downloaded: {len(df)}')
    df.to_csv('/opt/airflow/data/data.csv')
    logging.info('Data saved to /opt/airflow/data/data.csv')

dag = DAG(
    'download_data',
    default_args=default_args,
    description='DAG for downloading stock data',
    schedule_interval=timedelta(days=1),
)

download_task = PythonOperator(
    task_id='download_stock_data',
    python_callable=download_data,
    op_kwargs={'symbol': 'AAPL', 'period': '1y'},
    dag=dag,
)
