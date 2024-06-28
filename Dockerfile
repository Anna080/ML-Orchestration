FROM apache/airflow:2.9.2

USER root

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    libhdf5-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Mettre à jour pip
RUN pip install --upgrade pip

# Copier le fichier de dépendances et installer les packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les DAGs
COPY ./dags /opt/airflow/dags

# Exposer le port pour Flask
EXPOSE 5001