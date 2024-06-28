FROM python:3.12

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    libhdf5-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Définir le répertoire de travail
WORKDIR /appli

# Copier les fichiers nécessaires dans le conteneur
COPY requirements.txt .
COPY app.py .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par l'application FastAPI
EXPOSE 5001

# Commande pour lancer l'application FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]