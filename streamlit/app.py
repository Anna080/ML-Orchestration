import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Visualisation des Données")

# Charger les données
data_csv = pd.read_csv('/app/data/data.csv')
predictions_csv = pd.read_csv('/app/data/predictions.csv')

# Afficher les données
st.write("### Données d'origine")
st.write(data_csv.head())

st.write("### Prédictions")
st.write(predictions_csv.head())

# Tracer les graphiques
st.write("### Graphique des Données d'origine")
fig, ax = plt.subplots()
data_csv.plot(ax=ax)
st.pyplot(fig)

st.write("### Graphique des Prédictions")
fig, ax = plt.subplots()
predictions_csv.plot(ax=ax)
st.pyplot(fig)
