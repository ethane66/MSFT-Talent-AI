# streamlit_reseñas_asir.py

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import requests
import time
from pysentimiento import create_analyzer
import streamlit as st

# ---------------------
# Clases y Funciones
# ---------------------

class GoogleMapsReviewExtractor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        self.details_url = "https://maps.googleapis.com/maps/api/place/details/json"

    def _make_api_request(self, url, params):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return None

    def search_institutes(self, query, location=None, radius=10000, max_results=60):
        params = {
            "query": query,
            "key": self.api_key,
            "language": "es",
            "region": "es"
        }
        if location:
            params["location"] = location
            params["radius"] = radius

        all_results = []
        while True:
            data = self._make_api_request(self.base_url, params)
            if not data:
                break

            results = data.get("results", [])
            all_results.extend(results)

            if len(all_results) >= max_results:
                break

            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

            time.sleep(2)
            params = {
                "pagetoken": next_page_token,
                "key": self.api_key
            }

        return all_results[:max_results]

    def get_reviews(self, place_id, max_reviews=10):
        params = {
            "place_id": place_id,
            "key": self.api_key,
            "fields": "review,name,rating",
            "language": "es"
        }

        data = self._make_api_request(self.details_url, params)
        if not data or "result" not in data:
            return []

        result = data["result"]
        reviews = result.get("reviews", [])[:max_reviews]

        return [ {
            "text": review.get("text", ""),
            "rating": review.get("rating", 0),
            "time": datetime.fromtimestamp(review.get("time", 0)),
            "institute_name": result.get("name", "")
        } for review in reviews ]

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = create_analyzer(task="sentiment", lang="es")

    def analyze_review(self, text):
        try:
            result = self.analyzer.predict(text)
            sentiment_label = result.output
            if sentiment_label == "POS":
                sentiment = "bueno"
            elif sentiment_label == "NEU":
                sentiment = "neutro"
            else:
                sentiment = "malo"
            return {
                "sentiment": sentiment,
                "confidence": result.probas[sentiment_label]
            }
        except Exception as e:
            st.warning(f"Error en análisis de sentimiento: {str(e)}")
            return None

def analizar_comunidad(nombre_comunidad, api_key, location=None, max_reviews=10, max_centros=60):
    extractor = GoogleMapsReviewExtractor(api_key)
    analyzer = SentimentAnalyzer()

    query_general = f"FP Grado Superior ASIR {nombre_comunidad}"
    st.info(f"🔍 Buscando centros en '{nombre_comunidad}'...")
    resultados = extractor.search_institutes(query_general, location=location, max_results=max_centros)

    all_reviews = []

    for centro in resultados:
        nombre = centro.get("name")
        place_id = centro.get("place_id")
        if not place_id:
            continue

        reviews = extractor.get_reviews(place_id, max_reviews)
        if not reviews:
            continue

        st.write(f"📝 Analizando {len(reviews)} reseñas de '{nombre}'...")
        for review in reviews:
            analysis = analyzer.analyze_review(review["text"])
            if analysis:
                review.update(analysis)
                all_reviews.append(review)

    if not all_reviews:
        st.warning("❌ No se encontraron reseñas válidas.")
        return None

    return pd.DataFrame(all_reviews)

def mostrar_analisis_visual(df):
    st.subheader("📊 Resultados del análisis")

    avg_ratings = df.groupby("institute_name")["rating"].mean().round(2)
    mejor_puntuacion = avg_ratings.idxmax()
    st.success(f"🏆 Instituto con mejor puntuación media: **{mejor_puntuacion}** ({avg_ratings[mejor_puntuacion]})")

    sentiment_counts = df[df["sentiment"] == "bueno"].groupby("institute_name").size()
    if not sentiment_counts.empty:
        mejor_sentimiento = sentiment_counts.idxmax()
        st.success(f"💚 Instituto con más reseñas positivas: **{mejor_sentimiento}** ({sentiment_counts[mejor_sentimiento]})")

    st.bar_chart(avg_ratings)

    sentiment_plot_data = df.groupby(["institute_name", "sentiment"]).size().unstack(fill_value=0)
    st.bar_chart(sentiment_plot_data)

    fig = px.sunburst(
        df,
        path=["institute_name", "sentiment"],
        color="sentiment",
        color_discrete_map={"bueno": "green", "neutro": "gold", "malo": "red"},
        title="Distribución de sentimientos por instituto"
    )
    st.plotly_chart(fig)

    st.subheader("📋 Tabla de reseñas")
    st.dataframe(df[["institute_name", "text", "sentiment", "rating"]])

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar CSV", data=csv, file_name="analisis_reseñas.csv", mime="text/csv")

# ---------------------
# Interfaz Streamlit
# ---------------------

st.set_page_config(page_title="Análisis de Reseñas ASIR", layout="wide")
st.title("🔎 Análisis de Reseñas de Centros FP ASIR")

api_key = st.text_input("🔑 Ingresa tu Google Maps API Key", type="password")
comunidad = st.text_input("🏙️ Comunidad Autónoma", "Madrid")
location = st.text_input("📍 Coordenadas (opcional)", "40.4168,-3.7038")

if st.button("Analizar"):
    if not api_key:
        st.error("⚠️ Por favor, introduce tu Google Maps API Key.")
    else:
        df = analizar_comunidad(comunidad, api_key, location)
        if df is not None:
            mostrar_analisis_visual(df)
