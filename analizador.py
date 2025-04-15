import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pysentimiento import create_analyzer
import requests
import time
from datetime import datetime
import numpy as np
from IPython.display import display

# Define las funciones del extractor y analizador de sentimientos (tal como lo tienes en tu código)

class GoogleMapsReviewExtractor:
    # Código del extractor (sin cambios)
    pass

class SentimentAnalyzer:
    # Código del analizador de sentimientos (sin cambios)
    pass

# Función que se conecta con Streamlit para mostrar los resultados de manera interactiva
def mostrar_analisis_visual(df):
    avg_ratings = df.groupby("institute_name")["rating"].mean().round(2)
    sentiment_counts = df[df["sentiment"] == "bueno"].groupby("institute_name").size()

    st.subheader("Puntuación Media por Instituto")
    st.bar_chart(avg_ratings)

    st.subheader("Instituto con Mejor Puntuación Media")
    mejor_puntuacion = avg_ratings.idxmax()
    st.write(f"🏆 {mejor_puntuacion}: {avg_ratings[mejor_puntuacion]}")

    st.subheader("Distribución de Sentimientos por Instituto")
    sentiment_plot_data = df.groupby(["institute_name", "sentiment"]).size().unstack(fill_value=0)
    st.bar_chart(sentiment_plot_data)

    # Visualización con Plotly
    fig = px.sunburst(
        df,
        path=["institute_name", "sentiment"],
        color="sentiment",
        color_discrete_map={"bueno": "green", "neutro": "gold", "malo": "red"},
        title="Distribución interactiva de sentimientos"
    )
    st.plotly_chart(fig)

# Función para analizar la comunidad y obtener las reseñas
def analizar_comunidad(nombre_comunidad, location=None, max_reviews=10, max_centros=60):
    API_KEY = st.text_input("Introduce tu API Key de Google Maps:")
    
    if not API_KEY:
        st.error("Por favor, ingresa una API Key válida.")
        return None
    
    extractor = GoogleMapsReviewExtractor(API_KEY)
    analyzer = SentimentAnalyzer()

    query_general = f"FP Grado Superior ASIR {nombre_comunidad}"
    st.write(f"🔎 Buscando centros en '{nombre_comunidad}'...")
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
        st.write("❌ No se encontraron reseñas válidas para analizar")
        return None

    df = pd.DataFrame(all_reviews)
    return df

# Función principal de Streamlit
def run_streamlit_app():
    st.title("Análisis de Reseñas de Centros de Formación en FP ASIR")
    
    comunidad = st.text_input("Introduce el nombre de la comunidad", "Madrid")
    
    if st.button("Buscar y Analizar"):
        results_df = analizar_comunidad(comunidad, location="40.4168,-3.7038")  # Coordenadas de Madrid
        
        if results_df is not None:
            mostrar_analisis_visual(results_df)

            # Opción para descargar los resultados como CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Descargar resultados como CSV",
                data=csv,
                file_name="resultados_reseñas.csv",
                mime="text/csv"
            )
        else:
            st.write("No se encontraron resultados. Por favor, prueba con otra comunidad o verifica la API Key.")
    
# Ejecuta la aplicación
if __name__ == "__main__":
    run_streamlit_app()
