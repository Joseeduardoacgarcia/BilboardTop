import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Configuración de la página
st.set_page_config(page_title="Billboard Predictor App", page_icon="🎵", layout="centered")

st.title("🎵 Billboard 100 Predictor & Preview")
st.write("Ingresa el nombre de un artista y una canción para predecir si será un éxito en el Billboard y escuchar un preview.")

# Columnas requeridas por el modelo
columnas_modelo = [
    'tempo', 'durationSeg', 'anioNacimiento', 'chart_year', 'edad_artista',
    'mood_BROODING', 'mood_COOL', 'mood_DEFIANT', 'mood_EASYGOING', 'mood_EMPOWERING',
    'mood_ENERGIZING', 'mood_EXCITED', 'mood_FIERY', 'mood_GRITTY', 'mood_LIVELY',
    'mood_MELANCHOLY', 'mood_OTHER', 'mood_PEACEFUL', 'mood_ROMANTIC', 'mood_ROWDY',
    'mood_SENSUAL', 'mood_SENTIMENTAL', 'mood_SOPHISTICATED', 'mood_STIRRING',
    'mood_UPBEAT', 'mood_URGENT', 'mood_YEARNING', 'genre_ELECTRONICA', 'genre_JAZZ',
    'genre_OTHER', 'genre_POP', 'genre_ROCK', 'genre_SOUNDTRACK', 'genre_TRADITIONAL',
    'genre_URBAN', 'artist_type_MALE', 'artist_type_MIXED'
]

class CancionInfo(BaseModel):
    mood: str = Field(description="Estado de ánimo en mayúsculas (EMPOWERING, COOL, ENERGIZING, YEARNING, DEFIANT, EXCITED, SENSUAL, URGENT, SOPHISTICATED, ROWDY, BROODING, UPBEAT, AGGRESSIVE, MELANCHOLY, GRITTY, ROMANTIC, SENTIMENTAL, FIERY, EASYGOING, PEACEFUL, STIRRING, LIVELY, OTHER)")
    tempo: str = Field(description="Tempo en mayúsculas (SLOW TEMPO, MEDIUM TEMPO o FAST TEMPO)")
    genre: str = Field(description="Género en mayúsculas (URBAN, POP, TRADITIONAL, ALTERNATIVE & PUNK, ELECTRONICA, ROCK, SOUNDTRACK, JAZZ, OTHER)")
    artist_type: str = Field(description="Tipo de artista en mayúsculas (MALE, FEMALE o MIXED)")
    duration_seconds: float = Field(description="Duración en segundos")
    artist_birth_year: float = Field(description="Año aproximado de nacimiento o fundación")
    chart_year: int = Field(description="Año del chart")

def buscar_preview_itunes(cancion, artista):
    """Busca un preview de 30 segundos en la API pública de iTunes."""
    try:
        query = f"{cancion} {artista}"
        url = f"https://itunes.apple.com/search?term={query}&entity=musicTrack&limit=1"
        response = requests.get(url).json()
        if response.get("results"):
            track = response["results"][0]
            return {
                "preview_url": track.get("previewUrl"),
                "artwork_url": track.get("artworkUrl100"),
                "track_name": track.get("trackName"),
                "artist_name": track.get("artistName")
            }
    except Exception as e:
        pass
    return None

# Inputs
artista_input = st.text_input("Nombre del Artista:", "Ed Sheeran")
cancion_input = st.text_input("Nombre de la Canción:", "Shape of You")

if st.button("⚡ Predecir Éxito"):
        with st.spinner("Analizando características musicales con IA..."):
            try:
                client = genai.Client(api_key=api_key)
                prompt = f"Investiga y proporciona las características musicales detalladas de la canción '{cancion_input}' del artista '{artista_input}'. Sé realista."
                
                response = client.models.generate_content(
                    model='gemini-3.5-flash-lite',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CancionInfo,
                        temperature=0.1
                    ),
                )
                info = CancionInfo.model_validate_json(response.text)
                
                # Cargar el modelo
                modelo = joblib.load('modelo_random_forest.joblib')
                
                # Procesar entrada
                df_temp = pd.DataFrame([{
                    'tempo': info.tempo.strip().upper(),
                    'durationSeg': info.duration_seconds,
                    'anioNacimiento': info.artist_birth_year,
                    'chart_year': info.chart_year,
                    'edad_artista': info.chart_year - info.artist_birth_year,
                    'mood': info.mood.strip().upper(),
                    'genre': info.genre.strip().upper(),
                    'artist_type': info.artist_type.strip().upper()
                }])

                orden_tempo = {"SLOW TEMPO": 0, "MEDIUM TEMPO": 1, "FAST TEMPO": 2}
                df_temp["tempo"] = df_temp["tempo"].map(orden_tempo).fillna(1)
                df_encoded = pd.get_dummies(df_temp, columns=["mood", "genre", "artist_type"])

                for col in columnas_modelo:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0

                df_input = df_encoded[columnas_modelo].astype(float)
                prediccion = modelo.predict(df_input)[0]
                probabilidades = modelo.predict_proba(df_input)[0]

                # Mostrar Resultados de la Predicción
                st.subheader("📊 Diagnóstico del Modelo")
                if prediccion == 1:
                    st.success(f"🔥 ¡SÍ ENTRARÁ AL BILLBOARD! (Probabilidad: {probabilidades[1]*100:.1f}%)")
                else:
                    st.warning(f"❌ NO ENTRARÁ AL BILLBOARD (Probabilidad de entrar: {probabilidades[1]*100:.1f}%)")
                
                # Mostrar Datos Extraídos por la IA
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Género:** {info.genre}")
                    st.write(f"**Estado de ánimo:** {info.mood}")
                    st.write(f"**Tempo:** {info.tempo}")
                with col2:
                    st.write(f"**Duración:** {info.duration_seconds:.1f} segundos")
                    st.write(f"**Año Nacimiento Artista:** {info.artist_birth_year:.0f}")
                    st.write(f"**Edad del Artista:** {info.chart_year - info.artist_birth_year:.0f} años")

            except Exception as e:
                st.error(f"Error en la predicción: {e}")

        # Buscar Preview de la Canción
        with st.spinner("Buscando audio de previsualización..."):
            preview_data = buscar_preview_itunes(cancion_input, artista_input)
            if preview_data:
                st.markdown("--- ")
                st.subheader("🎵 Escuchar Preview")
                col_img, col_audio = st.columns([1, 3])
                with col_img:
                    if preview_data["artwork_url"]:
                        st.image(preview_data["artwork_url"], width=120)
                with col_audio:
                    st.write(f"**{preview_data['track_name']}** - *{preview_data['artist_name']}*")
                    st.audio(preview_data["preview_url"])
            else:
                st.info("No se pudo encontrar una previsualización de audio para esta canción en este momento.")
