import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re
import io

# Configuración de la página
st.set_page_config(page_title="Buscador de ONGs", page_icon="🌍", layout="wide")

# Cargar los datos
@st.cache_data
def load_data():
    sheet_id = "1veX2JY-ovYpubeXf2-uln7pK93Sq8vJT"
    sheet_name = "Resumen L1"

    # URL de exportación a formato Excel
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"

    # Cargar el archivo directamente desde Google Sheets
    sheet_resumen_l1 = pd.read_excel(url, sheet_name=sheet_name, engine='openpyxl')
    sheet_resumen_l1['RESEÑA_CLEAN'] = sheet_resumen_l1['RESEÑA'].apply(clean_text)
    sheet_resumen_l1['PILAR_CLEAN'] = sheet_resumen_l1['PILAR'].apply(clean_text)
    sheet_resumen_l1['POBLACION_CLEAN'] = sheet_resumen_l1['POBLACIÓN BENEFICIARIA'].apply(clean_text)
    sheet_resumen_l1['ZONA_CLEAN'] = sheet_resumen_l1['ZONA IMPACTO'].apply(clean_text)
    sheet_resumen_l1['COMBINED_TEXT'] = sheet_resumen_l1[['RESEÑA_CLEAN', 'PILAR_CLEAN', 'POBLACION_CLEAN','ZONA_CLEAN']].agg(' '.join, axis=1)
    return sheet_resumen_l1

# Limpieza de texto
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
    else:
        text = ""
    return text

# Cargar el modelo de embeddings
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Función para encontrar las ONGs más similares
def find_similar_ongs(description, embeddings, ong_names, combined_texts, model, top_n=5):
    description_embedding = model.encode(description, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(description_embedding, embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_n)
    similar_ongs = []
    for score, idx in zip(top_results.values, top_results.indices):
        similar_ongs.append({
            'ONG': ong_names[idx],
            'Score': score.item(),
            'Description': combined_texts[idx]
        })
    return similar_ongs

# Interfaz de usuario
def main():
    # Estilo general
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f4f8;
            }
            .stButton>button {
                background-color: #FF5733; /* Naranja */
                color: white;
                border-radius: 5px;
                padding: 15px 20px;
                font-size: 16px;
                width: 100%;
            }
            .stDownloadButton>button {
                background-color: #1D428A; /* Azul */
                color: white;
                border-radius: 5px;
                padding: 15px 20px;
                font-size: 16px;
                width: 100%;
            }
            .stMarkdown h1 {
                color: #1D428A; /* Azul */
                font-family: 'Arial', sans-serif;
                font-size: 36px;
                text-align: left;
            }
            .stMarkdown h3 {
                color: #FF5733; /* Naranja */
                font-size: 24px;
                font-weight: bold;
            }
            .logo {
                width: 150px; /* Tamaño adecuado */
                height: auto;
                display: block;
                margin-left: 0;
                margin-right: 20px;
            }
            .result-card {
                background-color: #ffffff;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 15px;
                border: 1px solid #ddd;
            }
            .result-card h4 {
                color: #1D428A; /* Azul */
                font-size: 18px;
            }
            .result-card p {
                color: #555;
                font-size: 14px;
            }
            .result-card .score {
                color: #FF5733; /* Naranja */
                font-weight: bold;
            }
            .result-card .description {
                color: #1D428A; /* Azul */
            }
            .button-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Contenedor de columnas para alinear el logo a la izquierda del título
    col1, col2 = st.columns([1, 4])
    with col1:
        # Logo de OLI redimensionado y alineado a la izquierda
        st.image("Assets/LOGO_Oli.png", use_container_width=True)
    with col2:
        st.title("🔍 Buscador de ONGs Relevantes")
        st.markdown(
            """
            #### Encuentra las organizaciones más relevantes para tus necesidades.  
            #### Solo ingresa una descripción y nosotros haremos el resto.  
            --- 
            """
        )

    # Contenedor de columnas para los botones y la descripción
    col1, col2 = st.columns([2, 1])  # La columna para la descripción toma el doble de espacio
    with col1:
        description = st.text_area(
            "Describe la organización o necesidad que estás buscando:",
            placeholder="Ejemplo: Busco una organización que trabaje con mujeres en áreas rurales.",
            height=150
        )
        # Campo para el número de resultados que el usuario desea ver
        top_n = st.number_input("Número de resultados a mostrar:", min_value=1, max_value=20, value=5, step=1)

    with col2:
        button1 = st.button("Buscar ONGs")
        button2 = st.button("Actualizar Base de Datos")

        # Si se presiona "Actualizar Base de Datos"
        if button2:
            st.cache_data.clear()
            st.success("Base de datos actualizada correctamente.")
            
    # Si se presiona "Buscar ONGs", buscar las organizaciones más relevantes
    if button1:
        if description.strip() != "":
            # Cargar los datos y el modelo
            sheet_resumen_l1 = load_data()
            model = load_model()

            # Embeddings precomputados
            combined_texts = sheet_resumen_l1['COMBINED_TEXT'].tolist()
            ong_names = sheet_resumen_l1['ONG'].tolist()
            embeddings = model.encode(combined_texts, convert_to_tensor=True)

            similar_ongs = find_similar_ongs(description, embeddings, ong_names, combined_texts, model, top_n)
            st.markdown(f"### Resultados más relevantes ({top_n} ONGs):")
            st.markdown("---")

            results = []
            for ong in similar_ongs:
                st.markdown(f"""
                <div class="result-card">
                    <h4>🏢 ONG: {ong['ONG']}</h4>
                    <p class="score"><strong>📊 Puntaje:</strong> {ong['Score']:.4f}</p>
                    <p class="description"><strong>📝 Descripción:</strong> {ong['Description']}</p>
                </div>
                """, unsafe_allow_html=True)
                results.append(ong)

            # Exportar a Excel
            df_results = pd.DataFrame(results)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_results.to_excel(writer, index=False, sheet_name='ONGs Relevantes')
            output.seek(0)  # Asegurarse de mover el puntero al inicio del archivo

            st.download_button(
                label="📥 Descargar resultados en Excel",
                data=output,
                file_name='ongs_recomendadas.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.error("Por favor, ingrese una descripción.")

if __name__ == "__main__":
    main()
