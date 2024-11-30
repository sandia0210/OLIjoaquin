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
            .stButton>button {
                background-color: #4CAF50; 
                color: white; 
                border-radius: 5px; 
                padding: 10px 20px;
            }
            .stDownloadButton>button {
                background-color: #FF5733;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
            }
            .stMarkdown h1 {
                color: #2E8B57;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🌟 Buscador de ONGs Relevantes")
    st.markdown(
        """
        #### Bienvenido al Buscador de ONGs 🌍  
        Encuentra las organizaciones más relevantes para tus necesidades.  
        Solo ingresa una descripción y nosotros haremos el resto.  
        ---
        """
    )

    # Botón para actualizar la base de datos
    with st.sidebar:
        st.header("🔄 Configuración")
        if st.button("Actualizar Base de Datos"):
            st.cache_data.clear()
            st.success("Base de datos actualizada correctamente.")

    # Cargar los datos y modelo
    sheet_resumen_l1 = load_data()
    model = load_model()

    # Embeddings precomputados
    combined_texts = sheet_resumen_l1['COMBINED_TEXT'].tolist()
    ong_names = sheet_resumen_l1['ONG'].tolist()
    embeddings = model.encode(combined_texts, convert_to_tensor=True)

    # Entrada del usuario
    st.subheader("🔍 Buscar ONGs")
    description = st.text_area(
        "Describe la organización o necesidad que estás buscando:",
        placeholder="Ejemplo: Busco una organización que trabaje con mujeres en áreas rurales."
    )

    # Selección del Top N
    top_n = st.number_input(
        "Selecciona cuántas ONGs relevantes deseas (Top N):",
        min_value=1,
        max_value=50,
        value=5,
        step=1
    )

    # Buscar ONGs relevantes
    if st.button("Buscar ONGs"):
        if description.strip() != "":
            similar_ongs = find_similar_ongs(description, embeddings, ong_names, combined_texts, model, top_n=top_n)
            st.markdown("### Resultados más relevantes:")
            st.markdown("---")

            results = []
            for ong in similar_ongs:
                st.markdown(f"""
                <div style="background-color:#f9f9f9;padding:10px;margin-bottom:10px;border-radius:5px;border:1px solid #ddd;">
                    <strong>🏢 ONG:</strong> {ong['ONG']}<br>
                    <strong>📊 Puntaje:</strong> {ong['Score']:.4f}<br>
                    <strong>📝 Descripción:</strong> {ong['Description']}
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
