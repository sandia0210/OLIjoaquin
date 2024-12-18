import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re
import io

# Configuración de la página
st.set_page_config(page_title="Buscador de ONGs", page_icon="🌍", layout="wide")

# Función para cargar datos desde Google Sheets
@st.cache_data
def load_data():
    sheet_id = "1mrCQubiQSk_Ilv9a5gbezH_zyu81DjCr_G1sBHZ93Lg"
    sheet_name = "Respuestas de formulario 1"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    df = pd.read_excel(url, sheet_name=sheet_name, engine='openpyxl')

    # Filtro por iniciativas activas
    df = df[df['¿Su iniciativa se encuentra activa?'].str.lower() != 'no']

    # Cambiar nombres
    df['Objetivo_General'] = df['En 50 palabras o menos, por favor, describe el objetivo general de la iniciativa/programa']
    df['Responsable'] = df['Nombre del Responsable'] + ' - ' + df['Cargo del Responsable']
    df['Contacto'] = df['Número de Contacto (Ejemplo: (+51) 949972341)']
    df['Facebook'] = df['Facebook de la organización (link)']
    df['Instagram'] = df['Instagram de la organización (link)']
    df['Página_web'] = df['Página web de la organización (link)']

    # Limpieza de texto relevante
    df['COMUNIDAD_CLEAN'] = df['Tipo de comunidad/población más beneficiado por las actividades de la organización']

    # Crear lista de departamentos con manejo de nulos
    df['Departamentos_lista'] = df.apply(
        lambda row: row['¿En qué departamentos del Perú operan?'].strip().split(', ')
        if pd.notna(row['¿En qué departamentos del Perú operan?']) and row['¿En qué parte del Perú opera la organización?'] == 'Operá en dos o más departamentos'
        else [row['¿En qué parte del Perú opera la organización?'].strip()]
        if pd.notna(row['¿En qué parte del Perú opera la organización?']) else [],
        axis=1
    )

    # Combinar prioridades en una sola columna
    prioridad_cols = [col for col in df.columns if col.startswith('Prioridad')]

    df['PRIORIDADES'] = (
        df[prioridad_cols]
        .apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    # Otras columnas combinadas
    df['DESCRIPCION'] = df['En 50 palabras o menos, por favor, describe el objetivo general de la iniciativa/programa'].fillna("").apply(clean_text)
    df['OBJETIVO_LARGO'] = df['En 50 palabras o menos, por favor, describe el objetivo a largo plazo de la iniciativa/programa'].fillna("").apply(clean_text)
    df['OBJETIVO_CORTO'] = df['En 50 palabras o menos, por favor, describe el objetivo a corto plazo de la iniciativa/programa'].fillna("").apply(clean_text)
    df['COMBINED_TEXT'] = df[['DESCRIPCION', 'OBJETIVO_LARGO', 'OBJETIVO_CORTO']].agg(' '.join, axis=1)

    return df

# Función para limpiar texto
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower().strip()
    else:
        text = ""
    return text

# Función para cargar el modelo de embeddings
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/LaBSE')

# Función para filtrar los datos según los criterios seleccionados
def filter_data(df, comunidades_seleccionadas, departamentos_seleccionados):
    if comunidades_seleccionadas:
        pattern = '|'.join([re.escape(c) for c in comunidades_seleccionadas])
        df = df[df['COMUNIDAD_CLEAN'].str.contains(pattern, case=False, na=False)]
    if departamentos_seleccionados:
        df = df[df['Departamentos_lista'].apply(lambda x: any(dep in x for dep in departamentos_seleccionados))]
    return df

# Función para encontrar ONGs similares
def find_similar_ongs(description, embeddings, ong_names, combined_texts, model, top_n=5):
    if len(embeddings) == 0:
        return []
    description_embedding = model.encode(description, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(description_embedding, embeddings)[0]
    top_results = torch.topk(cosine_scores, k=min(top_n, len(ong_names)))
    similar_ongs = []
    for score, idx in zip(top_results.values, top_results.indices):
        similar_ongs.append({
            'ONG': ong_names[idx],
            'Score': score.item()
        })
    return similar_ongs

# Interfaz principal
def main():

    # Estilo general
    st.markdown(
        """
        <style>
            .stButton>button {
                background-color: #E55200; 
                color: white; 
                border-radius: 5px; 
                padding: 10px 20px;
            }
            .stDownloadButton>button {
                background-color: #E55200;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
            }
            .stMarkdown h1 {
                color: #E55200;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Estilos CSS personalizados para el multiselect
    custom_css = """
    <style>
        .stMultiSelect [role="option"] {
            color: white; /* Color de texto de las opciones */
            font-weight: bold; /* Negrita para destacar */
        }
        .stMultiSelect .st-af { /* Opciones seleccionadas */
            background-color: #E55200; /* Fondo azul claro */
            color: white; /* Texto azul oscuro */
        }
    </style>
    """
    # Renderizamos los estilos en la app
    st.markdown(custom_css, unsafe_allow_html=True)


    # CSS personalizado para cambiar el color del sidebar
    sidebar_style = """
    <style>
        [data-testid="stSidebar"] {
            background-color: #A8B8C7; /* Fondo azul calmado */
            color: #002857; /* Texto en azul oscuro */
        }
        [data-testid="stSidebar"] .css-1d391kg {
            color: #002857; /* Color de texto en los inputs */
        }
        [data-testid="stSidebar"] .css-17eq0hr {
            color: #E55200; /* Color de los títulos al naranja vibrante */
        }
    </style>
    """

    # Aplicar los estilos en la app
    st.markdown(sidebar_style, unsafe_allow_html=True)
    
    st.markdown("<h1 style='color: #001d57';'>Buscador de ONGs Relevantes</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #001d57';'>Encuentra las organizaciones más relevantes según tus necesidades.</p>", unsafe_allow_html=True)

    
    with st.sidebar:
        st.image("Assets/LOGO_Oli.png", use_container_width=True)
        st.markdown("<h2 style='color: #001d57';'>↻ Configuración</h2>", unsafe_allow_html=True)
        if st.button("Actualizar Base de Datos"):
            st.cache_data.clear()
            st.success("Base de datos actualizada correctamente.")

    df = load_data()
    model = load_model()

    st.markdown("<h3 style='color: #001d57;'>↻ Filtros Iniciales</h3>", unsafe_allow_html=True)
    comunidades_seleccionadas = st.multiselect(
        "Seleccione el tipo de población beneficiaria:",

        options=df["Tipo de comunidad/población más beneficiado por las actividades de la organización"].unique().tolist(),
        placeholder='Selecciona una o varias opciones'
    )
    departamentos_seleccionados = st.multiselect(
        "Seleccione el departamento de operación:",
        options=[
            "Amazonas", "Áncash", "Apurimac", "Arequipa", "Ayacucho", "Cajamarca", "Callao",
            "Cusco", "Huancavelica", "Huánuco", "Ica", "Junín", "La Libertad", "Lambayeque",
            "Lima", "Loreto", "Madre de Dios", "Moquegua", "Pasco", "Piura", "Puno",
            "San Martín", "Tacna", "Tumbes", "Ucayali"
        ],
        placeholder='Selecciona una o varias opciones'
    )

    df_filtrado = filter_data(df, comunidades_seleccionadas, departamentos_seleccionados)

    if df_filtrado.empty:
        st.error("No results match your filter criteria. Please adjust the filters.")
        return

    combined_texts = df_filtrado['COMBINED_TEXT'].tolist()
    ong_names = df_filtrado['Nombre de la Iniciativa/Organización'].tolist()
    embeddings = model.encode(combined_texts, convert_to_tensor=True)

    st.markdown("<h3 style='color: #001d57;'>🔍 Buscar ONGs</h3>", unsafe_allow_html=True)
    description = st.text_area("Describe lo que buscas:", placeholder="ONG que trabaje con niños con cancer.")

    top_n = st.number_input(
        "Selecciona cuántas ONGs relevantes deseas (Top N):", 
        min_value=1, max_value=50, value=5, step=1
    )

    if st.button("Buscar ONGs"):
        if description.strip() != "":
            similar_ongs = find_similar_ongs(description, embeddings, ong_names, combined_texts, model, top_n=top_n)

            if not similar_ongs:
                st.warning("No similar ONGs found. Try refining your description or adjusting filters.")
            else:
                st.markdown("### Resultados más relevantes:")
                st.markdown("---")

                results = []
                for ong in similar_ongs:
                    ong_data = df_filtrado.iloc[ong_names.index(ong['ONG'])]
                    comunidad = ong_data['Tipo de comunidad/población más beneficiado por las actividades de la organización']
                    Objetivo_General = ong_data['Objetivo_General']
                    Responsable = ong_data['Responsable']
                    Contacto = ong_data['Contacto']
                    Facebook = ong_data['Facebook']
                    Instagram = ong_data['Instagram']
                    Página_web = ong_data['Página_web']
                    Departamentos = ', '.join(ong_data['Departamentos_lista'])

                    card_content = f"""
                    <div style="background-color:#f9f9f9;padding:10px;margin-bottom:10px;border-radius:5px;border:1px solid #ddd; border-left: 5px solid transparent; background-image: linear-gradient(to right, #001d57 5px, #f9f9f9 5px);">
                        <strong style='color:#001d57;'>🏢 ONG:</strong> {ong['ONG']}<br>
                        <strong style='color:#001d57;'>📊 Puntaje:</strong> {ong['Score']:.4f}<br>
                    """
                    if pd.notna(comunidad):
                        card_content += f"<strong style='color:#001d57;'>🌍 Comunidad:</strong> {comunidad}<br>"
                    if pd.notna(Departamentos):
                        card_content += f"<strong style='color:#001d57;'>🗺️ Departamentos:</strong> {Departamentos}<br>"
                    if pd.notna(Objetivo_General):
                        card_content += f"<strong style='color:#001d57;'>🎯 Objetivo General:</strong> {Objetivo_General}<br>"
                    if pd.notna(Responsable):
                        card_content += f"<strong style='color:#001d57;'>👤 Responsable:</strong> {Responsable}<br>"
                    if pd.notna(Contacto):
                        card_content += f"<strong style='color:#001d57;'>📞 Contacto:</strong> {Contacto}<br>"
                    if pd.notna(Facebook):
                        card_content += f'<strong style="color:#001d57;">📘 Facebook:</strong> <a href="{Facebook}" target="_blank">{Facebook}</a><br>'
                    if pd.notna(Instagram):
                        card_content += f'<strong style="color:#001d57;">📷 Instagram:</strong> <a href="{Instagram}" target="_blank">{Instagram}</a><br>'
                    if pd.notna(Página_web):
                        card_content += f'<strong style="color:#001d57;">🌐 Página web:</strong> <a href="{Página_web}" target="_blank">{Página_web}</a><br>'

                    card_content += "</div>"

                    st.markdown(card_content, unsafe_allow_html=True)

                    ong['Comunidad'] = comunidad
                    ong['Departamentos'] = Departamentos
                    ong['Objetivo General'] = Objetivo_General
                    ong['Responsable'] = Responsable
                    ong['Contacto'] = Contacto
                    ong['Facebook'] = Facebook
                    ong['Instagram'] = Instagram
                    ong['Página web'] = Página_web
                    results.append(ong)

                df_results = pd.DataFrame(results)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_results.to_excel(writer, index=False, sheet_name='ONGs Relevantes')
                output.seek(0)

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
