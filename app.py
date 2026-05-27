import streamlit as st
import pandas as pd
from agents.eda_agent import run_eda_agent
from agents.cleaning_agent import run_cleaning_agent
from agents.model_agent import run_model_agent

st.set_page_config(page_title="Data Analysis Multi-Agent", layout="wide")

def main():
    st.title("Sistema Multiagente para Análisis de Datos")
    
    # Agente Orquestador: Actúa como el controlador principal.
    # Maneja la UI base, carga de archivos y sincroniza el estado (Session State) entre los 3 agentes.
    
    # Usar un 'expander' permite ocultar el menú de carga haciendo clic en la flecha, limpiando la UI
    with st.sidebar.expander("Carga de Datos", expanded=True):
        uploaded_file = st.file_uploader("Sube tu archivo (.csv, .xlsx, .data)", type=['csv', 'xlsx', 'data'])
        
        has_header = True
        if uploaded_file is not None:
            # Opción para forzar lectura sin nombres de columnas en formatos propensos a ello
            if uploaded_file.name.endswith(('.csv', '.data')):
                has_header = st.checkbox("¿El archivo tiene cabecera?", value=True)
                
    if uploaded_file is not None:
        try:
            
            # Detectar si cambió el archivo o la configuración de cabecera para re-procesar
            should_parse = False
            if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != uploaded_file.name:
                should_parse = True
            elif 'last_has_header' not in st.session_state or st.session_state['last_has_header'] != has_header:
                should_parse = True
                
            if should_parse:
                if uploaded_file.name.endswith(('.csv', '.data')):
                    if has_header:
                        df = pd.read_csv(uploaded_file, sep=None, engine='python')
                    else:
                        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None)
                        df.columns = [f"variable{i+1}" for i in range(df.shape[1])]
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state['original_df'] = df.copy()
                st.session_state['cleaned_df'] = df.copy()
                st.session_state['last_uploaded_file'] = uploaded_file.name
                st.session_state['last_has_header'] = has_header
                
            st.sidebar.success("Datos cargados correctamente.")
            
            # Crear las vistas (Pestañas)
            tab1, tab2, tab3 = st.tabs(["Agente 1: EDA", "Agente 2: Limpieza de Datos", "Agente 3: Modelo Predictivo"])
            
            with tab1:
                st.header("Análisis Exploratorio de Datos (EDA)")
                run_eda_agent(st.session_state['original_df'])
                
            with tab2:
                st.header("Limpieza y Preparación de Datos")
                cleaned_df = run_cleaning_agent(st.session_state['cleaned_df'], st.session_state['original_df'])
                # Actualizar el cleaned_df en la sesión si hubo cambios ("Reenviar dataset")
                if cleaned_df is not None:
                    st.session_state['cleaned_df'] = cleaned_df
                    
            with tab3:
                st.header("Modelo Predictivo")
                run_model_agent(st.session_state['cleaned_df'], st.session_state['original_df'])
                
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
    else:
        st.info("Por favor, sube un archivo CSV o Excel en la barra lateral para comenzar.")
        # Limpiar el estado si no hay archivo
        st.session_state.clear()

if __name__ == "__main__":
    main()
