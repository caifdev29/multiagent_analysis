import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def get_outliers_mask(df):
    num_cols = df.select_dtypes(include=np.number).columns
    mask = pd.Series(False, index=df.index)
    
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        mask = mask | col_mask
        
    return mask

def run_cleaning_agent(df, original_df):
    st.write("### Estado Actual del Dataset")
    
    # Calcular métricas
    rows_with_nulls = df.isnull().any(axis=1).sum()
    
    outliers_mask = get_outliers_mask(df)
    rows_with_outliers = outliers_mask.sum()
    
    total_rows = len(df)
    
    # Lógica de categorías:
    # Dañados: tienen nulos
    # Anómalos: no tienen nulos, pero tienen outliers
    # Viables: ni nulos ni outliers
    
    null_mask = df.isnull().any(axis=1)
    
    danados_count = null_mask.sum()
    anomalos_count = (~null_mask & outliers_mask).sum()
    viables_count = total_rows - danados_count - anomalos_count
    
    col1, col2 = st.columns(2)
    col1.metric("Filas con Nulos", rows_with_nulls)
    col2.metric("Filas con Potenciales Outliers", rows_with_outliers)
    
    # Gráficos de estado
    status_df = pd.DataFrame({
        'Estado': ['Dañados (Nulos)', 'Anómalos (Outliers)', 'Viables'],
        'Cantidad': [danados_count, anomalos_count, viables_count]
    })
    
    st.write("#### Relación de Calidad de Datos")
    c1, c2 = st.columns(2)
    with c1:
        fig_bar = px.bar(status_df, x='Estado', y='Cantidad', title="Cantidad por Estado", color='Estado')
        st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        fig_pie = px.pie(status_df, names='Estado', values='Cantidad', title="Proporción por Estado (%)")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("---")
    st.write("### Opciones de Limpieza")
    
    modo = st.radio("Modo de Limpieza", ["Automático", "Manual"], horizontal=True)
    
    # Almacenar el dataframe temporal
    temp_df = df.copy()
    
    if modo == "Automático":
        st.write("#### Tratado de Nulos")
        nulos_opt = st.selectbox("Selecciona acción para nulos:", 
                                ["Eliminación de nulos", "Reemplazo de nulos (Media aritmética)", "Sin cambios (Colocar 0 / NA)"])
        
        st.write("#### Tratado de Outliers")
        outliers_opt = st.selectbox("Selecciona acción para outliers:",
                                   ["Eliminación de outliers", "Regularización de outliers (Limitar a rangos)", "Sin acciones"])
        
        if st.button("Aplicar Limpieza Automática y Reenviar Dataset"):
            # Procesar nulos
            if nulos_opt == "Eliminación de nulos":
                temp_df = temp_df.dropna()
            elif nulos_opt == "Reemplazo de nulos (Media aritmética)":
                num_cols = temp_df.select_dtypes(include=np.number).columns
                temp_df[num_cols] = temp_df[num_cols].fillna(temp_df[num_cols].mean())
                cat_cols = temp_df.select_dtypes(exclude=np.number).columns
                # Para categóricas no hay media, rellenar con moda
                for c in cat_cols:
                    temp_df[c] = temp_df[c].fillna(temp_df[c].mode()[0] if not temp_df[c].mode().empty else "NA")
            elif nulos_opt == "Sin cambios (Colocar 0 / NA)":
                num_cols = temp_df.select_dtypes(include=np.number).columns
                temp_df[num_cols] = temp_df[num_cols].fillna(0)
                cat_cols = temp_df.select_dtypes(exclude=np.number).columns
                temp_df[cat_cols] = temp_df[cat_cols].fillna("NA")
                
            # Procesar outliers
            if outliers_opt == "Eliminación de outliers":
                out_mask = get_outliers_mask(temp_df)
                temp_df = temp_df[~out_mask]
            elif outliers_opt == "Regularización de outliers (Limitar a rangos)":
                num_cols = temp_df.select_dtypes(include=np.number).columns
                for col in num_cols:
                    Q1 = temp_df[col].quantile(0.25)
                    Q3 = temp_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    temp_df[col] = np.clip(temp_df[col], lower, upper)
            
            # Codificación automática de categóricas requerida para el Agente 3
            # Convertimos strings a códigos numéricos
            cat_cols = temp_df.select_dtypes(exclude=np.number).columns
            for col in cat_cols:
                temp_df[col] = temp_df[col].astype('category').cat.codes
                
            st.success("Cambios automáticos aplicados. Dataset listo para Modelado.")
            return temp_df
            
    else:
        st.write("#### Edición Manual de Datos")
        st.info("Modifica directamente la tabla. Los cambios de Nulos/Outliers automáticos son ignorados en este modo.")
        edited_df = st.data_editor(temp_df, num_rows="dynamic", use_container_width=True)
        
        if st.button("Aplicar Limpieza Manual y Reenviar Dataset"):
            # Codificación de variables string para modelos
            cat_cols = edited_df.select_dtypes(exclude=np.number).columns
            for col in cat_cols:
                edited_df[col] = edited_df[col].astype('category').cat.codes
            st.success("Cambios manuales guardados. Dataset listo para Modelado.")
            return edited_df
            
    st.markdown("---")
    if st.button("Reiniciar Dataset (Volver al Original)"):
        st.warning("Se ha reiniciado el dataset a su estado original.")
        return original_df.copy()
        
    return None
