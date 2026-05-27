import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda_agent(df):
    """
    Agente 1: Análisis Exploratorio de Datos (EDA).
    Este agente se encarga de dar un resumen del dataset original (antes de la limpieza).
    Genera estadísticas descriptivas numéricas y categóricas, y crea visualizaciones (boxplots, barras, heatmap).
    
    Variables configurables:
    - cols_per_row (int): Controla cuántos boxplots se muestran por fila (actualmente 3).
    - factor_iqr (float): Multiplicador del rango intercuartílico (IQR) para detectar outliers (típicamente 1.5).
    """
    st.subheader("Información General de Datos")
    
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    total_nulls = df.isnull().sum().sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Cantidad de Registros", total_rows)
    col2.metric("Recuento de Columnas", total_cols)
    col3.metric("Datos Nulos (Total)", total_nulls)
    
    st.markdown("---")
    
    # Separar variables numéricas y categóricas
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    st.subheader("Análisis de Variables Numéricas")
    if not num_cols:
        st.info("No se encontraron variables numéricas en el dataset.")
    else:
        cols_per_row = 3
        for i in range(0, len(num_cols), cols_per_row):
            row_cols = num_cols[i:i+cols_per_row]
            st_cols = st.columns(len(row_cols))
            
            for j, col in enumerate(row_cols):
                with st_cols[j]:
                    st.markdown(f"#### `{col}`")
                    
                    series = df[col].dropna()
                    if len(series) == 0:
                        st.warning(f"La columna {col} está vacía o contiene solo nulos.")
                        continue
                    
                    mean_val = series.mean()
                    median_val = series.median()
                    mode_val = series.mode().iloc[0] if not series.mode().empty else np.nan
                    
                    # Calcular outliers con IQR
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_count = ((series < lower_bound) | (series > upper_bound)).sum()
                    
                    # Mostrar métricas en formato texto compacto
                    st.markdown(f"**Media:** {mean_val:.2f} | **Mediana:** {median_val:.2f}")
                    st.markdown(f"**Moda:** {mode_val:.2f} | **Outliers:** {outliers_count}")
                    
                    # Boxplot compacto
                    fig = px.box(df, y=col, title=f"Boxplot de {col}", height=300)
                    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            
    st.subheader("Análisis de Variables Categóricas")
    if not cat_cols:
        st.info("No se encontraron variables categóricas en el dataset.")
    else:
        for col in cat_cols:
            st.markdown(f"#### Variable: `{col}`")
            
            # Bar chart (Conteo)
            val_counts = df[col].value_counts().reset_index()
            val_counts.columns = [col, 'Conteo']
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                fig_bar = px.bar(val_counts, x=col, y='Conteo', title=f"Distribución de {col} (Conteo)")
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with col_b:
                fig_pie = px.pie(val_counts, names=col, values='Conteo', title=f"Proporción de {col} (%)", hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            st.markdown("---")
            
    st.subheader("Matriz de Correlación")
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                             title="Heatmap de Correlación (Pearson)",
                             color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Se necesitan al menos 2 variables numéricas para calcular la matriz de correlación.")
