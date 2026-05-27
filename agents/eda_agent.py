import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda_agent(df):
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
        for col in num_cols:
            st.markdown(f"#### Variable: `{col}`")
            
            # Dropna for calculations
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
            
            # Mostrar métricas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Media", f"{mean_val:.2f}")
            m2.metric("Mediana", f"{median_val:.2f}")
            m3.metric("Moda", f"{mode_val:.2f}")
            m4.metric("Potenciales Outliers", outliers_count)
            
            # Boxplot
            fig = px.box(df, y=col, title=f"Boxplot de {col}")
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
