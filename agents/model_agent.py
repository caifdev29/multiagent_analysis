import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

def run_model_agent(df, original_df):
    st.write("### Configuración del Modelo Predictivo")
    
    if df.empty:
        st.error("El dataset está vacío. Por favor, revisa la limpieza de datos.")
        return
        
    if df.isnull().any().any():
        st.warning("El dataset aún contiene valores nulos. Los modelos podrían fallar. Te sugerimos volver a la pestaña de Limpieza.")
        
    # Clasificar columnas según su tipo en el dataset original
    num_cols = original_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = original_df.select_dtypes(exclude=np.number).columns.tolist()
    
    # Separar la selección de la variable objetivo en 2 categorías en la UI
    st.write("#### Categoría de la Variable Objetivo")
    tipo_variable = st.radio("Selecciona la categoría de la variable objetivo a predecir:", 
                             ["Variables Numéricas", "Variables Categóricas"], horizontal=True)
    
    target_col = None
    if tipo_variable == "Variables Numéricas":
        if num_cols:
            target_col = st.selectbox("Selecciona la variable objetivo (Numérica):", num_cols)
        else:
            st.warning("No hay variables numéricas en el dataset original.")
    else:
        if cat_cols:
            target_col = st.selectbox("Selecciona la variable objetivo (Categórica):", cat_cols)
        else:
            st.warning("No hay variables categóricas en el dataset original.")
            
    if target_col is None:
        st.error("Por favor, selecciona una variable objetivo para continuar.")
        return
    
    # Ignorar variables categóricas para el entrenamiento, utilizando únicamente numéricas (X)
    features = [c for c in num_cols if c != target_col and c in df.columns]
    
    if len(features) == 0:
        st.error("No hay suficientes variables numéricas para utilizar como características (X).")
        return
        
    if st.button("Entrenar Modelos"):
        # Asegurarse de que no hay nulos que rompan scikit-learn
        model_df = df.dropna()
        
        X = model_df[features]
        y = model_df[target_col]
        
        # Validar si y es numérico
        if not pd.api.types.is_numeric_dtype(y):
            st.error("La variable objetivo debe ser numérica para un problema de regresión. Por favor, asegúrate de aplicar la codificación de variables en la pestaña de Limpieza de Datos primero.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. Regresión Lineal
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
        
        # 2. Árbol de Decisión
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X_train, y_train)
        dt_preds = dt.predict(X_test)
        dt_rmse = np.sqrt(mean_squared_error(y_test, dt_preds))
        
        st.markdown("---")
        st.write("### Resultados de Evaluación")
        
        col1, col2 = st.columns(2)
        col1.metric("RMSE - Regresión Lineal", f"{lr_rmse:.4f}")
        col2.metric("RMSE - Árbol de Decisión", f"{dt_rmse:.4f}")
        
        best_model = "Regresión Lineal" if lr_rmse < dt_rmse else "Árbol de Decisión"
        st.success(f"**El mejor modelo es: {best_model}** (Menor RMSE)")
        
        # Comparativa Gráfica de Errores (RMSE)
        rmse_df = pd.DataFrame({
            'Modelo': ['Regresión Lineal', 'Árbol de Decisión'],
            'RMSE': [lr_rmse, dt_rmse]
        })
        
        fig_rmse = px.bar(rmse_df, x='Modelo', y='RMSE', color='Modelo', title="Comparación de RMSE (Menor es mejor)")
        st.plotly_chart(fig_rmse, use_container_width=True)
        
        st.write("### Gráficos de Dispersión: Valores Reales vs Predichos")
        st.info("Nota: Reemplaza a la 'Matriz de Confusión' al tratarse de un problema de Regresión.")
        
        # Comparativa Scatter (Real vs Predicho)
        c1, c2 = st.columns(2)
        
        with c1:
            fig_lr = px.scatter(x=y_test, y=lr_preds, labels={'x': 'Valores Reales', 'y': 'Predicciones'}, 
                                title="Regresión Lineal", opacity=0.6)
            # Agregar línea ideal
            min_val = min(y_test.min(), lr_preds.min())
            max_val = max(y_test.max(), lr_preds.max())
            fig_lr.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal', line=dict(dash='dash', color='red')))
            st.plotly_chart(fig_lr, use_container_width=True)
            
        with c2:
            fig_dt = px.scatter(x=y_test, y=dt_preds, labels={'x': 'Valores Reales', 'y': 'Predicciones'}, 
                                title="Árbol de Decisión", opacity=0.6)
            fig_dt.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal', line=dict(dash='dash', color='red')))
            st.plotly_chart(fig_dt, use_container_width=True)
