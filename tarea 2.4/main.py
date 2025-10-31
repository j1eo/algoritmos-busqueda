import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# Configuración de la aplicación
st.set_page_config(page_title="K-Means con PCA y Comparativa", layout="wide")
st.title("Clustering Interactivo con K-Means y PCA (Comparación Antes/Después)")
st.write("""
Sube tus datos, aplica K-Means, y observa cómo el algoritmo agrupa los puntos en un espacio reducido con PCA (2D o 3D).  
También puedes comparar la distribución antes y después del clustering.
""")

# --- Subir archivo ---
st.sidebar.header("Subir datos")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Archivo cargado correctamente.")
    st.write("### Vista previa de los datos:")
    st.dataframe(data.head())

    # Filtrar columnas numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("El archivo debe contener al menos dos columnas numéricas.")
    else:
        st.sidebar.header("Configuración del modelo")

        # Seleccionar columnas a usar
        selected_cols = st.sidebar.multiselect(
            "Selecciona las columnas numéricas para el clustering:",
            numeric_cols,
            default=numeric_cols
        )

        # Parámetros de clustering
        k = st.sidebar.slider("Número de clusters (k):", 1, 10, 3)

        # --- NUEVOS PARÁMETROS ---
        st.sidebar.subheader("Parámetros de K-Means")

        init = st.sidebar.selectbox(
            "Método de inicialización (init):",
            options=["k-means++", "random"],
            index=0
        )

        max_iter = st.sidebar.number_input(
            "Número máximo de iteraciones (max_iter):",
            min_value=1, max_value=1000, value=300, step=50
        )

        n_init = st.sidebar.number_input(
            "Número de inicializaciones diferentes (n_init):",
            min_value=1, max_value=50, value=10, step=1
        )

        random_state = st.sidebar.number_input(
            "Semilla aleatoria (random_state):",
            min_value=0, max_value=9999, value=42, step=1
        )

        n_components = st.sidebar.radio("Visualización PCA:", [2, 3], index=0)

        # --- Datos y modelo ---
        X = data[selected_cols].dropna()

        if X.shape[0] < k:
            st.error(f"No hay suficientes filas para {k} clusters. Reduce el valor de k.")
        else:
            # --- Cálculo con indicador de carga ---
            with st.spinner("Calculando K-Means..."):
                kmeans = KMeans(
                    n_clusters=k,
                    init=init,
                    max_iter=max_iter,
                    n_init=n_init,
                    random_state=random_state
                )
                kmeans.fit(X)
                data['Cluster'] = kmeans.labels_

            # --- PCA con manejo de errores ---
            try:
                if X.shape[1] < n_components:
                    st.warning(
                        f"Solo hay {X.shape[1]} columnas numéricas seleccionadas, "
                        f"por lo tanto se usará PCA con {X.shape[1]} componente(s)."
                    )
                    n_components = X.shape[1]

                with st.spinner("Calculando y aplicando PCA..."):
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(X)

                pca_cols = [f'PCA{i+1}' for i in range(n_components)]
                pca_df = pd.DataFrame(X_pca, columns=pca_cols)
                pca_df['Cluster'] = data['Cluster']

                # Mostrar varianza explicada
                explained_var = pca.explained_variance_ratio_ * 100
                var_text = " | ".join([f"PCA{i+1}: {v:.2f}%" for i, v in enumerate(explained_var)])
                st.info(f"Varianza explicada por componente: {var_text}")

                # --- Visualización antes del clustering ---
                with st.spinner("Generando gráfico: Distribución original (antes de K-Means)..."):
                    st.subheader("Distribución original (antes de K-Means)")
                    if n_components == 2:
                        fig_before = px.scatter(
                            pca_df,
                            x='PCA1',
                            y='PCA2',
                            title="Datos originales proyectados con PCA (sin agrupar)",
                            color_discrete_sequence=["gray"]
                        )
                    elif n_components >= 3:
                        fig_before = px.scatter_3d(
                            pca_df,
                            x='PCA1',
                            y='PCA2',
                            z='PCA3',
                            title="Datos originales proyectados con PCA (sin agrupar)",
                            color_discrete_sequence=["gray"]
                        )
                    else:
                        fig_before = None

                    if fig_before:
                        st.plotly_chart(fig_before, use_container_width=True)

                # --- Visualización después del clustering ---
                with st.spinner("Generando gráfico: Datos agrupados con K-Means..."):
                    st.subheader(f"Datos agrupados con K-Means (k = {k})")
                    if n_components == 2:
                        fig_after = px.scatter(
                            pca_df,
                            x='PCA1',
                            y='PCA2',
                            color=pca_df['Cluster'].astype(str),
                            title="Clusters visualizados en 2D con PCA",
                            color_discrete_sequence=px.colors.qualitative.Vivid
                        )
                    elif n_components >= 3:
                        fig_after = px.scatter_3d(
                            pca_df,
                            x='PCA1',
                            y='PCA2',
                            z='PCA3',
                            color=pca_df['Cluster'].astype(str),
                            title="Clusters visualizados en 3D con PCA",
                            color_discrete_sequence=px.colors.qualitative.Vivid
                        )
                    else:
                        fig_after = None

                    if fig_after:
                        st.plotly_chart(fig_after, use_container_width=True)

                # --- Centroides ---
                st.subheader("Centroides de los clusters (en espacio PCA)")
                with st.spinner("Calculando centroides..."):
                    centroides_pca = pd.DataFrame(pca.transform(kmeans.cluster_centers_), columns=pca_cols)
                    st.dataframe(centroides_pca)

            except Exception as e:
                st.error(f"Error al realizar PCA o graficar: {e}")

            # --- Método del Codo ---
            st.subheader("Método del Codo (Elbow Method)")
            if st.button("Calcular número óptimo de clusters"):
                with st.spinner("Calculando método del codo..."):
                    inertias = []
                    K = range(1, 11)
                    progress = st.progress(0)
                    for i, k_val in enumerate(K):
                        km = KMeans(
                            n_clusters=k_val,
                            init=init,
                            max_iter=max_iter,
                            n_init=n_init,
                            random_state=random_state
                        )
                        km.fit(X)
                        inertias.append(km.inertia_)
                        progress.progress((i + 1) / len(K))

                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    plt.plot(K, inertias, 'bo-')
                    plt.title('Método del Codo')
                    plt.xlabel('Número de Clusters (k)')
                    plt.ylabel('Inercia (SSE)')
                    plt.grid(True)
                    st.pyplot(fig2)

            # --- Descarga de resultados ---
            st.subheader("Descargar datos con clusters asignados")
            buffer = BytesIO()
            data.to_csv(buffer, index=False, encoding='utf-8')
            st.download_button(
                label="Descargar CSV con Clusters",
                data=buffer.getvalue(),
                file_name="datos_clusterizados.csv",
                mime="text/csv"
            )

else:
    st.info("Carga un archivo CSV en la barra lateral para comenzar.")
    st.write("""
    Ejemplo de formato:
    | Ingreso_Anual | Gasto_Tienda | Edad |
    |----------------|--------------|------|
    | 45000 | 350 | 28 |
    | 72000 | 680 | 35 |
    | 28000 | 210 | 22 |
    """)
