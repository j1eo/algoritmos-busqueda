import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import time
import os

st.title("K-Means")

# --- Cargar datos ---
uploaded_file = st.file_uploader("Sube un archivo CSV (opcional)", type=["csv"])

# Obtener la ruta absoluta del archivo clientes.csv (en la misma carpeta del script)
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, "clientes.csv")

with st.spinner("Cargando datos..."):
    time.sleep(1)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Vista previa de los datos originales (archivo subido)")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.subheader("Vista previa de los datos originales (archivo por defecto)")
    else:
        st.error("No se encontró 'clientes.csv' y no se subió ningún archivo.")
        st.stop()

    st.dataframe(df.head())

# --- Escalado de datos ---
escalador = MinMaxScaler().fit(df.values)
df_scaled = pd.DataFrame(escalador.transform(df.values), columns=["Saldo", "transacciones"])

st.subheader("Datos escalados")
st.dataframe(df_scaled.head())

# --- Calcular método del codo ---
st.subheader("Método del Codo para determinar el número de clusters (k)")
with st.spinner("Calculando para distintos valores de k..."):
    time.sleep(1)
    inercias = []
    K_range = range(2, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df_scaled.values)
        inercias.append(kmeans.inertia_)

fig_codo, ax = plt.subplots(figsize=(6, 5), dpi=100)
ax.plot(K_range, inercias, marker="o", color="purple", linewidth=2)
ax.set_xlabel("Número de clusters (k)", fontsize=14)
ax.set_ylabel("Inercia", fontsize=14)
ax.set_title("Método del Codo", fontsize=16)
ax.grid(True)
st.pyplot(fig_codo)

st.info("Observa el punto donde la inercia deja de disminuir bruscamente para elegir el mejor k.")

# --- Elegir número de clusters ---
k_optimo = st.slider("Selecciona el número de clusters (k)", 2, 9, 3)

# --- Aplicar KMeans con el k seleccionado ---
with st.spinner(f"Entrenando modelo K-Means con k = {k_optimo}..."):
    time.sleep(1)
    kmeans = KMeans(n_clusters=k_optimo, random_state=42).fit(df_scaled.values)
    df_scaled["cluster"] = kmeans.labels_

st.success(f"Modelo entrenado con k = {k_optimo}")

# --- Mostrar resultados ---
st.subheader("Resultados")
st.write("Centroides:")
st.write(pd.DataFrame(kmeans.cluster_centers_, columns=["Saldo", "transacciones"]))
st.write(f"Inercia: {kmeans.inertia_:.2f}")

# --- Graficar clusters ---
st.subheader("Visualización")
fig_clusters, ax2 = plt.subplots(figsize=(6, 5), dpi=100)
colores = [
    "red", "blue", "orange", "green", "purple", "pink", "brown",
    "gray", "cyan", "olive", "navy", "magenta", "lime", "teal"
]


for cluster in range(kmeans.n_clusters):
    ax2.scatter(df_scaled[df_scaled["cluster"] == cluster]["Saldo"],
                df_scaled[df_scaled["cluster"] == cluster]["transacciones"],
                s=180, color=colores[cluster], alpha=0.5, label=f"Cluster {cluster+1}")
    ax2.scatter(kmeans.cluster_centers_[cluster][0],
                kmeans.cluster_centers_[cluster][1],
                marker="P", s=280, color=colores[cluster])

ax2.set_title(f"Clusters K-Means (k={k_optimo})", fontsize=16)
ax2.set_xlabel("Saldo en cuenta de ahorros", fontsize=13)
ax2.set_ylabel("Veces que usó tarjeta de crédito", fontsize=13)
ax2.legend()
st.pyplot(fig_clusters)
