import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("Regresión Lineal Simple")

uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Vista de datos")
    st.dataframe(df, use_container_width=True)

    st.write("Editar datos")
    edited_df = st.data_editor(df, use_container_width=True)

    cols = edited_df.columns.tolist()
    col_x = st.selectbox("Columna X", cols)
    col_y = st.selectbox("Columna Y", cols, index=1)

    test_size_slider = st.slider("Proporción de test", 0.1, 0.9, 0.3, 0.05)

    if st.button("Entrenar"):
        df_xy = edited_df[[col_x, col_y]].copy()
        df_xy[col_x] = pd.to_numeric(df_xy[col_x], errors="coerce")
        df_xy[col_y] = pd.to_numeric(df_xy[col_y], errors="coerce")
        df_xy = df_xy.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_xy) < 2:
            st.error("Se requieren al menos 2 filas numéricas para entrenar.")
            st.stop()

        X = df_xy[[col_x]]
        y = df_xy[col_y]

        n = len(df_xy)
        min_train_rows = 2 if n > 2 else 1
        max_test_fraction = (n - min_train_rows) / n
        min_test_fraction = 1 / n

        adjusted = False
        test_size = test_size_slider
        if test_size > max_test_fraction:
            test_size = max_test_fraction
            adjusted = True
        if test_size < min_test_fraction:
            test_size = min_test_fraction
            adjusted = True
        if adjusted:
            st.warning(
                "La proporcion de test se ajusto para asegurar al menos una fila en test y "
                f"{min_train_rows} filas en entrenamiento."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.model = model
        st.session_state.col_x = col_x
        st.session_state.col_y = col_y
        st.session_state.edited_df = df_xy

        x_min, x_max = float(X[col_x].min()), float(X[col_x].max())
        grid = pd.DataFrame({col_x: np.linspace(x_min, x_max, 200)})
        grid[col_y] = model.predict(grid[[col_x]])
        st.session_state.grid = grid

        mse = float(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred) if len(y_test) >= 2 else float("nan")

        st.session_state.metrics_df = pd.DataFrame({
            "Pendiente (a)": [float(model.coef_[0])],
            "Intersección (b)": [float(model.intercept_)],
            "R²": [None if np.isnan(r2) else float(r2)],
            "MSE": [mse]
        })

        domain = ["Datos reales", "Línea de regresión"]
        colors = ["#1f77b4", "#e74c3c"]

        scatter = (
            alt.Chart(df_xy)
            .transform_calculate(key="'Datos reales'")
            .mark_circle(size=50, opacity=0.6)
            .encode(
                x=alt.X(col_x, title=col_x),
                y=alt.Y(col_y, title=col_y),
                color=alt.Color("key:N",
                                scale=alt.Scale(domain=domain, range=colors),
                                legend=alt.Legend(title="Leyenda")),
                tooltip=[col_x, col_y]
            )
        )
        line = (
            alt.Chart(grid.sort_values(col_x))
            .transform_calculate(key="'Línea de regresión'")
            .mark_line()
            .encode(
                x=col_x, y=col_y,
                color=alt.Color("key:N",
                                scale=alt.Scale(domain=domain, range=colors),
                                legend=alt.Legend(title="Leyenda"))
            )
        )
        st.write("Resultados")
        st.table(st.session_state.metrics_df)
        st.altair_chart(scatter + line, use_container_width=True)

if "model" in st.session_state:
    st.write("Resultados")
    st.table(st.session_state.metrics_df)

    st.write("Predicción con valor X")
    x_new = st.number_input(f"Valor para {st.session_state.col_x}:", value=0.0)
    if st.button("Predecir"):
        m = st.session_state.model
        col_x = st.session_state.col_x
        col_y = st.session_state.col_y

        pred = float(m.predict(pd.DataFrame({col_x: [x_new]}))[0])
        st.write(f"Predicción de {col_y}: **{pred:.4f}**")

        xmin = st.session_state.edited_df[col_x].min()
        xmax = st.session_state.edited_df[col_x].max()
        if x_new < xmin or x_new > xmax:
            st.warning(f"El valor ingresado está fuera del rango de entrenamiento [{xmin:.4f}, {xmax:.4f}].")

        new_point = pd.DataFrame({col_x: [x_new], col_y: [pred]})

        domain = ["Datos reales", "Línea de regresión", "Predicción"]
        colors  = ["#1f77b4", "#e74c3c", "gold"]

        scatter = (
            alt.Chart(st.session_state.edited_df)
            .transform_calculate(key="'Datos reales'")
            .mark_circle(size=50, opacity=0.6)
            .encode(
                x=col_x, y=col_y,
                color=alt.Color("key:N",
                                scale=alt.Scale(domain=domain, range=colors),
                                legend=alt.Legend(title="Leyenda"))
            )
        )
        line = (
            alt.Chart(st.session_state.grid.sort_values(col_x))
            .transform_calculate(key="'Línea de regresión'")
            .mark_line()
            .encode(
                x=col_x, y=col_y,
                color=alt.Color("key:N",
                                scale=alt.Scale(domain=domain, range=colors),
                                legend=alt.Legend(title="Leyenda"))
            )
        )
        dot = (
            alt.Chart(new_point)
            .transform_calculate(key="'Predicción'")
            .mark_point(size=120, shape="diamond")
            .encode(
                x=col_x, y=col_y,
                color=alt.Color("key:N",
                                scale=alt.Scale(domain=domain, range=colors),
                                legend=alt.Legend(title="Leyenda")),
                tooltip=[col_x, col_y]
            )
        )
        st.altair_chart(scatter + line + dot, use_container_width=True)
else:
    st.info("Sube un CSV, selecciona columnas y entrena el modelo.")
