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
        X = edited_df[[col_x]]
        y = edited_df[col_y]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_slider, random_state=42
        )

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Guardar estado
        st.session_state.model = model
        st.session_state.col_x = col_x
        st.session_state.col_y = col_y
        st.session_state.edited_df = edited_df

       
        grid = pd.DataFrame({col_x: np.linspace(X[col_x].min(), X[col_x].max(), 100)})
        grid[col_y] = model.predict(grid[[col_x]])
        st.session_state.grid = grid

        
        st.write("Resultados")
        st.table({
            "Pendiente (a)": [float(model.coef_[0])],
            "Intersección (b)": [float(model.intercept_)],
            "R²": [float(r2_score(y_test, y_pred))],
            "MSE": [float(mean_squared_error(y_test, y_pred))]
        })

        
        scatter = alt.Chart(edited_df).mark_circle(size=50, opacity=0.6).encode(
            x=alt.X(col_x, title=col_x),
            y=alt.Y(col_y, title=col_y),
            tooltip=[col_x, col_y]
        )
        line = alt.Chart(grid).mark_line().encode(x=col_x, y=col_y)
        st.altair_chart(scatter + line, use_container_width=True)

#Prediccion
if "model" in st.session_state:
    st.write("Predicción con valor X")
    x_new = st.number_input(f"Valor para {st.session_state.col_x}:", value=0.0)
    if st.button("Predecir"):
        m = st.session_state.model
        col_x = st.session_state.col_x
        col_y = st.session_state.col_y
        pred = float(m.predict([[x_new]])[0])
        st.write(f"Predicción de {col_y}: **{pred:.4f}**")

        # Punto nuevo
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
            alt.Chart(st.session_state.grid)
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
