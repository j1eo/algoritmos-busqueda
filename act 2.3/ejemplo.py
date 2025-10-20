import streamlit as st
st.title('Sumadora de dos numeros')

num1 = st.number_input('Ingrese el primer numero', value=0)
num2 = st.number_input('Ingrese el segundo numero', value=0)

suma = num1 + num2
st.write(f'La suma de {num1} + {num2} es: {suma}')