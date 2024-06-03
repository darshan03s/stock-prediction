import streamlit as st
import os

with open('./pages/stock-lstm.html', "r", encoding="utf-8") as file:
    html_content = file.read()

st.html(html_content)