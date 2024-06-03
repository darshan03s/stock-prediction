import streamlit as st
import os

st.set_page_config(page_title='Stock Prediction | Notebook', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

with open('./pages/stock-lstm.html', "r", encoding="utf-8") as file:
    html_content = file.read()

st.html(html_content)