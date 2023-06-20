import streamlit as st
import subprocess

# Streamlit app
st.title("Menu Utama")

# Menu 1: Fungsi Satu
if st.button("Menu Fungsi Satu"):
    subprocess.Popen(["streamlit", "run", "fungsisatu.py"])

# Menu 2: Fungsi Dua
if st.button("Menu Fungsi Dua"):
    subprocess.Popen(["streamlit", "run", "fungsidua.py"])

# Menu 3: SNN
if st.button("Menu SNN"):
    subprocess.Popen(["streamlit", "run", "snn.py"])
