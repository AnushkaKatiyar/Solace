import pandas as pd
from io import BytesIO
import streamlit as st
import streamlit_lottie as st_lottie
import requests

# Function to convert dataframe to Excel bytes
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
    processed_data = output.getvalue()
    return processed_data

# Function to load Lottie JSON from URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Call this after result_df is ready:
if 'result_df' in locals() or 'result_df' in globals():
    excel_data = to_excel(result_df)
    st.download_button("📥 Download Excel", data=excel_data, file_name="Solace_Estimates.xlsx")

    lottie_url = "https://assets9.lottiefiles.com/private_files/lf30_editor_8zonlf.json"
    lottie_json = load_lottie_url(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=150)
