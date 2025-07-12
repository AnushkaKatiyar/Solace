from io import BytesIO

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
    processed_data = output.getvalue()
    return processed_data

excel_data = to_excel(result_df)
st.download_button("📥 Download Excel", excel_data, file_name="Solace_Estimates.xlsx")

import streamlit_lottie as st_lottie
import requests

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets9.lottiefiles.com/private_files/lf30_editor_8zonlf.json"
lottie_json = load_lottie_url(lottie_url)
st_lottie(lottie_json, height=150)