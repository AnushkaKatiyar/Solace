from io import BytesIO

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
    processed_data = output.getvalue()
    return processed_data

excel_data = to_excel(result_df)
st.download_button("📥 Download Excel", excel_data, file_name="Solace_Estimates.xlsx")
