# Solace Streamlit App (Enhanced Version)


import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from mistralai import Mistral, UserMessage, SystemMessage
from PIL import Image
import json
from streamlit_lottie import st_lottie
import requests
import io


# === Paths ===
MODEL_DIR = "models/"
ASSETS_DIR = "assets/"
LOGO_PATH = os.path.join(ASSETS_DIR, "Solace_logo.png")


# === Load models and encoders ===
try:
    with open(os.path.join(MODEL_DIR, "low_custom.pkl"), "rb") as f:
        model_low = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "mid_custom.pkl"), "rb") as f:
        model_mid = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "high_custom.pkl"), "rb") as f:
        model_high = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "duration_model.pkl"), "rb") as f:
        duration_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "ohe.pkl"), "rb") as f:
        ohe = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "ohe_duration.pkl"), "rb") as f:
        ohe_duration = pickle.load(f)
except Exception as e:
    st.error(f"🔴 Error loading models: {e}")
    st.stop()


bert_model = SentenceTransformer('all-MiniLM-L6-v2')
model_dict = {'low': model_low, 'mid': model_mid, 'high': model_high}


# === Mistral API Key ===
mistral_api_key = st.secrets["mistral_api_key"]
client = Mistral(api_key=mistral_api_key)


# === Phase Mapping ===
phase_mapping = {
    "Scope": "I. Scope",
    "Design": "II. Design",
    "CM": "III. CM - Construction Management",
    "CM,Art,F&E": "IV. Furniture Equipment & Art",
    "CM,F&E": "V. Commissioning",
    "F&E": "VI. Inspection & Testing",
    "Purch & Install": "VII. Purch & Install",
    "Construction": "VIII. Construction"
}


# === Helper function to load Lottie animation ===
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# === Sidebar ===
with st.sidebar:
    st.image(LOGO_PATH, width=120)


    # Load and display Lottie animation
    lottie_anim = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
    if lottie_anim:
        st_lottie(lottie_anim, speed=1, width=150, height=150, key="sidebar_anim")


    st.title("Solace")
    st.markdown("🚧 *NYC School Construction Estimator*")
    st.markdown("---")
    st.markdown("Created for Solace Technologies")
    st.markdown("🔗 [GitHub Repo](https://github.com/AnushkaKatiyar)")
    st.markdown("💬 Powered by Mistral + ML Models")


# === Dark Mode Toggle & CSS Injection ===
dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")


dark_css = """
<style>
    html, body, [class*="st-"] {
        background-color: #0E1117;
        color: white;
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #161b22;
        color: white;
    }
    label, .stTextInput label, .stSelectbox label {
        color: white !important;
    }
    .stButton > button {
        background-color: #1E90FF;
        color: white;
        border-radius: 5px;
    }
    .stDataFrame, .stTable {
        background-color: #161b22;
        color: white;
    }
</style>
"""


if dark_mode:
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    # Reset to light mode
    st.markdown("<style>body, .stApp {background-color: white; color: black;}</style>", unsafe_allow_html=True)


# === Helper Functions ===
def get_detailed_plan_from_mistral(description):
    prompt = f'''
Generate a JSON construction plan for the following NYC school project:


"{description}"


Output should be a list of 8 phases. Each phase must include:
- Phase: (string) e.g. "I. Scope",
- Description: (string),a short description,
- Subtasks: (list of strings),6 to 10 detailed subtasks as part of a "Subphase Breakdown" list with name
- Permissions Required: (list of strings),required NYC government permissions (e.g., SCA, DoE, FDNY),
- Vendors: (list of strings),1–2 **actual NYC-based vendors or well-known relevant companies** (avoid placeholders like 'VendorX', 'VendorA'),
- Estimated Labor: (integer)
- Subphase Breakdown: (list of phases and subtasks(8 phases and 6-10 subtasks) from above as dicts). Each dict must have:
  - Name: (string)
  - Duration (weeks): (float)
  - Cost (USD): (float)


Return ONLY valid JSON. Here's the format:
[
  {{
    "Phase": "I. Scope",
    "Description": "...",
    "Subtasks": ["task1", "task2"],
    "Permissions Required": ["SCA"],
    "Vendors": ["STV Group", "AECOM"], 
    "Estimated Labor": 10,
    "Subphase Breakdown": [
      {{
        "Name": "Site Survey",
        "Duration (weeks)": 1,
        "Cost (USD)": 5000
      }},
      {{
        "Name": "Planning",
        "Duration (weeks)": 2,
        "Cost (USD)": 8000
      }},
      {{
        "Name": "Design",
        "Duration (weeks)": 2,
        "Cost (USD)": 8000
      }},
      {{
        "Name": "Preperation",
        "Duration (weeks)": 2,
        "Cost (USD)": 8000
      }},
    ]
  }},
  ...
]
'''
    messages = [
        SystemMessage(content="You are a helpful assistant for school construction in NYC."),
        UserMessage(content=prompt),
    ]
    response = client.chat.complete(model="mistral-medium", messages=messages)
    return response.choices[0].message.content


def prepare_features_for_duration(description, phase_name):
    df = pd.DataFrame([{
        "description_no_stopwords": description,
        "Project Phase Name": phase_name,
        "project_status": "Complete",
        "timeline_status": "Complete"
    }])
    embedding = bert_model.encode(df["description_no_stopwords"].tolist())
    cat_feats = ohe_duration.transform(df[["Project Phase Name", "project_status", "timeline_status"]])
    return np.hstack([embedding, cat_feats])


def prepare_single_row(description, phase, duration_weeks):
    df = pd.DataFrame([{
        "Project Phase Name": phase,
        "project_status": "Complete",
        "timeline_status": "Complete",
        "end_date_missing": True,
        "duration_days": duration_weeks * 7
    }])
    embedding = bert_model.encode([description])
    cat_feats = ohe.transform(df[["Project Phase Name", "project_status", "timeline_status", "end_date_missing"]])
    num_feats = scaler.transform(df[["duration_days"]])
    return np.hstack([embedding, cat_feats, num_feats])


# === Layout and Input ===
st.set_page_config(page_title="Solace Estimator", layout="wide")
logo = Image.open(LOGO_PATH)
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=180)
with col2:
    st.markdown("<h2 style='margin-bottom: 0; color: #1E90FF;'>NYC School Construction Cost & Schedule Estimator</h2>", unsafe_allow_html=True)


with st.expander("📝 Customize Project", expanded=True):
    description = st.text_area("Enter Project Description:", "Hi Solace, we are planning to build a new science lab for a high school in Manhattan with latest equipment, Can you create a detailed plan?")
    bucket = st.selectbox("Select Cost Bucket:", ["low", "mid", "high"])


# === Estimation ===
if st.button("Estimate Cost and Schedule", key="run_button"):
    with st.spinner("⏳ Generating predictions and fetching plan..."):
        detailed_json = get_detailed_plan_from_mistral(description)
        try:
            json_start = detailed_json.find("[")
            json_end = detailed_json.rfind("]") + 1
            json_data = json.loads(detailed_json[json_start:json_end])
            detailed_df = pd.DataFrame(json_data)
        except Exception as e:
            st.error(f"❌ Could not parse response: {e}")
            st.code(detailed_json, language="json")
            st.stop()


        model = model_dict[bucket]
        predictions = []


        for phase_code, display_name in phase_mapping.items():
            X_dur = prepare_features_for_duration(description, phase_code)
            duration_weeks = duration_model.predict(X_dur)[0]
            X_cost = prepare_single_row(description, phase_code, duration_weeks)
            cost = model.predict(X_cost)[0]
            predictions.append({
                "Phase": display_name,
                "Predicted Duration (weeks)": round(duration_weeks, 2),
                "Predicted Cost (USD)": round(max(cost, 0), 2)
            })


        result_df = pd.DataFrame(predictions)
        total_cost = result_df["Predicted Cost (USD)"].sum()
        total_duration = result_df["Predicted Duration (weeks)"].sum()


        tab1, tab2 = st.tabs(["📊 Cost & Duration", "📋 Detailed Plan"])


        with tab1:
            st.subheader("📊 Cost & Schedule Breakdown")


            # Create a copy and format for display
            result_df_formatted = result_df.copy()
            result_df_formatted["Predicted Cost (USD)"] = result_df_formatted["Predicted Cost (USD)"].apply(lambda x: f"${x:,.2f}")
          
            def format_duration(weeks_float):
                weeks = int(weeks_float)
                days = int(round((weeks_float - weeks) * 7))
                return f"{weeks} weeks {days} days"
            result_df_formatted["Duration"] = result_df["Predicted Duration (weeks)"].apply(format_duration)


            # Display formatted table
            st.dataframe(result_df_formatted[["Phase", "Predicted Cost (USD)", "Duration"]], use_container_width=True)




            # Show metrics
            colA, colB = st.columns(2)
            colA.metric("💰 Total Estimated Cost", f"${total_cost:,.2f}")
            colB.metric("🕒 Total Duration", f"{total_duration:.1f} weeks")


            # Charts still need numeric values
          
          
            import plotly.express as px
            fig = px.bar(
            result_df,
            x="Predicted Cost (USD)",
            y="Phase",
            orientation='h',  # horizontal bars
            title="Cost per Phase",
            labels={"Predicted Cost (USD)": "Cost (USD)", "Phase": "Project Phase"},
            )


            # Customize layout for font size and axes labels
            fig.update_layout(
                xaxis_title_font=dict(size=16),
                yaxis_title_font=dict(size=16),
                xaxis_tickangle=45,
                xaxis_tickfont=dict(size=14),
                yaxis_tickfont=dict(size=14),
                margin=dict(l=100, r=40, t=50, b=50),
                height=400,
            )


            st.plotly_chart(fig, use_container_width=True)
          
            fig = px.line(
            result_df,
            x="Phase",
            y="Predicted Duration (weeks)",
            title="Duration per Phase",
            )


            fig.update_layout(
                xaxis_title="Phase",
                yaxis_title="Duration (weeks)",
                xaxis_tickangle=45,        # Rotate x-axis labels 45 degrees
                font=dict(size=16),        # Increase font size
                margin=dict(l=40, r=40, t=40, b=80),  # Adjust margins if needed
            )


            st.plotly_chart(fig, use_container_width=True)


        with tab2:
            st.subheader("📋 Phase-wise Construction Plan")
            for _, row in detailed_df.iterrows():
                with st.expander(f"{row['Phase']} – {row['Description']}"):
                    st.markdown("**Subtasks:**")
                    for task in row.get("Subtasks", []):
                        st.markdown(f"- {task}")
                    st.markdown(f"**Permissions Required:** {', '.join(row.get('Permissions Required', []))}")
                    st.markdown(f"**Vendors:** {', '.join(row.get('Vendors', []))}")
                    st.markdown(f"**Estimated Labor:** {row.get('Estimated Labor', 'N/A')} workers")
          
          


            # --- Summary Table: Duration & Cost by Phase + Subphase ---
            st.subheader("📊 Summary Table: Duration & Cost by Phase + Subphase")


            def format_duration(val):
                if val == int(val):
                    return str(int(val))  # no decimals if whole number
                else:
                    return str(round(val, 2))  # up to 2 decimals if needed


            rows = []
            for phase in detailed_df.itertuples():
                subphases = json_data[phase.Index].get("Subphase Breakdown", [])
                total_duration = sum(sp.get("Duration (weeks)", 0) for sp in subphases)
                total_cost = sum(sp.get("Cost (USD)", 0) for sp in subphases)


                # Phase total row
                rows.append({
                    "Phase/Subphase": phase.Phase,
                    "Duration (weeks)": format_duration(total_duration),
                    "Cost (USD)": total_cost
                })


                # Subphase rows
                for sp in subphases:
                    rows.append({
                        "Phase/Subphase": f"   ↳ {sp['Name']}",
                        "Duration (weeks)": format_duration(sp["Duration (weeks)"]),
                        "Cost (USD)": sp["Cost (USD)"]
                    })


            summary_df = pd.DataFrame(rows)


            # Format Phase rows bold in st.dataframe using styler
            def highlight_phase(row):
                if not row["Phase/Subphase"].startswith("   ↳"):
                    return ['font-weight: bold; background-color: #D9EAD3'] * len(row)
                else:
                    return [''] * len(row)


            summary_df_styled = summary_df.style.apply(highlight_phase, axis=1)\
                .format({"Cost (USD)": "${:,.2f}"})


            st.dataframe(summary_df_styled, use_container_width=True)


            # Add export to Excel button
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)


                workbook  = writer.book
                worksheet = writer.sheets['Summary']


                # Format header
                header_format = workbook.add_format({'bold': True, 'bg_color': '#B7DEE8'})
                for col_num, value in enumerate(summary_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)


                # Format phase rows bold + background color
                phase_format = workbook.add_format({'bold': True, 'bg_color': '#D9EAD3'})
                for i, val in enumerate(summary_df["Phase/Subphase"]):
                    if not val.startswith("   ↳"):
                        worksheet.set_row(i + 1, None, phase_format)


                # Format cost column as currency
                currency_format = workbook.add_format({'num_format': '$#,##0.00'})
                worksheet.set_column('C:C', 15, currency_format)
                # Optionally format Duration column width
                worksheet.set_column('B:B', 18)


            output.seek(0)
            st.download_button(
                label="📥 Download Summary as Excel",
                data=output,
                file_name="solace_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

