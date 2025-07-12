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

# === Paths ===
MODEL_DIR = "models/"
ASSETS_DIR = "assets/"
LOGO_PATH = os.path.join(ASSETS_DIR, "Solace_logo.png")

# === Load models and encoders with error handling ===
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

# === Mistral API Key from Streamlit Secrets ===
mistral_api_key = st.secrets["mistral_api_key"]
client = Mistral(api_key=mistral_api_key)

# === Phase Mapping ===
phase_mapping = {
    "Scope": "I. Scope",
    "Design": "II. Design",
    "CM": "III. CM - Construction Management",
    "CM,Art,F&E": "IV. CM, Art & FE",
    "CM,F&E": "V. CM & FE",
    "F&E": "VI. F&E",
    "Purch & Install": "VII. Purch & Install",
    "Construction": "VIII. Construction"
}

# === Helper Functions ===
def get_detailed_plan_from_mistral(description):
    prompt = f'''
Generate a detailed construction plan for all 8 phases for the following project: "{description}"
Each phase should include: a short description, 6–10 subtasks, required NYC government permissions (e.g., SCA, DoE, FDNY), 1–2 relevant vendors, and estimated labor size.
Return the result as a JSON list of 8 items in this format:
[
  {{
    "Phase": "I. Scope",
    "Description": "...",
    "Subtasks": ["task1", "task2", ...],
    "Permissions Required": ["SCA", "FDNY"],
    "Vendors": ["VendorX", "VendorY"],
    "Estimated Labor": 12
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
        "project_status": "PI",
        "timeline_status": "Incomplete"
    }])
    embedding = bert_model.encode(df["description_no_stopwords"].tolist())
    cat_feats = ohe_duration.transform(df[["Project Phase Name", "project_status", "timeline_status"]])
    return np.hstack([embedding, cat_feats])

def prepare_single_row(description, phase, duration_weeks):
    df = pd.DataFrame([{
        "Project Phase Name": phase,
        "project_status": "PI",
        "timeline_status": "Incomplete",
        "end_date_missing": True,
        "duration_days": duration_weeks * 7
    }])
    embedding = bert_model.encode([description])
    cat_feats = ohe.transform(df[["Project Phase Name", "project_status", "timeline_status", "end_date_missing"]])
    num_feats = scaler.transform(df[["duration_days"]])
    return np.hstack([embedding, cat_feats, num_feats])

# === Streamlit UI ===
st.set_page_config(page_title="Solace Estimator", layout="wide")
logo = Image.open(LOGO_PATH)
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=180)
with col2:
    st.markdown("<h2 style='margin-bottom: 0; color: #1E90FF;'> NYC School Construction Cost & Schedule Estimator</h2>", unsafe_allow_html=True)

description = st.text_area("Enter Project Description:", "New science lab construction for high school in Manhattan with latest equipment.")
bucket = st.selectbox("Select Cost Bucket:", ["low", "mid", "high"])

if st.button("Estimate Cost and Schedule", key="run_button"):
    with st.spinner("Generating assistant output and predictions..."):
        detailed_json = get_detailed_plan_from_mistral(description)

        try:
            json_start = detailed_json.find("[")
            json_end = detailed_json.rfind("]") + 1
            json_data = json.loads(detailed_json[json_start:json_end])
            detailed_df = pd.DataFrame(json_data)
        except Exception as e:
            st.error(f"❌ Could not parse the detailed plan response: {e}")
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
            st.subheader("🔍 Cost & Schedule Breakdown")
            st.dataframe(result_df, use_container_width=True)
            st.success(f"💰 Total Estimated Cost: ${total_cost:,.2f}")
            st.info(f"🕒 Total Estimated Duration: {total_duration:.1f} weeks")

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.pie(result_df["Predicted Cost (USD)"], labels=result_df["Phase"], autopct='%1.1f%%')
            ax1.set_title("Cost Distribution by Phase", fontsize=14)
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(result_df["Phase"], result_df["Predicted Duration (weeks)"], marker='o', linestyle='-', color='skyblue')
            ax2.set_title("Predicted Duration by Phase", fontsize=14)
            ax2.set_ylabel("Duration (weeks)", fontsize=12)
            ax2.set_xlabel("Phase", fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)

        with tab2:
            st.subheader("📋 Detailed Construction Plan")
            for _, row in detailed_df.iterrows():
                with st.expander(f"{row['Phase']} – {row['Description']}"):
                    st.markdown("**Subtasks:**")
                    for task in row.get("Subtasks", []):
                        st.markdown(f"- {task}")
                    st.markdown(f"**Permissions Required:** {', '.join(row.get('Permissions Required', []))}")
                    st.markdown(f"**Vendors:** {', '.join(row.get('Vendors', []))}")
                    st.markdown(f"**Estimated Labor:** {row.get('Estimated Labor', 'N/A')} workers")

    
