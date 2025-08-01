
import pandas as pd
import numpy as np
import streamlit as st
from mistralai import Mistral, UserMessage, SystemMessage, AssistantMessage
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import re
import time
import os
from streamlit_lottie import st_lottie
import requests
import io
from sentence_transformers import SentenceTransformer
import pickle
import copy

# --- Load Models ---
MODEL_DIR = "models"
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

# === Phase Mapping ===
phase_mapping = {
    "I. Scope": "I. Site Preperation",
    "II. Design": "II. Foundation",
    "III. Commissioning": "III. Commissioning",
    "IV. Purch & Install": "IV. Purch & Install",
    "V. Construction": "V. Construction"
}

if "project_type" not in st.session_state:
    st.session_state.project_type = None
if "cost_bucket" not in st.session_state:
    st.session_state.cost_bucket = None
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

ASSETS_DIR = "assets/"
LOGO_PATH = os.path.join(ASSETS_DIR, "Solace_logo.png")
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
    st.markdown("🚧 *Project Management made easy*")
    st.markdown("---")
    st.markdown("Created for Solace Technologies")
    st.markdown("🔗 [GitHub Repo](https://github.com/AnushkaKatiyar)")
    st.markdown("💬 Powered by Mistral + ML Models")

# === Model Setup ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load API key from Streamlit secrets
mistral_api_key = st.secrets["mistral_api_key"]
client = Mistral(api_key=mistral_api_key)
st.set_page_config(page_title="AI Chatbot Assistant", layout="wide")

st.markdown("""
<div style="
    max-width: 800px;
    margin: 20px auto 40px auto;
    padding: 25px 30px;
    text-align: center;
    background-color: #cce6ff;
    border-radius: 12px;
    box-shadow: 0 6px 15px rgba(0, 102, 204, 0.3);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
">
    <h1 style="color: #004080; margin-bottom: 8px; font-weight: 700; font-size: 2.8rem;">
        🤖 AI Assistant for Project Management
    </h1>
    <h4 style="color: #004080; margin-top: 0; font-weight: 400;">
        What type of project are you planning?
    </h4>
</div>
""", unsafe_allow_html=True)


# Store selection in session state
if "project_type" not in st.session_state:
    st.session_state.project_type = None

# Add some spacing
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)


# Custom CSS
st.markdown("""
    <style>
    /* Style buttons */
    .stButton > button {
        background-color: #1E90FF !important;
        color: white !important;
        padding: 12px 20px;
        border: none;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        width: 250px;
        transition: 0.3s ease-in-out;
        display: block;
        margin: auto;
    }
    .stButton > button:hover {
        background-color: #1C86EE !important;
        transform: scale(1.04);
    }

    /* Center images in columns */
    .image-container {
        display: flex;
        justify-content: center;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

if st.session_state.project_type is None:
    image_width = 100

    # Row 1: Buttons
    spacer1, col1, spacer2 = st.columns([0.5, 2, 0.5])
    with col1:
        if st.button(" New Project", key="new"):
            st.session_state.project_type = "new"
            st.session_state.cost_bucket = "high"
        
    # with col2:
    #     if st.button(" Upgrades", key="upgrade"):
    #         st.session_state.project_type = "upgrade"
    #         st.session_state.cost_bucket = "mid"
        
    # with col3:
    #     if st.button(" Repair & Maintenance", key="repair"):
    #         st.session_state.project_type = "repair"
    #         st.session_state.cost_bucket = "low"
        

    # # Row 2: Images (center aligned)
    # spacer1_img, img_col1, img_col2, img_col3, spacer2_img = st.columns([1, 2, 2, 1.3, 0.4])
    # with img_col1:
    #     st.markdown('<div class="image-container">', unsafe_allow_html=True)
    #     st.image("assets/New_Construction.jpg", width=image_width)
    #     st.markdown('</div>', unsafe_allow_html=True)
    # with img_col2:
    #     st.markdown('<div class="image-container">', unsafe_allow_html=True)
    #     st.image("assets/Upgrade.png", width=image_width)
    #     st.markdown('</div>', unsafe_allow_html=True)
    # with img_col3:
    #     st.markdown('<div class="image-container">', unsafe_allow_html=True)
    #     st.image("assets/Repair.jpg", width=image_width)
    #     st.markdown('</div>', unsafe_allow_html=True)

# Show content based on selection
st.markdown("---")
###################################################################
###################################################################
###################################################################
###################################################################
if st.session_state.project_type == "new":
    st.subheader("New Project Planning")
    # your existing pipeline goes here (assistant, model predictions, etc.)

    def animated_typing(message, delay=0.03):
        placeholder = st.empty()
        full_text = ""
        for char in message:
            full_text += char
            placeholder.markdown(f"**{full_text}**")
            time.sleep(delay)

    if "has_seen_welcome" not in st.session_state:
        st.session_state.has_seen_welcome = True
        with st.chat_message("assistant"):
            animated_typing("Hi, Welcome to Solace AI Project Manager Demo 👋\n\nI can generate project plan for a new school development in New York City based on your requirements. This is the scope of the demo.\n\n Can I please help make the plan for you? ")

    # Define the questions to ask sequentially
    questions = [
        ("ProjectDescription", "Please describe the project in a few sentences."),
        ("Location", "Which part of NYC is the school located in?"),
        ("Grades", "How many grades will the school have?"),
        ("StudentsPerClass", "What is the average number of students per class?"),
        ("Timeline", "What is the expected construction timeline (in months)?"),
        ("SquareFootage", "What is the square footage of the construction?"),
        ("SpecialReqs", "Are there any special facilities or requirements needed?"),       
        # ("Floors", "How many floors will the building have?"),
        # ("DemolitionNeeded", "Is demolition needed?"),
        # ("Basement", "Is a basement needed?"),
    ]

    # Initialize session state for collected info and chat history
    if "collected_info" not in st.session_state:
        st.session_state.collected_info = {key: None for key, _ in questions}
        

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "final_plan" not in st.session_state:
        st.session_state.final_plan = None

    # Track the last asked question key so we know where to save answer
    if "last_question_key" not in st.session_state:
        st.session_state.last_question_key = None


    # Function to find the next unanswered question
    def get_next_question():
        for key, question in questions:
            if st.session_state.collected_info[key] in [None, ""]:
                return key, question
        return None, None

    client = Mistral(api_key=st.secrets["mistral_api_key"])

    # Capture user input
    user_input = st.chat_input("Type your answer here...")

    if user_input:
        # Save user input as answer to the last asked question
        if st.session_state.last_question_key is not None:
            st.session_state.collected_info[st.session_state.last_question_key] = user_input

        # Append user message to chat history
        st.session_state.chat_history.append(UserMessage(content=user_input))

        # Find the next question to ask
        next_key, next_question = get_next_question()
        st.session_state.last_question_key = next_key

        # Compose system prompt with current collected info and next question to ask
        if next_question:
            system_prompt = f"""
    You are an expert NYC school construction planner assistant.

    Current collected info:
    {json.dumps(st.session_state.collected_info, indent=2)}

    Ask only the questions defined (they have been predefined, so you will get the questions, no need to add too much) and wait for user response, do not repeat question. 
    Do not display unnecessary information or the previous questions asked. Do provide like the average for each value, like average square foot required for a school construction and average class size etc.
    Do not display json to the user.
    Also tell the user that they are advised to answer the questions asked and can provide more information for context but they will be asked the guided questions.
    Next question:
    {next_question}
    """
        else:
            system_prompt = f"""
    You have collected all the necessary project information:
    {json.dumps(st.session_state.collected_info, indent=2)}
    Display in formatted way, not json
    Inform the user that all info is collected and ask if they want to generate the construction plan.
    Ask each question only once, do not repeat the previous question, ask only the defined 7 questions.
    """

        # Compose messages to send to the model
        messages = [SystemMessage(content=system_prompt)] + st.session_state.chat_history

        # Call the Mistral model
        response = client.chat.complete(
            model="mistral-medium",
            messages=messages,
        )
        assistant_reply = response.choices[0].message.content.strip()

        # Append assistant reply to chat history
        st.session_state.chat_history.append(AssistantMessage(content=assistant_reply))

    # Display the full chat history
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, UserMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # When all questions answered, show button to generate plan
    next_key, next_question = get_next_question()
    if next_key is None:
        if st.button("🚧 Generate Project Plan"):
            summary_prompt = f"""
    Using the collected info, generate a detailed construction plan in JSON format with phases, subtasks, vendors, permissions, materials, and labor.

    Output should be a list of 5 phases, depending on the user inputs. Each phase must include:
    - Phase: (string) e.g. "I. Site Preperation", "II. Foundation", "III. Comissioning", "IV. Purchase & Install", "V. Construction",
    - Description: (string),a short description,
    - Subphases/subtaskes: 5-10 sub tasks within the phases
    - Subphase Breakdown: (list of phases and subtasks(5-10 phases and 5-10 subtasks) from above as dicts). Each dict must have:
    - Name: (string)
    - Description(string)
    - Cost (USD): (number)
    - Labor Category
    - Vendor: (list of strings),1–2 **only actual NYC-based vendors or well-known relevant companies, not made up names** (avoid placeholders like 'VendorX', 'VendorA'),
    - Permission if needed: (list of strings),required NYC government permissions (e.g., SCA, DoE, FDNY),
    - Duration (weeks): (number)- Please predict realistic numbers based on actual construction timelines and if the user has provided a timeline, try to get the values in that ballpark but if they are unrealistic, then predict normal values,
    - Resources & Material-Raw materials used in construction
    - Item-should have the name and describe for which phases and subtask it is needed
    - Quantity-number followed by correct units e.g-metric tonne, feet etc
    - Cost (USD): (number), please predict realistic numbers and should not exceed 60% of the total estimated cost(the total of all the resources should be under 12 million)
    

    Collected info:
    {json.dumps(st.session_state.collected_info, indent=2)}

    Only output JSON with this structure:
    {{ 
    "ConstructionPhases": [
        {{
        "PhaseName": "string",
        "Description": "string",
        "EstimatedCost": number,
        "DurationEstimate": number,
        "Subtasks": [
            {{
            "SubtaskName": "string",
            "Description": "string",
            "CostEstimate": number,
            "DurationEstimate": number,
            "LaborCategories": [],
            "Vendors": [],
            "Permissions": []
            }}
        ],
        "LaborCategories": [],
        "Vendors": [],
        "Permissions": []
        }}
    ],
    "Resources & Materials": {{
        "CategoryName": [
        {{
            "Item": "string",
            "QuantityEstimate": "string",
            "EstimatedCost": number
        }}
        ]
    }}
    }}
    No extra explanation.
    """
            messages = [
                SystemMessage(content="You summarize the project info and generate the final JSON plan."),
                UserMessage(content=summary_prompt),
            ]
            response = client.chat.complete(
                model="mistral-medium",
                messages=messages,
            )
            final_json = response.choices[0].message.content.strip()
            st.session_state.final_plan = final_json

        description = st.session_state.collected_info.get("ProjectDescription", "")
        phases_json_str = json.dumps(phase_mapping, indent=2)
        ai_input = f"""
        Based on the following project description, estimate the expected duration in weeks for each of the following construction phases, answer in numeric, no ranges:

        Phases:
        {phases_json_str}

        Project Description:
        {description}

        Reply in this format (JSON):
        {{
            "I. Scope": "<duration in weeks>",
            "II. Design": "<duration in weeks>",
            "III. Commissioning": "<duration in weeks>",
            "IV. Purch & Install": "<duration in weeks>",
            "V. Construction": "<duration in weeks>"
        }}
        """
        messages = [
            SystemMessage(content="You are an expert NYC school construction planner."),
            UserMessage(content=ai_input)
        ]
        response = client.chat.complete(
            model="mistral-medium",
            messages=messages
        )
        response_text = response.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            clean_json = match.group(0)
            ai_durations = json.loads(clean_json)
        else:
            print("JSON not found in AI response")
            ai_durations = {}
        def predict_cost_duration(description, bucket, ai_durations):
            predictions = []

            for phase_code, display_name in phase_mapping.items():
                # Get AI-estimated duration (in weeks)
                try:
                    raw_val = ai_durations.get(phase_code, "0")
                    # Remove any non-numeric part like " weeks"
                    numeric_str = "".join(c for c in raw_val if c.isdigit() or c == ".")

                    duration_weeks = float(numeric_str) if numeric_str else 0
                except (ValueError, TypeError):
                    duration_weeks = 0  # fallback if parsing fails

                # Prepare cost features using AI duration
                X_cost = prepare_single_row(description, phase_code, duration_weeks)
                model = model_dict[bucket]
                cost = model.predict(X_cost)[0]

                predictions.append({
                    "Phase": display_name,
                    "Predicted Duration (weeks)": round(duration_weeks, 2),
                    "Predicted Cost (USD)": round(max(cost, 0), 2),
                })

            result_df = pd.DataFrame(predictions)
            return result_df
        
    def clean_json_string(raw_json):
        return raw_json.strip().removeprefix("```json").removesuffix("```").strip()

    def safe_format_cost(x):
        try:
            return "${:,.0f}".format(float(x))
        except (ValueError, TypeError):
            return str(x)

    if st.session_state.final_plan:
        # If it's still a string, clean and parse it
        if isinstance(st.session_state.final_plan, str):
            def clean_json_string(raw_json):
                return raw_json.strip().removeprefix("```json").removesuffix("```").strip()
            
            cleaned = clean_json_string(st.session_state.final_plan)
            try:
                parsed_json = json.loads(cleaned)
                st.session_state.final_plan = parsed_json
            except json.JSONDecodeError as e:
                st.error(f"JSON decode failed: {e}")
                st.stop()

        # Now it's a proper dict in session state — ready for rendering
        final_plan = st.session_state.final_plan

    if "final_plan" in st.session_state and st.session_state.final_plan is not None:
        plan = st.session_state.final_plan
        phases = plan.get("ConstructionPhases", [])
        st.divider()
        st.markdown(
                """
                <div style="
                    display: inline-block;
                    padding: 8px 20px;
                    border-top-left-radius: 10px;
                    border-top-right-radius: 10px;
                    background-color: #0077b6;  /* nice blue tab color */
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    font-family: sans-serif;
                    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
                    margin-bottom: -2px;
                ">
                    ML-Based Cost & Schedule Estimates
                </div>
                """,
                unsafe_allow_html=True,
            )               
        description = st.session_state.collected_info.get("ProjectDescription", "") 
        bucket = st.session_state.get("bucket", "high")  # fallback to high
        with st.spinner("Running prediction model..."):
            try:
                result_df = predict_cost_duration(description, bucket,ai_durations)
                total_cost = result_df["Predicted Cost (USD)"].sum()
                total_duration = result_df["Predicted Duration (weeks)"].sum()
                result_df["Predicted Cost Raw"] = result_df["Predicted Cost (USD)"]
                result_df["Predicted Cost (USD)"] = result_df["Predicted Cost (USD)"].apply(lambda x: f"${x:,.2f}")
                result_df["Duration"] = result_df["Predicted Duration (weeks)"].apply(
                    lambda w: f"{int(w)} weeks {int((w % 1) * 7)} days"
                )
                st.dataframe(result_df[["Phase", "Predicted Cost (USD)", "Duration"]], use_container_width=True)
                col1, col2 = st.columns(2)
                col1.metric("Total Estimated Cost", f"${total_cost:,.2f}")
                col2.metric("Total Estimated Duration", f"{total_duration:.1f} weeks")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        st.markdown(
            """
            <div style="margin: 30px 0 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 10px; font-size: 18px; color: grey; font-style: italic;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 18px; margin-right: 8px;">ℹ️</span>
                    <span><b>Note:</b> These predictions are based on a machine learning model trained using past school construction projects.
                    The model's average margin of error is about ±$651,000, and its accuracy is around 98.8%. </span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 18px; margin-right: 8px;">⏱️</span>
                    <span><b>Duration estimate:</b> The predicted duration is a simple total of all individual task durations. 
                    In real-world projects, some tasks may overlap or depend on each other, so the actual timeline may be shorter or longer.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
#########################################################################   
        st.markdown(
            """
            <div style="
                display: inline-block;
                padding: 8px 20px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                background-color: #0077b6;  /* nice blue tab color */
                color: white;
                font-size: 20px;
                font-weight: bold;
                font-family: sans-serif;
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
                margin-bottom: -2px;
            ">
                Project Phases & Subtasks
            </div>
            """,
            unsafe_allow_html=True,
        )

        total_predicted_cost = 0
        total_predicted_duration = 0

        if 'result_df' in locals() and not result_df.empty:
            total_predicted_cost = result_df["Predicted Cost Raw"].sum()
            total_predicted_duration = result_df["Predicted Duration (weeks)"].sum()
        else:
            total_predicted_cost = sum(phase.get("EstimatedCost", 0) for phase in phases)
            total_predicted_duration = sum(phase.get("DurationEstimate", 0) for phase in phases)

        # Loop over phases and calculate percentages
        for i, phase in enumerate(phases):
            phase_name = phase["PhaseName"]
            with st.expander(f"📌 {phase_name}", expanded=True):
                rows = []

                phase_cost = phase.get("EstimatedCost", 1e-6)  # avoid div zero
                phase_duration = phase.get("DurationEstimate", 1e-6)

                ml_duration = phase_duration
                ml_cost = phase_cost

                if 'result_df' in locals() and not result_df.empty and i < len(result_df):
                    ml_duration = result_df.iloc[i]["Predicted Duration (weeks)"]
                    ml_cost = result_df.iloc[i]["Predicted Cost Raw"]

                # Convert to percentages of total
                cost_pct = (ml_cost / total_predicted_cost) * 100 if total_predicted_cost else 0
                duration_pct = (ml_duration / total_predicted_duration) * 100 if total_predicted_duration else 0

                rows.append({
                    "Task": f"{phase_name}",
                    "Description": phase.get("Description", ""),
                    "Duration (%)": f"{duration_pct:.1f}%",
                    "Estimated Cost (%)": f"{cost_pct:.1f}%",
                    "Labor Categories": ", ".join(phase.get("LaborCategories", [])),
                    "Vendors": ", ".join(phase.get("Vendors", [])),
                    "Permissions": ", ".join(phase.get("Permissions", [])),
                })

                # Subtasks (indented with arrow)
                for sub in phase.get("Subtasks", []):
                    sub_cost = sub.get("CostEstimate", 0)
                    sub_duration = sub.get("DurationEstimate", 0)

                    cost_pct = (sub_cost / phase_cost) * 100 if phase_cost else 0
                    duration_pct = (sub_duration / phase_duration) * 100 if phase_duration else 0

                    rows.append({
                        "Task": f"  ↳ {sub.get('SubtaskName', '')}",
                        "Description": sub.get("Description", ""),
                        "Duration (%)": f"{duration_pct:.1f}%",
                        "Estimated Cost (%)": f"{cost_pct:.1f}%",
                        "Labor Categories": ", ".join(sub.get("LaborCategories", [])),
                        "Vendors": ", ".join(sub.get("Vendors", [])),
                        "Permissions": ", ".join(sub.get("Permissions", [])),
                    })

                # Build and display DataFrame as usual
                df_phase = pd.DataFrame(rows)
                df_phase["Estimated Cost (%)"] = df_phase["Estimated Cost (%)"].apply(safe_format_cost)
                st.dataframe(df_phase, use_container_width=True)
     
    ####################################################################    
 
        st.markdown(
                """
                <div style="
                    display: inline-block;
                    padding: 8px 20px;
                    border-top-left-radius: 10px;
                    border-top-right-radius: 10px;
                    background-color: #0077b6;  /* nice blue tab color */
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    font-family: sans-serif;
                    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
                    margin-bottom: -2px;
                ">
                    Resources & Materials
                </div>
                """,
                unsafe_allow_html=True,
            )
        # --- STEP 1: Calculate ML model total cost ---
        ml_total_cost = total_cost
        # --- STEP 2: Collect resource/materials and calculate total cost ---
        resources = plan.get("Resources & Materials", {})
        materials_rows = []
        raw_costs = []  # to store raw numeric costs for logic

        for category, items in resources.items():
            for item in items:
                cost = item.get("EstimatedCost", 0)
                raw_costs.append(cost)
                materials_rows.append({
                    "Category": category,
                    "Item": item.get("Item", ""),
                    "Quantity Estimate": item.get("QuantityEstimate", "N/A"),
                    "Estimated Cost": cost  # keep as number for now
                })

        resource_total_cost = sum(raw_costs)
        threshold = 0.6 * ml_total_cost


        # --- STEP 3: Adjust if necessary ---
        if resource_total_cost > threshold:
            delta = resource_total_cost - threshold
            original_total = sum(row["Estimated Cost"] for row in materials_rows)
            for row in materials_rows:
                original_cost = row["Estimated Cost"]
                if original_total > 0:
                    adjustment = (original_cost / original_total) * delta
                else:
                    adjustment = 0
                new_cost = max(1, original_cost - adjustment)  # Enforce minimum $1
                row["Estimated Cost"] = new_cost
        # --- STEP 4: Format costs and display ---
        final_total = sum(row["Estimated Cost"] for row in materials_rows)

        for row in materials_rows:
            row["Estimated Cost"] = f"${row['Estimated Cost']:,.0f}"

        materials_df = pd.DataFrame(materials_rows)
        st.table(materials_df)   
        st.markdown(f"**Total Materials Cost:** ${final_total:,.0f}")     

    ####################################################################
        # Collect labor and vendor info
        all_labors = set()
        all_vendors = set()

        for phase in phases:
            all_labors.update(phase.get("LaborCategories", []))
            all_vendors.update(phase.get("Vendors", []))

            for sub in phase.get("Subtasks", []):
                all_labors.update(sub.get("LaborCategories", []))
                all_vendors.update(sub.get("Vendors", []))

        # Pill rendering function
        def render_pills(title, items, color="#e0f0ff"):
            st.markdown(
                f"""
                <div style="margin-top: 20px;">
                    <h4 style="margin-bottom: 10px;">{title}</h4>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        {''.join(f'<div style="padding: 6px 12px; background-color: {color}; border-radius: 20px; font-size: 14px; color: #333; border: 1px solid #ccc;">{item}</div>' for item in sorted(items))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Display if any data found
        if all_labors or all_vendors:
            st.markdown(
                """
                <div style="
                    display: inline-block;
                    padding: 8px 20px;
                    border-top-left-radius: 10px;
                    border-top-right-radius: 10px;
                    background-color: #0077b6;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    font-family: sans-serif;
                    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
                    margin-bottom: -2px;
                ">
                    Project Resources
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                if all_labors:
                    render_pills("Labor Categories", all_labors, color="#B0C4DE")  # Light green
                else:
                    st.subheader("Labor Categories")
                    st.write("No labor categories found.")

            with col2:
                if all_vendors:
                    render_pills("Vendor Types", all_vendors, color="#ADD8E6")  # Light coral
                else:
                    st.subheader("Vendor Types")
                    st.write("No vendor types found.")
        else:
            st.info("No labor or vendor types found in this plan.")
   
    ####################################################################
        
    ##################################################
        import plotly.express as px
        import pandas as pd

        # Make sure you have your phases data
        if "final_plan" in st.session_state and st.session_state.final_plan:
            plan = st.session_state.final_plan
            phases = plan.get("ConstructionPhases", [])

            # Prepare data lists
            phase_labels = []
            phase_costs = []
            phase_durations = []
            total_cost = 0
            total_duration = 0

            for phase in phases:
                phase_labels.append(phase["PhaseName"])
                cost = phase.get("EstimatedCost", 0)
                duration = phase.get("DurationEstimate", 0)
                total_cost += cost
                total_duration += duration
                phase_costs.append(cost)
                phase_durations.append(duration)

            df = pd.DataFrame({
                "Phase": phase_labels,
                "Cost": phase_costs,
                "Duration": phase_durations,
            })

            # Cost Pie Chart
            st.markdown(
                """
                <div style="
                    display: inline-block;
                    padding: 8px 20px;
                    border-top-left-radius: 10px;
                    border-top-right-radius: 10px;
                    background-color: #0077b6;  /* nice blue tab color */
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    font-family: sans-serif;
                    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
                    margin-bottom: -2px;
                ">
                    💰 Cost Distribution
                </div>
                """,
                unsafe_allow_html=True,
            )
            fig_pie = px.pie(
                df,
                names="Phase",
                values="Cost",
                title="Cost Distribution by Phase",
                hole=0.4,
            )
            # Increase font size and make label text black
            fig_pie.update_traces(
                textposition="outside",
                textinfo="percent+label",
                textfont=dict(size=18, color='black')  # Bigger and black font for labels
            )

            # Make title bigger and black
            fig_pie.update_layout(
                title=dict(
                    text="Cost Distribution by Phase",
                    font=dict(size=20, color='black')
                )
            )

            st.plotly_chart(fig_pie, use_container_width=True)

            # Duration Line Chart
            st.markdown(
                """
                <div style="
                    display: inline-block;
                    padding: 8px 20px;
                    border-top-left-radius: 10px;
                    border-top-right-radius: 10px;
                    background-color: #0077b6;  /* nice blue tab color */
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    font-family: sans-serif;
                    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
                    margin-bottom: -2px;
                ">
                    ⏱ Duration by Phase 
                </div>
                """,
                unsafe_allow_html=True,
            )
            fig_line = px.line(
                df,
                x="Phase",
                y="Duration",
                markers=True,
                title="Duration by Phase",
            )
            fig_line.update_layout(
            margin=dict(l=40, r=20, t=50, b=80),
            xaxis=dict(
                tickangle=-45,
                title=dict(
                    text="Phase",
                    font=dict(color='black', size=20)
                ),
                tickfont=dict(color='black', size=18),
                showline=True,
                linecolor='black',
                showgrid=True,
                gridcolor='lightgrey',
            ),
            yaxis=dict(
                title=dict(
                    text="Duration (weeks)",
                    font=dict(color='black', size=20)
                ),
                tickfont=dict(color='black', size=18),
                showline=True,
                linecolor='black',
            )
        )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No construction phases data available.")
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
# elif project_type == "🚧 Upgrades":
#     st.subheader("Upgrade Planning")
# elif st.session_state.project_type == "upgrade":
#     st.subheader("Upgrade Project Planning")
#     st.info("🚧 We're here to help you upgrade existing facilities.")
#     def animated_typing(message, delay=0.03):
#         placeholder = st.empty()
#         full_text = ""
#         for char in message:
#             full_text += char
#             placeholder.markdown(f"**{full_text}**")
#             time.sleep(delay)
#     if "has_seen_upgrade_welcome" not in st.session_state:
#         st.session_state.has_seen_upgrade_welcome = True
#         with st.chat_message("assistant"):
#             animated_typing("Hey there 👋\n\nLet's plan your school upgrade project! Here are some examples to inspire you:")
#             st.markdown("""
#             **Examples:**
#             - Smart Board or AV System Installations  
#             - HVAC System Upgrade  
#             - Bathroom Modernization  
#             - Security System Upgrades  
#             - Solar Panel Installation  
#             - LED Lighting Retrofit  
#             - Accessibility Improvements  
#             - IT Infrastructure Overhaul  
#             - Library Renovation  
#             - Playground Equipment Upgrade  
#             - Kitchen/Cafeteria Modernization  
#             - Fire Suppression System Upgrades  
#             """)

#     # Define upgrade-specific questions
#     upgrade_questions = [
#         ("UpgradeDescription", "What kind of upgrade are you planning?"),
#         ("TargetArea", "Which part of the school is being upgraded (e.g., library, HVAC, playground)?"),
#         ("ImprovementGoal", "What is the intended benefit or goal of this upgrade (e.g., energy savings, accessibility)?"),
#         ("OccupiedStatus", "Is the building currently in use during the upgrade?"),
#         ("InfrastructureLimitations", "Are there infrastructure or power limitations to consider?"),
#         ("Timeline", "What is your desired completion timeline (in weeks)?"),
#     ]

#     # Init state
#     if "upgrade_info" not in st.session_state:
#         st.session_state.upgrade_info = {key: None for key, _ in upgrade_questions}
#     if "upgrade_chat" not in st.session_state:
#         st.session_state.upgrade_chat = []
#     if "upgrade_last_q" not in st.session_state:
#         st.session_state.upgrade_last_q = None
#     if "upgrade_plan" not in st.session_state:
#         st.session_state.upgrade_plan = None

#     def get_next_upgrade_question():
#         for key, q in upgrade_questions:
#             if st.session_state.upgrade_info[key] in [None, ""]:
#                 return key, q
#         return None, None

#     upgrade_input = st.chat_input("Describe your upgrade project...")

#     if upgrade_input:
#         if st.session_state.upgrade_last_q:
#             st.session_state.upgrade_info[st.session_state.upgrade_last_q] = upgrade_input
#         st.session_state.upgrade_chat.append(UserMessage(content=upgrade_input))
#         next_key, next_q = get_next_upgrade_question()
#         st.session_state.upgrade_last_q = next_key

#         if next_q:
#             prompt = f"""
#             You are an expert NYC school **upgrade** planner.
#             Ask only the defined upgrade questions.
#             Collected so far:
#             {json.dumps(st.session_state.upgrade_info, indent=2)}

#             Ask only the next missing question:
#             {next_q}
#             """
#         else:
#             prompt = f"""
#             All necessary upgrade info collected:
#             {json.dumps(st.session_state.upgrade_info, indent=2)}

#             Inform the user and ask if you'd like to generate a detailed upgrade plan with phases, subtasks, vendors, costs, labor, and materials.
#             """

#         messages = [SystemMessage(content=prompt)] + st.session_state.upgrade_chat
#         reply = client.chat.complete(model="mistral-small", messages=messages)
#         reply_text = reply.choices[0].message.content.strip()
#         st.session_state.upgrade_chat.append(SystemMessage(content=reply_text))

#     for msg in st.session_state.upgrade_chat:
#         role = "user" if isinstance(msg, UserMessage) else "assistant"
#         with st.chat_message(role):
#             st.markdown(msg.content)

#     next_key, _ = get_next_upgrade_question()
#     if next_key is None:
#         st.session_state.collected_info = st.session_state.upgrade_info.copy()
#         if st.button("⚙️ Generate Upgrade Plan"):
#             upgrade_summary_prompt = f"""
#             Using the collected info, generate a detailed upgrade construction plan in **valid JSON format only**. Follow the structure exactly.

#             Your JSON must include:
#             1. "ConstructionPhases": array of 5–10 phases with:
#                 - "PhaseName", "Description", "EstimatedCost", "DurationEstimate"
#                 - "Subtasks": each with SubtaskName, Description, CostEstimate, DurationEstimate, LaborCategories, Vendors, Permissions
#                 - "LaborCategories", "Vendors", "Permissions Required"

#             2. "ResourcesAndMaterials": array of materials with:
#                 - "Category", "Item", "QuantityEstimate", "EstimatedCost"

#             ❗ JSON ONLY. No explanation or markdown.

#             User Info:
#             {json.dumps(st.session_state.collected_info, indent=2)}

#             Respond with just the JSON:
#             {{
#             "ConstructionPhases": [...],
#             "ResourcesAndMaterials": [...]
#             }}
#             """
#             messages = [
#                 SystemMessage(content="You generate upgrade plans in JSON."),
#                 UserMessage(content=upgrade_summary_prompt),
#             ]
#             response = client.chat.complete(model="mistral-small", messages=messages)
#             response_str = response.choices[0].message.content.strip()
#             st.session_state.upgrade_plan_raw = response_str
#             st.session_state.upgrade_plan = response_str
#             st.session_state.upgrade_plan_parsed = None

#     if st.session_state.upgrade_plan:
#         if "upgrade_plan_parsed" not in st.session_state:
#             st.session_state.upgrade_plan_parsed = None

#         if st.session_state.upgrade_plan_raw and st.session_state.upgrade_plan_parsed is None:
#             raw_json_str = st.session_state.upgrade_plan_raw.strip().removeprefix("```json").removesuffix("```").strip()
#             try:
#                 parsed = json.loads(raw_json_str)
#                 st.session_state.upgrade_plan_parsed = parsed
#             except Exception as e:
#                 st.error("Invalid JSON: " + str(e))
#                 st.stop()

#         if st.session_state.upgrade_plan_parsed:
#             final = st.session_state.upgrade_plan_parsed
#         else:
#             st.info("No valid upgrade plan found.")

#         st.subheader("📈 Final Upgrade Plan")

#         def safe_format_cost(cost):
#             try:
#                 return f"${float(cost):,.2f}"
#             except (ValueError, TypeError):
#                 return "N/A"

#         phases = final.get("ConstructionPhases", [])
#         for phase in phases:
#             with st.expander(f"📌 {phase['PhaseName']}", expanded=True):
#                 rows = [{
#                     "Task": phase["PhaseName"],
#                     "Description": phase.get("Description", ""),
#                     "Estimated Cost ($)": safe_format_cost(phase.get("EstimatedCost", 0)),
#                     "Duration (weeks)": phase.get("DurationEstimate", 0),
#                     "Labor Categories": ", ".join(phase.get("LaborCategories", [])),
#                     "Vendors": ", ".join(phase.get("Vendors", [])),
#                     "Permissions": ", ".join(phase.get("Permissions Required", [])),
#                 }]
#                 for sub in phase.get("Subtasks", []):
#                     rows.append({
#                         "Task": f"  ↳ {sub.get('SubtaskName', '')}",
#                         "Description": sub.get("Description", ""),
#                         "Estimated Cost ($)": safe_format_cost(sub.get("CostEstimate", 0)),
#                         "Duration (weeks)": sub.get("DurationEstimate", 0),
#                         "Labor Categories": ", ".join(sub.get("LaborCategories", [])),
#                         "Vendors": ", ".join(sub.get("Vendors", [])),
#                         "Permissions": ", ".join(sub.get("Permissions", [])),
#                     })
#                 st.dataframe(pd.DataFrame(rows), use_container_width=True)

#         st.subheader("🧱 Upgrade Resources & Materials")
#         resources = final.get("ResourcesAndMaterials", [])
#         mat_rows = [{
#             "Category": item.get("Category", ""),
#             "Item": item.get("Item", ""),
#             "Quantity Estimate": item.get("QuantityEstimate", ""),
#             "Estimated Cost": safe_format_cost(item.get("EstimatedCost", 0)),
#         } for item in resources]
#         st.dataframe(pd.DataFrame(mat_rows))

#         df_chart = pd.DataFrame({
#             "Phase": [p["PhaseName"] for p in phases],
#             "Cost": [p.get("EstimatedCost", 0) for p in phases],
#             "Duration": [p.get("DurationEstimate", 0) for p in phases],
#         })

#         st.subheader("💰 Cost by Upgrade Phase")
#         st.plotly_chart(px.pie(df_chart, names="Phase", values="Cost", title="Cost Distribution", hole=0.4), use_container_width=True)

#         st.subheader("⏱ Timeline by Phase")
#         fig = px.line(df_chart, x="Phase", y="Duration", markers=True)
#         fig.update_layout(yaxis_title="Weeks", xaxis_tickangle=-45)
#         st.plotly_chart(fig, use_container_width=True)

        

#         st.markdown(f"**Total Estimated Cost:** ${int(df_chart['Cost'].sum()):,}")
#         def parse_duration_to_weeks(duration_str):
#             if isinstance(duration_str, str):
#                 match = re.search(r"(\d+)", duration_str)
#                 if match:
#                     num = int(match.group(1))
#                     if "month" in duration_str.lower():
#                         return num * 4
#                     elif "week" in duration_str.lower():
#                         return num
#                     elif "day" in duration_str.lower():
#                         return round(num / 7, 2)
#             return None
#         df_chart["Duration_Weeks"] = df_chart["Duration"].apply(parse_duration_to_weeks)
#         valid_durations = df_chart["Duration_Weeks"].dropna()
        
# ################################################################################
#         st.divider()
#         st.subheader("🧮 ML-Based Cost & Schedule Estimates")

#         description = st.session_state.collected_info.get("UpgradeDescription", "")  
#         bucket = st.session_state.get("bucket", "mid")  # fallback to mid

#         if st.button("Estimate Cost and Schedule (ML)", key="ml_estimate_button"):
#             with st.spinner("Running prediction model..."):
#                 try:
#                     result_df = predict_cost_duration(description, bucket,ai_durations)

#                     total_cost = result_df["Predicted Cost (USD)"].sum()
#                     total_duration = result_df["Predicted Duration (weeks)"].sum()

#                     result_df["Predicted Cost (USD)"] = result_df["Predicted Cost (USD)"].apply(lambda x: f"${x:,.2f}")
#                     result_df["Duration"] = result_df["Predicted Duration (weeks)"].apply(
#                         lambda w: f"{int(w)} weeks {int((w % 1) * 7)} days"
#                     )

#                     st.dataframe(result_df[["Phase", "Predicted Cost (USD)", "Duration"]], use_container_width=True)

#                     col1, col2 = st.columns(2)
#                     col1.metric("💰 Total Estimated Cost", f"${total_cost:,.2f}")
#                     col2.metric("🕒 Total Estimated Duration", f"{total_duration:.1f} weeks")

#                 except Exception as e:
#                     st.error(f"Prediction failed: {e}")

#         if not valid_durations.empty:
#             st.markdown(f"**Total Estimated Duration:** {int(valid_durations.sum())} weeks")
#         else:
#             st.warning("⚠️ No valid duration data found.")


       
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
# elif project_type == "🛠 Repair & Maintenance":
#     st.subheader("Repair / Maintenance Planning")
# elif st.session_state.project_type == "repair":
#     st.subheader("Repair & Maintenance Planning")
#     st.info("🛠 Let’s get those repairs underway!")

#     def animated_typing(message, delay=0.03):
#         placeholder = st.empty()
#         full_text = ""
#         for char in message:
#             full_text += char
#             placeholder.markdown(f"**{full_text}**")
#             time.sleep(delay)

#     if "has_seen_repair_welcome" not in st.session_state:
#         st.session_state.has_seen_repair_welcome = True
#         with st.chat_message("assistant"):
#             animated_typing("Hi there 👋\n\nI'm here to help you plan your school repair or maintenance project.\n\nHere are a few examples to get you started:")

#             st.markdown("""
#             **Examples:**
#             - Boiler Repair (leaking or outdated)
#             - Roof Leak Repair
#             - Mold Remediation
#             - Broken Window Replacement
#             - Fire Alarm System Fix
#             - Pest Control
#             - Elevator Repair
#             - Emergency Plumbing (e.g., burst pipes)
#             - Lead Paint Stabilization
#             - Cracked Sidewalk Repair
#             - Ceiling Tile Replacement
#             - Lighting Fixture Repairs
#             - HVAC Maintenance
#             - Asbestos Abatement
#             """)

#     # Define repair questions
#     repair_questions = [
#         ("RepairDescription", "What kind of repair or maintenance is needed?"),
#         ("Location", "Which area of the school is affected (e.g., cafeteria, roof, classroom)?"),
#         ("Urgency", "Is this an emergency repair or scheduled maintenance?"),
#         ("BuildingStatus", "Is the building currently occupied or vacant?"),
#         ("AccessConstraints", "Are there access or safety concerns (e.g., asbestos, confined spaces)?"),
#         ("Timeline", "What is your desired timeline (in weeks)?"),
#     ]

#     # Init state
#     if "repair_info" not in st.session_state:
#         st.session_state.repair_info = {key: None for key, _ in repair_questions}
#     if "repair_chat" not in st.session_state:
#         st.session_state.repair_chat = []
#     if "repair_last_q" not in st.session_state:
#         st.session_state.repair_last_q = None
#     if "repair_plan" not in st.session_state:
#         st.session_state.repair_plan = None

#     # Ask next unanswered question
#     def get_next_repair_question():
#         for key, q in repair_questions:
#             if st.session_state.repair_info[key] in [None, ""]:
#                 return key, q
#         return None, None

#     # Chat input
#     repair_input = st.chat_input("Describe your repair project...")

#     if repair_input:
#         if st.session_state.repair_last_q:
#             st.session_state.repair_info[st.session_state.repair_last_q] = repair_input
#         st.session_state.repair_chat.append(UserMessage(content=repair_input))
#         next_key, next_q = get_next_repair_question()
#         st.session_state.repair_last_q = next_key

#         if next_q:
#             prompt = f"""
#             You are an expert NYC school repair planner.

#             Collected so far:
#             {json.dumps(st.session_state.repair_info, indent=2)}

#             Ask only the next missing question:
#             {next_q}
#             """
#         else:
#             prompt = f"""
#             All necessary repair info collected:
#             {json.dumps(st.session_state.repair_info, indent=2)}

#             Inform the user and ask if you'd like to generate a detailed plan with phases, subtasks, vendors, costs, labor, and materials.
#             """

#         messages = [SystemMessage(content=prompt)] + st.session_state.repair_chat
#         reply = client.chat.complete(model="mistral-small", messages=messages)
#         reply_text = reply.choices[0].message.content.strip()
#         st.session_state.repair_chat.append(SystemMessage(content=reply_text))

#     # Render chat
#     for msg in st.session_state.repair_chat:
#         role = "user" if isinstance(msg, UserMessage) else "assistant"
#         with st.chat_message(role):
#             st.markdown(msg.content)

#     # Plan generation
#     next_key, _ = get_next_repair_question()
#     if next_key is None:
#         st.session_state.collected_info = st.session_state.repair_info.copy()
#         if st.button("🛠 Generate Repair Plan"):
#             if "collected_info" not in st.session_state:
#                 st.session_state.collected_info = {}
#             repair_summary_prompt = f"""
#         Using the collected info, generate a detailed construction plan in **valid JSON format only**. Follow the exact structure described below.

#         Your JSON must include:
#         1. "ConstructionPhases" — a JSON array (not a dict) of 5–10 phases.
#         2. Each phase should have:
#         - "PhaseName": string (e.g., "I. Scope")
#         - "Description": string
#         - "EstimatedCost": number (USD)
#         - "DurationEstimate": number (weeks)
#         - "Subtasks": a JSON array of 5–10 objects. Each subtask must include:
#             - "SubtaskName": string
#             - "Description": string
#             - "CostEstimate": number (USD)
#             - "DurationEstimate": number (weeks)
#             - "LaborCategories": JSON array of strings
#             - "Vendors": JSON array of 1–2 real NYC-based vendors (not placeholders)
#             - "Permissions": JSON array of NYC government permissions (e.g., SCA, DOE, FDNY)
#         - "LaborCategories": JSON array of strings
#         - "Vendors": JSON array of strings
#         - "Permissions Required": JSON array of strings

#         3. "ResourcesAndMaterials" — a JSON array of raw materials. Each item must include:
#         - "Category": string (Name the phase in which material will be used)
#         - "Item": string (ONLY include relevant items based on user request. eg. if user mentions minor electrical repairs, DO NOT include unrelated construction materials like steel, concrete, or wood)
#         - "QuantityEstimate": string (include units, e.g., "5 metric tonnes" or quantity whatever relevant, eg. if its light fixtre then estimate how many will be needed based on user prompt)
#         - "EstimatedCost": number (USD) (estimate based on material and quantity)

#         ❗ JSON Formatting Rules:
#         - DO NOT use numeric keys like "0": {{...}}, "1": {{...}}. Use JSON arrays (square brackets []) instead.
#         - DO NOT include any text, explanation, or markdown outside the JSON.
#         - The output must be valid, parseable JSON and match this structure **exactly**.

#         Here is the user-provided context:
#         {json.dumps(st.session_state.collected_info, indent=2)}

#         Respond with only the JSON:
#         {{
#         "ConstructionPhases": [
#             {{
#             "PhaseName": "string",
#             "Description": "string",
#             "EstimatedCost": number,
#             "DurationEstimate": number,
#             "Subtasks": [
#                 {{
#                 "SubtaskName": "string",
#                 "Description": "string",
#                 "CostEstimate": number,
#                 "DurationEstimate": number,
#                 "LaborCategories": ["string"],
#                 "Vendors": ["string"],
#                 "Permissions": ["string"]
#                 }}
#             ],
#             "LaborCategories": ["string"],
#             "Vendors": ["string"],
#             "Permissions Required": ["string"]
#             }}
#         ],
#         "ResourcesAndMaterials": [
#             {{
#             "Category": "string",
#             "Item": "string",
#             "QuantityEstimate": "string",
#             "EstimatedCost": number
#             }}
#         ]
#         }}
#         """

#             messages = [
#                 SystemMessage(content="You summarize the project info and generate the final JSON plan."),
#                 UserMessage(content=repair_summary_prompt),
#             ]
#             response = client.chat.complete(model="mistral-small", messages=messages)
#             # st.session_state.repair_plan = response.choices[0].message.content.strip()
#             # Extract assistant message content
#             response_str = response.choices[0].message.content.strip()

#             # Save the raw response for reference
#             st.session_state.repair_plan_raw = response_str     
#             st.session_state.repair_plan = response_str 
#             st.session_state.repair_plan_parsed = None      

#     # Render final plan if exists
#     if st.session_state.repair_plan:
#         # Clean and parse
#         # One-time parser to avoid reparsing every rerun
#         if "repair_plan_parsed" not in st.session_state:
#             st.session_state.repair_plan_parsed = None

#         if st.session_state.repair_plan_raw and st.session_state.repair_plan_parsed is None:
#             raw_json_str = st.session_state.repair_plan_raw.strip().removeprefix("```json").removesuffix("```").strip()
#             try:
#                 parsed = json.loads(raw_json_str)
#                 st.session_state.repair_plan_parsed = parsed
#             except Exception as e:
#                 st.error("Invalid JSON: " + str(e))
#                 st.stop()

#         # Now safely reference parsed JSON
#         if st.session_state.repair_plan_parsed:
#             final = st.session_state.repair_plan_parsed
#             # Render tables, charts, etc. using `final`
#         else:
#             st.info("No valid repair plan found.")
#         st.subheader("🧰 Final Repair Plan")
#         # st.json(final)
#         def safe_format_cost(cost):
#             try:
#                 return f"${float(cost):,.2f}"
#             except (ValueError, TypeError):
#                 return "N/A"
#         # --- Phases Table ---
#         phases = final.get("ConstructionPhases", [])
#         for phase in phases:
#             with st.expander(f"📌 {phase['PhaseName']}", expanded=True):
#                 rows = [{
#                     "Task": phase["PhaseName"],
#                     "Description": phase.get("Description", ""),
#                     "Estimated Cost ($)": safe_format_cost(phase.get("EstimatedCost", 0)),
#                     "Duration (weeks)": phase.get("DurationEstimate", 0),
#                     "Labor Categories": ", ".join(phase.get("LaborCategories", [])),
#                     "Vendors": ", ".join(phase.get("Vendors", [])),
#                     "Permissions": ", ".join(phase.get("Permissions", [])),
#                 }]
#                 for sub in phase.get("Subtasks", []):
#                     rows.append({
#                         "Task": f"  ↳ {sub.get('SubtaskName', '')}",
#                         "Description": sub.get("Description", ""),
#                         "Estimated Cost ($)": safe_format_cost(sub.get("CostEstimate", 0)),
#                         "Duration (weeks)": sub.get("DurationEstimate", 0),
#                         "Labor Categories": ", ".join(sub.get("LaborCategories", [])),
#                         "Vendors": ", ".join(sub.get("Vendors", [])),
#                         "Permissions": ", ".join(phase.get("Permissions", [])),
#                     })
#                 st.dataframe(pd.DataFrame(rows), use_container_width=True)

#         # --- Materials Table ---
#         st.subheader("🧱 Resources & Materials")
#         resources = final.get("ResourcesAndMaterials", [])
#         mat_rows = []
#         for item in resources:
#             mat_rows.append({
#                 "Category": item.get("Category", ""),
#                 "Item": item.get("Item", ""),
#                 "Quantity Estimate": item.get("QuantityEstimate", "N/A"),
#                 "Estimated Cost": safe_format_cost(item.get("EstimatedCost", 0)),
#             })
#         st.dataframe(pd.DataFrame(mat_rows))

#         # --- Summary Chart ---
#         df_chart = pd.DataFrame({
#             "Phase": [p["PhaseName"] for p in phases],
#             "Cost": [p.get("EstimatedCost", 0) for p in phases],
#             "Duration": [p.get("DurationEstimate", 0) for p in phases],
#         })

#         st.subheader("💰 Cost Distribution")
#         fig = px.pie(df_chart, names="Phase", values="Cost", title="Cost by Phase", hole=0.4)
#         fig.update_traces(textposition="outside", textinfo="percent+label")
#         st.plotly_chart(fig, use_container_width=True)

#         st.subheader("⏱ Duration by Phase")
#         fig2 = px.line(df_chart, x="Phase", y="Duration", markers=True)
#         fig2.update_layout(yaxis_title="Weeks", xaxis_tickangle=-45)
#         st.plotly_chart(fig2, use_container_width=True)

#         st.markdown(f"**Total Estimated Cost:** ${int(df_chart['Cost'].sum()):,}")
#         st.markdown(f"**Total Estimated Duration:** {int(df_chart['Duration'].sum())} weeks")

# ################################################################        
#         st.divider()
#         st.subheader("🧮 ML-Based Cost & Schedule Estimates")

#         description = st.session_state.collected_info.get("ProjectDescription", "")  
#         bucket = st.session_state.get("bucket", "low")  # fallback to low

#         if st.button("Estimate Cost and Schedule (ML)", key="ml_estimate_button"):
#             with st.spinner("Running prediction model..."):
#                 try:
#                     result_df = predict_cost_duration(description, bucket,ai_durations)

#                     total_cost = result_df["Predicted Cost (USD)"].sum()
#                     total_duration = result_df["Predicted Duration (weeks)"].sum()

#                     result_df["Predicted Cost (USD)"] = result_df["Predicted Cost (USD)"].apply(lambda x: f"${x:,.2f}")
#                     result_df["Duration"] = result_df["Predicted Duration (weeks)"].apply(
#                         lambda w: f"{int(w)} weeks {int((w % 1) * 7)} days"
#                     )

#                     st.dataframe(result_df[["Phase", "Predicted Cost (USD)", "Duration"]], use_container_width=True)

#                     col1, col2 = st.columns(2)
#                     col1.metric("💰 Total Estimated Cost", f"${total_cost:,.2f}")
#                     col2.metric("🕒 Total Estimated Duration", f"{total_duration:.1f} weeks")

#                 except Exception as e:
#                     st.error(f"Prediction failed: {e}")

# # Add a back/reset button
# if st.button("🔙 Go Back"):
#     st.session_state.project_type = None