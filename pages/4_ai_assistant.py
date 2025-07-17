import streamlit as st
import json
import pandas as pd
import plotly.express as px
from mistralai import Mistral, UserMessage, SystemMessage

# === Mistral API Client Setup ===
mistral_api_key = st.secrets["mistral_api_key"]
client = Mistral(api_key=mistral_api_key)

# === Questions to ask one by one ===
questions = [
    "Which part of NYC is the school located in?",
    "How many grades will the school have?",
    "What is the average number of students per class?",
    "What is the expected construction timeline (in months)?",
    "Are there any special facilities or requirements needed?"
]

if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "answers" not in st.session_state:
    st.session_state.answers = [""] * len(questions)
if "plan_json" not in st.session_state:
    st.session_state.plan_json = None
if "loading" not in st.session_state:
    st.session_state.loading = False

def next_question():
    if st.session_state.current_question < len(questions) - 1:
        st.session_state.current_question += 1

def prev_question():
    if st.session_state.current_question > 0:
        st.session_state.current_question -= 1

def generate_plan(description, answers):
    prompt = f"""
You are an expert NYC school construction planner. 
Based on the following information, generate a very detailed construction cost and scheduling plan including phases, subtasks, permissions, vendors, labor, materials, equipment categories, furniture, and labor categories.

Project description:
{description}

Answers to follow-up questions:
1. Location: {answers[0]}
2. Number of grades: {answers[1]}
3. Students per class: {answers[2]}
4. Construction timeline (months): {answers[3]}
5. Special requirements: {answers[4]}

Output a JSON with the following format (only JSON, no extra text):

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
      "Permissions Required": []
    }}
  ],
  "Resources & Materials": {{...}} 
}}
"""
    messages = [
        SystemMessage(content="You are a helpful assistant for NYC school construction planning."),
        UserMessage(content=prompt),
    ]
    response = client.chat.complete(model="mistral-medium", messages=messages)
    return response.choices[0].message.content


st.title("AI Assistant: NYC School Construction Cost & Schedule Plan")

# Step 0: Get project description from user
description = st.text_area("Enter a short project description", 
    value="Create a very detailed cost and scheduling plan for new school construction.")

# Show the current question and input box
st.markdown(f"### Question {st.session_state.current_question + 1} of {len(questions)}")
answer = st.text_input(questions[st.session_state.current_question], 
                       value=st.session_state.answers[st.session_state.current_question],
                       key="answer_input")

# Save the answer
st.session_state.answers[st.session_state.current_question] = answer

# Navigation buttons
col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button("Previous") and st.session_state.current_question > 0:
        prev_question()
with col2:
    next_clicked = st.button("Next")
    if next_clicked:
        if answer.strip() == "":
            st.warning("Please answer before continuing.")
        else:
            next_question()
            
with col3:
    if st.button("Submit All Answers"):
        if any(not a.strip() for a in st.session_state.answers):
            st.warning("Please answer all questions before submitting.")
        else:
            st.session_state.loading = True
            with st.spinner("Generating detailed construction plan..."):
                try:
                    plan_text = generate_plan(description, st.session_state.answers)
                    # Try parsing JSON safely
                    json_start = plan_text.find("{")
                    json_end = plan_text.rfind("}") + 1
                    plan_json_str = plan_text[json_start:json_end]
                    st.session_state.plan_json = json.loads(plan_json_str)
                except Exception as e:
                    st.error(f"Failed to generate or parse plan JSON: {e}")
                    st.session_state.plan_json = None
            st.session_state.loading = False


# --- AFTER SUBMISSION: Display Table + Charts ---
if st.session_state.plan_json:
    data = st.session_state.plan_json
    phases = data.get("ConstructionPhases", [])

    # Prepare rows for DataFrame: main phase + subtasks each as a row
    rows = []
    total_cost = 0
    total_duration = 0

    for i, phase in enumerate(phases, 1):
        subtasks = phase.get("Subtasks", [])
        
        # Aggregate subtasks info
        agg_duration = 0
        agg_cost = 0
        all_labor = set()
        all_vendors = set()
        all_permissions = set()

        for stask in subtasks:
            agg_duration += stask.get("DurationEstimate", 0) if isinstance(stask.get("DurationEstimate"), (int, float)) else 0
            agg_cost += stask.get("CostEstimate", 0) if isinstance(stask.get("CostEstimate"), (int, float)) else 0
            all_labor.update(stask.get("LaborCategories", []))
            all_vendors.update(stask.get("Vendors", []))
            all_permissions.update(stask.get("Permissions", []))

        # Fallback to phase-level if no subtasks or aggregate zero
        if agg_duration == 0:
            agg_duration = phase.get("DurationEstimate", 0)
        if agg_cost == 0:
            agg_cost = phase.get("EstimatedCost", 0)
        if not all_labor:
            all_labor.update(phase.get("LaborCategories", []))
        if not all_vendors:
            all_vendors.update(phase.get("Vendors", []))
        if not all_permissions:
            all_permissions.update(phase.get("Permissions Required", []))

        total_cost += agg_cost
        try:
            total_duration += float(agg_duration)
        except (ValueError, TypeError):
            pass  # skip if not a number

        # Add main phase row
        rows.append({
            "Phase": f"{i}. {phase.get('PhaseName', 'Unnamed Phase')}",
            "Description": phase.get("Description", ""),
            "Duration (weeks)": agg_duration,
            "Estimated Cost ($)": agg_cost,
            "Labor Categories": ", ".join(sorted(all_labor)) if all_labor else "N/A",
            "Vendors": ", ".join(sorted(all_vendors)) if all_vendors else "N/A",
            "Permissions": ", ".join(sorted(all_permissions)) if all_permissions else "N/A",
            "IsPhase": True
        })

        # Add subtask rows (indented)
        for st_idx, stask in enumerate(subtasks, 1):
            rows.append({
                "Phase": f" {stask.get('SubtaskName', f'Subtask {st_idx}')}",
                "Description": stask.get("Description", ""),
                "Duration (weeks)": stask.get("DurationEstimate", "N/A"),
                "Estimated Cost ($)": stask.get("CostEstimate", "N/A"),
                "Labor Categories": ", ".join(stask.get("LaborCategories", [])) if stask.get("LaborCategories") else "N/A",
                "Vendors": ", ".join(stask.get("Vendors", [])) if stask.get("Vendors") else "N/A",
                "Permissions": ", ".join(stask.get("Permissions", [])) if stask.get("Permissions") else "N/A",
                "IsPhase": False
            })
    df = pd.DataFrame(rows)

    # Show table with some styling for main phases vs subtasks
    def highlight_phases(row):
        return ['font-weight: bold; background-color: #e6f2ff' if row.IsPhase else '' for _ in row]

    st.header("🗂 Construction Phases Summary")
    st.dataframe(df.style.apply(highlight_phases, axis=1), use_container_width=True, height=1000)

    # Charts
    if not df.empty:
        # Filter only main phases for charts
        phases_df = df[df["IsPhase"] == True]
        phases_df["Estimated Cost ($)"] = pd.to_numeric(phases_df["Estimated Cost ($)"], errors="coerce").fillna(0)
        fig_cost = px.pie(phases_df, values="Estimated Cost ($)", names="Phase", title="Cost Distribution by Phase")
        st.plotly_chart(fig_cost, use_container_width=True)

        fig_duration = px.line(phases_df, x="Phase", y="Duration (weeks)", markers=True, title="Duration by Phase")
        st.plotly_chart(fig_duration, use_container_width=True)

    st.markdown(f"### Total Estimated Cost: ${total_cost:,.2f}")
    st.markdown(f"### Total Estimated Duration: {total_duration} weeks")

    st.caption("🧪 Disclaimer: This is a prototype by Solace Technologies. Estimates are AI-generated and may not reflect actual costs.")


