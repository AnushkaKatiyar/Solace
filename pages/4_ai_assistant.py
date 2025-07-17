import streamlit as st
import json
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
  "ConstructionPhases": [...], 
  "Resources & Materials": {...} 
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
    if st.button("Next") and st.session_state.current_question < len(questions) - 1:
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

# After submission, show the detailed plan nicely formatted
if st.session_state.plan_json:
    data = st.session_state.plan_json
    
    st.header("🗂 Construction Phases")
    for i, phase in enumerate(data.get("ConstructionPhases", []), 1):
        st.subheader(f"{i}. {phase.get('Phase','No phase name')} – {phase.get('Description','')}")
        
        st.markdown("**Subtasks:**")
        subtasks = phase.get("Subtasks", [])
        # If subtasks are dicts, format keys nicely
        if subtasks and isinstance(subtasks[0], dict):
            for st_idx, subtask in enumerate(subtasks, 1):
                st.markdown(f"- **Subtask {st_idx}:**")
                for key, val in subtask.items():
                    # If val is a list, join nicely, else show value
                    if isinstance(val, list):
                        val_str = ", ".join(str(x) for x in val)
                    else:
                        val_str = str(val)
                    st.markdown(f"  - {key}: {val_str}")
        else:
            # Subtasks might just be strings
            for st_idx, subtask in enumerate(subtasks, 1):
                st.markdown(f"- {subtask}")

        # Permissions Required
        perms = phase.get("Permissions Required") or phase.get("Permissions", [])
        if perms:
            st.markdown(f"**Permissions Required:** {', '.join(perms)}")
        else:
            st.markdown("**Permissions Required:** N/A")

        # Vendors
        vendors = phase.get("Vendors", [])
        if vendors:
            st.markdown(f"**Vendors:** {', '.join(vendors)}")
        else:
            st.markdown("**Vendors:** N/A")

        # Estimated Labor
        est_lab = phase.get("Estimated Labor")
        if est_lab is not None:
            st.markdown(f"**Estimated Labor:** {est_lab:,} workers")
        else:
            st.markdown("**Estimated Labor:** N/A workers")

        # Subphase Breakdown
        subphases = phase.get("Subphase Breakdown", [])
        if subphases:
            st.markdown("**Subphase Breakdown:**")
            for sp in subphases:
                name = sp.get("Name", "Unknown")
                duration = sp.get("Duration (weeks)", "N/A")
                cost = sp.get("Cost (USD)", 0)
                st.markdown(f"- {name}: {duration} weeks, Cost: ${cost:,.2f}")
        else:
            st.markdown("**Subphase Breakdown:** N/A")

        st.markdown("---")
