# ai_assistant.py

import streamlit as st
from mistralai import Mistral, UserMessage, SystemMessage

# === Setup ===
st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("🧠 AI Assistant — NYC School Construction Planner")

mistral_api_key = st.secrets["mistral_api_key"]
client = Mistral(api_key=mistral_api_key)

# === Session state ===
if "stage" not in st.session_state:
    st.session_state.stage = "initial_input"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "follow_up_questions" not in st.session_state:
    st.session_state.follow_up_questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []

# === Stage 1: Initial User Prompt ===
if st.session_state.stage == "initial_input":
    user_input = st.chat_input("Describe your school construction project...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        followup_prompt = f"""
You are a helpful and expert assistant specializing in NYC public school construction planning. A user has just described their project, and your task is to ask 4–5 intelligent follow-up questions that will help you generate a complete, phase-by-phase cost and schedule plan.

The questions should help clarify:
- The type and size of the school
- Location-specific constraints
- Facility needs
- Student/grade information
- Timeline or budget expectations

Be concise, and ask all questions in one message. Here's the user’s input:

\"{user_input}\"

Ask 4–5 questions before you begin planning.
"""

        response = client.chat.complete(
            model="mistral-medium",
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=followup_prompt),
            ]
        )
        questions = response.choices[0].message.content
        st.session_state.follow_up_questions = questions.strip().split("\n")
        st.session_state.stage = "collecting_answers"

# === Stage 2: Ask Follow-Up Questions ===
if st.session_state.stage == "collecting_answers":
    st.markdown("### 📋 Please answer the following questions:")
    with st.form("followup_form"):
        inputs = []
        for i, q in enumerate(st.session_state.follow_up_questions):
            answer = st.text_input(label=q, key=f"answer_{i}")
            inputs.append(answer)
        submitted = st.form_submit_button("Submit Answers")

    if submitted:
        st.session_state.answers = inputs
        st.session_state.stage = "generating_plan"

# === Stage 3: Generate Final Plan ===
if st.session_state.stage == "generating_plan":
    st.info("⏳ Generating detailed plan from AI...")

    # Combine user input + answers
    original_input = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"][0]
    combined_info = f"User description: {original_input}\n\nFollow-up Answers:\n"
    for q, a in zip(st.session_state.follow_up_questions, st.session_state.answers):
        combined_info += f"- {q.strip()}: {a.strip()}\n"

    final_prompt = f"""
You are a helpful construction planning assistant for NYC public school projects. Based on the details below, generate a complete JSON construction plan.

Use the following details to prepare a plan:
{combined_info}

Output format (valid JSON only):
A list of 8 construction phases. For each phase, include:
- "Phase": Name (e.g. "I. Scope")
- "Description": Short summary
- "Subtasks": List of 6–10 detailed subtasks
- "Permissions Required": List of NYC-specific agencies (e.g., SCA, DOE, DOB, FDNY)
- "Vendors": List of 1–2 NYC-based vendors (real or commonly known)
- "Estimated Labor": Integer (number of workers)
- "Subphase Breakdown": List of dicts with:
  - "Name"
  - "Duration (weeks)"
  - "Cost (USD)"

Also include a second section (same JSON output) titled **"Resources & Materials"**:
- "Materials Needed": List of quantities for bricks, concrete, steel, glass, etc.
- "Equipment (by category)": Electrical, plumbing, bathroom, classroom, outdoors
- "Furniture Needed": Desks, chairs, boards, storage, etc.
- "Labor Categories": Types of workers required (e.g., electricians, plumbers, masons)
- "Special Notes": Anything else that would impact cost or schedule.

Do not include any explanations, just return valid JSON only.
"""

    response = client.chat.complete(
        model="mistral-medium",
        messages=[
            SystemMessage(content="You are a helpful school construction assistant."),
            UserMessage(content=final_prompt),
        ]
    )

    plan_output = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": plan_output})
    st.session_state.stage = "display_plan"

# === Stage 4: Show Plan ===
if st.session_state.stage == "display_plan":
    st.markdown("### ✅ AI-Generated Construction Plan")
    st.code(st.session_state.chat_history[-1]["content"], language="json")
