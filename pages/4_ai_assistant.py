import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import pandas as pd

# Initialize Mistral
api_key = st.secrets["MISTRAL_API_KEY"]  # or manually: "sk-..."
client = MistralClient(api_key=api_key)
model = "mistral-small-latest"

# Session state
if "followups" not in st.session_state:
    st.session_state.followups = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "final_plan" not in st.session_state:
    st.session_state.final_plan = None

st.title("📋 AI Assistant for School Construction Planning")

# Step 1: User Input
user_input = st.text_area("📝 Describe your construction project:", height=100)

# Step 2: Follow-up loop
if user_input and not st.session_state.followups:
    with st.spinner("🤖 Thinking of follow-up questions..."):
        prompt = f"""You're a school construction planning AI. Based on the project description below, ask 4–5 specific follow-up questions that will help you generate a detailed plan including 8 phases, costs, durations, labor, vendors, permissions, materials, and equipment.

Project description:
{user_input}

Respond ONLY with numbered questions in plain English.
"""
        response = client.chat(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)]
        )
        questions = response.choices[0].message.content.strip().split("\n")
        st.session_state.followups = [q for q in questions if q.strip()]

# Step 3: Show and answer follow-ups
answers = []
if st.session_state.followups:
    st.subheader("🔍 Follow-up Questions")
    for i, q in enumerate(st.session_state.followups):
        answer = st.text_input(f"{q}", key=f"answer_{i}")
        answers.append(answer)

# Step 4: Generate plan
if st.button("🚀 Generate Plan"):
    if not all(answers):
        st.warning("Please answer all the follow-up questions before generating the plan.")
    else:
        with st.spinner("🛠️ Generating your detailed plan..."):
            messages = [
                ChatMessage(role="system", content="You are an expert school construction planning assistant."),
                ChatMessage(role="user", content=f"Project: {user_input}"),
            ]
            for i, (q, a) in enumerate(zip(st.session_state.followups, answers)):
                messages.append(ChatMessage(role="user", content=f"{q}\n{a}"))

            messages.append(ChatMessage(role="user", content="""
Now generate a detailed JSON output that includes the following for each of the 8 standard construction phases:

- PhaseName
- Subtasks (6-10) with:
    - SubtaskName
    - EstimatedCostUSD (integer)
    - EstimatedDurationWeeks (integer)
    - LaborNeeded (list of roles)
Make sure each Phase has a name and valid numeric cost/duration.
"""))

            final_response = client.chat(model=model, messages=messages)
            try:
                # Parse JSON from Mistral response
                import json
                plan_data = json.loads(final_response.choices[0].message.content)
                st.session_state.final_plan = plan_data
            except Exception as e:
                st.error(f"❌ Failed to parse AI response as JSON: {e}")
                st.stop()

# Step 5: Show Output Table
if st.session_state.final_plan:
    st.subheader("📊 Detailed Construction Plan")
    records = []
    for idx, phase in enumerate(st.session_state.final_plan.get("Phases", []), start=1):
        phase_name = phase.get("PhaseName", f"Phase {idx}")
        for st_idx, subtask in enumerate(phase.get("Subtasks", []), start=1):
            records.append({
                "Phase": f"{phase_name}",
                " Subtask": f" {subtask.get('SubtaskName', f'Subtask {st_idx}')}",
                "Cost (USD)": subtask.get("EstimatedCostUSD", "N/A"),
                "Duration (Weeks)": subtask.get("EstimatedDurationWeeks", "N/A"),
                "Labor": ", ".join(subtask.get("LaborNeeded", []))
            })
    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True)
