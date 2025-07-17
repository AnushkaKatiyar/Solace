import streamlit as st
import json
import pandas as pd
import plotly.express as px
from mistralai import Mistral, UserMessage, SystemMessage

# --- SETUP ---
st.set_page_config(page_title="AI Assistant Planning", layout="wide")

mistral_api_key = st.secrets["mistral_api_key"]
client = Mistral(api_key=mistral_api_key)

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

def ask_ai(project_description, answers):
    prompt = f"""
You are an expert school construction planner in New York City.
Based on the following information, generate a very detailed cost and duration plan.

Project Description:
{project_description}

Follow-up answers:
1. Location: {answers[0]}
2. Grades: {answers[1]}
3. Students per class: {answers[2]}
4. Timeline: {answers[3]} months
5. Special requirements: {answers[4]}

Format your answer as a JSON object with the following:
- A list called `phases`, where each phase contains:
  - name, description, duration_weeks, estimated_cost
  - subtasks: list of strings
  - permissions: list of strings
  - vendors: list of strings
  - materials, labor, equipment, furniture: dicts (category -> details)
"""
    messages = [
        SystemMessage(content="You are a precise, expert-level construction cost planner."),
        UserMessage(content=prompt)
    ]
    response = client.chat(model="mistral-large-latest", messages=messages)
    return json.loads(response.choices[0].message.content)

st.title("AI Assistant: School Construction Plan")

project_description = st.text_area("Describe the construction project:", height=150)

# Q&A Loop
if st.session_state.plan_json is None:
    q_num = st.session_state.current_question

    st.subheader(f"Step {q_num + 1}: {questions[q_num]}")

    # Use a unique key per question so inputs keep state when navigating back/forth
    answer_key = f"answer_input_{q_num}"
    answer = st.text_input("Answer:", value=st.session_state.answers[q_num], key=answer_key)

    st.session_state.answers[q_num] = answer  # Save immediately after input

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("⬅️ Back", disabled=q_num == 0):
            st.session_state.current_question = max(q_num - 1, 0)
    with col2:
        if st.button("➡️ Next", disabled=q_num == len(questions) - 1 or answer.strip() == ""):
            st.session_state.current_question = min(q_num + 1, len(questions) - 1)
    with col3:
        # Disable the button only if any answer is empty or project description missing
        disabled_submit = any(not a.strip() for a in st.session_state.answers) or not project_description.strip()
        if st.button("Generate Plan", disabled=disabled_submit):
            st.session_state.loading = True
            with st.spinner("Generating plan from AI..."):
                try:
                    plan = ask_ai(project_description, st.session_state.answers)
                    st.session_state.plan_json = plan
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                finally:
                    st.session_state.loading = False

# Display results
if st.session_state.plan_json:
    st.success("✅ Plan generated successfully!")
    phases = st.session_state.plan_json.get("phases", [])

    df = pd.DataFrame([
        {
            "Phase": p["name"],
            "Duration (weeks)": p["duration_weeks"],
            "Estimated Cost ($)": p["estimated_cost"]
        }
        for p in phases
    ])
    st.dataframe(df, use_container_width=True)

    fig1 = px.pie(df, values="Estimated Cost ($)", names="Phase", title="Cost Distribution by Phase")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(df, x="Phase", y="Duration (weeks)", markers=True, title="Duration by Phase")
    st.plotly_chart(fig2, use_container_width=True)

    for p in phases:
        with st.expander(f"📦 {p['name']} Details"):
            st.markdown(f"**Description:** {p['description']}")
            st.markdown("**Subtasks:**")
            st.markdown("- " + "\n- ".join(p.get("subtasks", [])))

            st.markdown("**Permissions Needed:** " + ", ".join(p.get("permissions", [])))
            st.markdown("**Vendors:** " + ", ".join(p.get("vendors", [])))

            def dict_to_md_table(d, title):
                if not d:
                    return f"_No {title.lower()} listed._"
                table = "| Category | Details |\n|---|---|\n"
                table += "\n".join([f"| {k} | {v} |" for k, v in d.items()])
                return table

            st.markdown("**Materials:**")
            st.markdown(dict_to_md_table(p.get("materials", {}), "Materials"))

            st.markdown("**Labor:**")
            st.markdown(dict_to_md_table(p.get("labor", {}), "Labor"))

            st.markdown("**Equipment:**")
            st.markdown(dict_to_md_table(p.get("equipment", {}), "Equipment"))

            st.markdown("**Furniture:**")
            st.markdown(dict_to_md_table(p.get("furniture", {}), "Furniture"))

    st.caption("🧪 Disclaimer: This is a prototype by Solace Technologies. Estimates are AI-generated and may not reflect actual costs.")
