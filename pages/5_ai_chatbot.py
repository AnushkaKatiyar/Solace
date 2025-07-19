import streamlit as st
from mistralai import Mistral, UserMessage, SystemMessage
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load API key from Streamlit secrets
mistral_api_key = st.secrets["mistral_api_key"]
client = Mistral(api_key=mistral_api_key)

st.set_page_config(page_title="AI Chatbot Assistant", layout="wide")
st.title("🛠️ AI Assistant for NYC School Construction")

# Define the questions to ask sequentially
questions = [
    ("Location", "Which part of NYC is the school located in?"),
    ("Grades", "How many grades will the school have?"),
    ("StudentsPerClass", "What is the average number of students per class?"),
    ("Timeline", "What is the expected construction timeline (in months)?"),
    ("SpecialReqs", "Are there any special facilities or requirements needed?"),
    ("SquareFootage", "What is the square footage of the construction?"),
    ("Floors", "How many floors will the building have?"),
    ("DemolitionNeeded", "Is demolition needed?"),
    ("LotAvailable", "If a feature, is the lot already available?"),
    ("Basement", "Is a basement needed?"),
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

Ask only the next missing question once.
Do NOT repeat previous questions or user answers.
Wait for user's answer before asking anything else.
If all questions are answered, tell the user that all info is collected and they can ask to generate the plan.

Next question:
{next_question}
"""
    else:
        system_prompt = f"""
You have collected all the necessary project information:
{json.dumps(st.session_state.collected_info, indent=2)}

Inform the user that all info is collected and ask if they want to generate the construction plan.
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
    st.session_state.chat_history.append(SystemMessage(content=assistant_reply))

# Display the full chat history
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, UserMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# When all questions answered, show button to generate plan
next_key, next_question = get_next_question()
if next_key is None:
    if st.button("🚧 Generate Construction Plan"):
        summary_prompt = f"""
Using the collected info, generate a detailed construction plan in JSON format with phases, subtasks, vendors, permissions, materials, and labor.

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
      "Permissions Required": []
    }}
  ],
  "Resources & Materials": {{
    "CategoryName": [
      {{
        "Item": "string",
        "QuantityEstimate": number,
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

# Display final plan JSON if exists
if st.session_state.final_plan:
    st.subheader("📦 Final Construction Plan")
    st.code(st.session_state.final_plan, language="json")


###Rendering the json on UI

try:
    parsed_json = json.loads(final_json)
    st.session_state.final_plan = parsed_json
except json.JSONDecodeError:
    st.error("Failed to parse JSON. Please make sure the AI returned valid JSON.")
    st.stop()

import pandas as pd

if "final_plan" in st.session_state:
    plan = st.session_state.final_plan
    phases = plan.get("ConstructionPhases", [])

    st.subheader("📋 Construction Phases & Subtasks")

    for phase in phases:
        st.markdown(f"### 🏗️ {phase['PhaseName']}")
        st.markdown(f"**Description:** {phase['Description']}")

        phase_data = {
            "Phase": phase["PhaseName"],
            "Description": phase["Description"],
            "Duration (weeks)": f"{int(round(phase['DurationEstimate']))} weeks",
            "Cost ($)": "${:,.0f}".format(phase["EstimatedCost"]),
            "Vendors": ", ".join(phase["Vendors"]),
            "Permissions": ", ".join(phase["Permissions Required"]),
        }
        st.table(pd.DataFrame([phase_data]))

        # Subtasks Table
        st.markdown("#### 🔧 Subtasks")
        subtask_rows = []
        for sub in phase["Subtasks"]:
            subtask_rows.append({
                "Subtask": sub["SubtaskName"],
                "Description": sub["Description"],
                "Duration (weeks)": f"{int(round(sub['DurationEstimate']))} weeks",
                "Cost ($)": "${:,.0f}".format(sub["CostEstimate"]),
                "Labor": ", ".join(sub["LaborCategories"]),
                "Vendors": ", ".join(sub["Vendors"]),
                "Permissions": ", ".join(sub["Permissions"]),
            })
        st.table(pd.DataFrame(subtask_rows))

st.subheader("🧱 Resources & Materials")
resources = plan.get("Resources & Materials", {})

if resources:
    materials_df = pd.DataFrame([
        {"Category": key, "Items": ", ".join(value)} for key, value in resources.items()
    ])
    st.table(materials_df)
else:
    st.info("No resources or materials specified.")

# Optional: Display unique labor categories used across all phases
all_labors = set()
for phase in phases:
    all_labors.update(phase.get("LaborCategories", []))
    for sub in phase["Subtasks"]:
        all_labors.update(sub.get("LaborCategories", []))

if all_labors:
    st.subheader("👷 Labor Categories")
    st.markdown(", ".join(sorted(all_labors)))

import matplotlib.pyplot as plt

# For summary values
total_cost = 0
total_duration = 0
phase_labels = []
phase_costs = []
phase_durations = []

for phase in phases:
    phase_labels.append(phase["PhaseName"])
    cost = phase["EstimatedCost"]
    duration = phase["DurationEstimate"]
    total_cost += cost
    total_duration += duration
    phase_costs.append(cost)
    phase_durations.append(duration)

# Cost Pie Chart
st.subheader("💰 Cost Distribution")
fig1, ax1 = plt.subplots()
ax1.pie(phase_costs, labels=phase_labels, autopct='%1.1f%%', startangle=140)
ax1.axis('equal')
st.pyplot(fig1)

# Duration Line Chart
st.subheader("⏱ Duration by Phase")
fig2, ax2 = plt.subplots()
ax2.plot(phase_labels, phase_durations, marker='o')
ax2.set_ylabel("Duration (weeks)")
ax2.set_title("Duration by Phase")
st.pyplot(fig2)

st.subheader("📊 Summary Totals")

st.markdown(f"**Total Estimated Cost:** ${total_cost:,.0f}")
st.markdown(f"**Total Estimated Duration:** {int(round(total_duration))} weeks (~{int(round(total_duration / 4))} months)")

