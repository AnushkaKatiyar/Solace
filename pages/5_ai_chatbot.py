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

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def format_money(value):
    return f"${int(value):,}"

def format_duration(value):
    return f"{int(value)} weeks"

def render_construction_output(parsed_json):
    phases = parsed_json["ConstructionPhases"]
    materials = parsed_json.get("Resources & Materials", {})
    
    total_duration = 0
    total_cost = 0

    st.subheader("📋 Construction Plan Overview")

    for phase in phases:
        st.markdown(f"### 🔹 {phase['PhaseName']}")
        st.markdown(f"**Description:** {phase['Description']}")
        st.markdown(f"**Duration:** {format_duration(phase['DurationEstimate'])}")
        st.markdown(f"**Cost:** {format_money(phase['EstimatedCost'])}")

        total_duration += phase["DurationEstimate"]
        total_cost += phase["EstimatedCost"]

        # Phase-level metadata
        st.markdown("**Vendors:** " + ", ".join(phase["Vendors"]))
        st.markdown("**Permissions Required:** " + ", ".join(phase["Permissions Required"]))
        st.markdown("**Labor Categories:** " + ", ".join(phase["LaborCategories"]))

        # Subtasks Table
        if phase["Subtasks"]:
            sub_df = pd.DataFrame([
                {
                    "Subtask": s["SubtaskName"],
                    "Description": s["Description"],
                    "Duration": format_duration(s["DurationEstimate"]),
                    "Cost": format_money(s["CostEstimate"]),
                    "Vendors": ", ".join(s["Vendors"]),
                    "Permissions": ", ".join(s["Permissions"]),
                    "Labor": ", ".join(s["LaborCategories"])
                }
                for s in phase["Subtasks"]
            ])
            st.markdown("**Subtasks:**")
            st.dataframe(sub_df, use_container_width=True)

    # Resources & Materials Section
    st.markdown("---")
    st.subheader("🛠️ Resources & Materials")

    if materials:
        for category, items in materials.items():
            st.markdown(f"**{category}**")
            if isinstance(items, list):
                st.write(", ".join(items))
            elif isinstance(items, dict):
                mat_df = pd.DataFrame([{"Item": k, "Details": v} for k, v in items.items()])
                st.dataframe(mat_df, use_container_width=True)
            else:
                st.write(items)
    else:
        st.write("No materials info available.")

    # Charts
    st.markdown("---")
    st.subheader("📊 Duration and Cost Charts")

    chart_df = pd.DataFrame([
        {
            "Phase": p["PhaseName"],
            "Duration (weeks)": p["DurationEstimate"],
            "Cost ($)": p["EstimatedCost"]
        }
        for p in phases
    ])

    # Duration Line Chart
    st.markdown("**⏱️ Duration per Phase**")
    st.line_chart(chart_df.set_index("Phase")["Duration (weeks)"])

    # Cost Pie Chart
    st.markdown("**💸 Cost Distribution**")
    fig, ax = plt.subplots()
    ax.pie(chart_df["Cost ($)"], labels=chart_df["Phase"], autopct='%1.0f%%', startangle=140)
    st.pyplot(fig)

    # Summary
    st.markdown("---")
    st.subheader("📌 Summary")
    st.markdown(f"**Total Duration:** {format_duration(total_duration)}")
    st.markdown(f"**Total Estimated Cost:** {format_money(total_cost)}")

