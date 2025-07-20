import streamlit as st
from mistralai import Mistral, UserMessage, SystemMessage
import json
import pandas as pd
import matplotlib.pyplot as plt
import re

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

Output should be a list of 5-10 phases, depending on the user inputs. Each phase must include:
- Phase: (string) e.g. "I. Scope",
- Description: (string),a short description,
- Subphases/subtaskes: 5-10 sub tasks within the phases
- Subphase Breakdown: (list of phases and subtasks(5-10 phases and 5-10 subtasks) from above as dicts). Each dict must have:
  - Name: (string)
  - Description(string)
  - Cost (USD): (number)
  - Labor Category
  - Vendor: (list of strings),1–2 **actual NYC-based vendors or well-known relevant companies** (avoid placeholders like 'VendorX', 'VendorA'),
  - Permission if needed: (list of strings),required NYC government permissions (e.g., SCA, DoE, FDNY),
  - Duration (weeks): (number)
- Resources & Material-Raw materials used in construction
  - Item-should have the name and describe for which phases and subtask it is needed
  - Quantity-in correct units e.g-metric tonne, feet etc
  - Cost (USD): (number)
  

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

def clean_json_string(raw_json):
    return raw_json.strip().removeprefix("```json").removesuffix("```").strip()

if st.session_state.final_plan:
    # Optional: Add a header
    st.subheader("📦 Final Construction Plan")

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
    # You can now use final_plan to render sections, tables, etc.
    

# #Display final plan JSON if exists
# if st.session_state.final_plan:
#     st.subheader("📦 Final Construction Plan")
#     st.code(
#         st.session_state.final_plan
#         if isinstance(st.session_state.final_plan, str)
#         else json.dumps(st.session_state.final_plan, indent=2),
#         language="json",
#     )
    
#     if isinstance(st.session_state.final_plan, str):
#         st.write("Raw JSON string:", st.session_state.final_plan)    
#         cleaned = clean_json_string(st.session_state.final_plan)
#         st.write("Cleaned JSON string:", cleaned)
#         try:
#             parsed_json = json.loads(cleaned)
#             st.session_state.final_plan = parsed_json
#         except json.JSONDecodeError as e:
#             st.error(f"JSON decode failed: {e}")
#             st.stop()
#     else:
#         st.write("Parsed plan (dict):", st.session_state.final_plan)

if "final_plan" in st.session_state and st.session_state.final_plan is not None:
    plan = st.session_state.final_plan
    phases = plan.get("ConstructionPhases", [])
    st.subheader("📋 Construction Phases & Subtasks")
    # Iterate through each phase
    for phase in final_plan.get("phases", []):
        with st.expander(f"{phase.get('phase_name', 'Unnamed Phase')}"):
            # Display summary info as a table
            summary_data = {
                "Estimated Cost": [phase.get("estimated_cost", "N/A")],
                "Estimated Duration (weeks)": [phase.get("estimated_duration", "N/A")],
                "Labor Needs": [", ".join(phase.get("labor", []))],
                "Vendors": [", ".join(phase.get("vendors", []))],
                "Permissions": [", ".join(phase.get("permissions", []))]
            }

        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)

        # Display subtasks (if any)
        subtasks = phase.get("subtasks", [])
        if subtasks:
            st.markdown("**Subtasks:**")
            for i, task in enumerate(subtasks, start=1):
                st.markdown(f"- {i}. {task}")
    # for phase in phases:
    #     st.markdown(f"### 🏗️ {phase['PhaseName']}")
    #     st.markdown(f"**Description:** {phase.get('Description', '')}")
    #     phase_data = {
    #         "Phase": phase["PhaseName"],
    #         "Description": phase.get("Description", ""),
    #         "Duration (weeks)": f"{int(round(phase.get('DurationEstimate', 0)))} weeks",
    #         "Cost ($)": "${:,.0f}".format(phase.get("EstimatedCost", 0)),
    #         "Vendors": ", ".join(phase.get("Vendors", [])),
    #         "Permissions": ", ".join(phase.get("Permissions Required", [])),
    #     }
    #     st.table(pd.DataFrame([phase_data]))
    #     # Subtasks Table
    #     st.markdown("#### 🔧 Subtasks")
    #     subtask_rows = []
    #     for sub in phase.get("Subtasks", []):
    #         subtask_rows.append(
    #             {
    #                 "Subtask": sub.get("SubtaskName", ""),
    #                 "Description": sub.get("Description", ""),
    #                 "Duration (weeks)": f"{int(round(sub.get('DurationEstimate', 0)))} weeks",
    #                 "Cost ($)": "${:,.0f}".format(sub.get("CostEstimate", 0)),
    #                 "Labor": ", ".join(sub.get("LaborCategories", [])),
    #                 "Vendors": ", ".join(sub.get("Vendors", [])),
    #                 "Permissions": ", ".join(sub.get("Permissions", [])),
    #             }
    #         )
    #     st.table(pd.DataFrame(subtask_rows))
    st.subheader("🧱 Resources & Materials")
    resources = plan.get("Resources & Materials", {})
    if resources:
        # Corrected this part to handle list of dicts per category
        materials_list = []
        for category, items in resources.items():
            items_str = ", ".join([item.get("Item", "") for item in items])
            materials_list.append({"Category": category, "Items": items_str})
        materials_df = pd.DataFrame(materials_list)
        st.table(materials_df)
    else:
        st.info("No resources or materials specified.")
    # Display unique labor categories used across all phases and subtasks
    all_labors = set()
    for phase in phases:
        all_labors.update(phase.get("LaborCategories", []))
        for sub in phase.get("Subtasks", []):
            all_labors.update(sub.get("LaborCategories", []))

    if all_labors:
        st.subheader("👷 Labor Categories")
        st.markdown(", ".join(sorted(all_labors)))

    # Summary charts
    total_cost = 0
    total_duration = 0
    phase_labels = []
    phase_costs = []
    phase_durations = []

    for phase in phases:
        phase_labels.append(phase["PhaseName"])
        cost = phase.get("EstimatedCost", 0)
        duration = phase.get("DurationEstimate", 0)
        total_cost += cost
        total_duration += duration
        phase_costs.append(cost)
        phase_durations.append(duration)

    # Cost Pie Chart
    st.subheader("💰 Cost Distribution")
    fig1, ax1 = plt.subplots()
    ax1.pie(phase_costs, labels=phase_labels, autopct="%1.1f%%", startangle=140)
    ax1.axis("equal")
    st.pyplot(fig1)

    # Duration Line Chart
    st.subheader("⏱ Duration by Phase")
    fig2, ax2 = plt.subplots()
    ax2.plot(phase_labels, phase_durations, marker="o")
    ax2.set_ylabel("Duration (weeks)")
    ax2.set_title("Duration by Phase")
    st.pyplot(fig2)

    st.subheader("📊 Summary Totals")
    st.markdown(f"**Total Estimated Cost:** ${total_cost:,.0f}")
    st.markdown(
        f"**Total Estimated Duration:** {int(round(total_duration))} weeks (~{int(round(total_duration / 4))} months)"
    )
