import streamlit as st
from mistralai import Mistral, UserMessage, SystemMessage
import json
import os

# Load your Mistral API key securely
mistral_api_key = st.secrets["mistral_api_key"]
client = Mistral(api_key=mistral_api_key)

st.set_page_config(page_title="AI Chatbot Assistant", layout="wide")
st.title("🛠️ AI Assistant for NYC School Construction")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "collected_info" not in st.session_state:
    st.session_state.collected_info = {
        "Location": None,
        "Grades": None,
        "StudentsPerClass": None,
        "Timeline": None,
        "SpecialReqs": None,
        "SquareFootage": None,
        "Floors": None,
        "DemolitionNeeded": None,
        "LotAvailable": None,
        "Basement": None,
    }

if "final_plan" not in st.session_state:
    st.session_state.final_plan = None

# Capture user input
user_input = st.chat_input("Ask me anything about your school construction project...")

if user_input:
    st.session_state.chat_history.append(UserMessage(content=user_input))

    # System prompt
    system_prompt = """
You are an expert NYC school construction planner assistant. 
Extract relevant planning details from the conversation and store them. 
Once you have enough information, generate a structured construction plan in JSON, with phases, subtasks, vendors, permissions, materials, and labor. 
Wait to respond with a plan until user has provided sufficient detail.

When generating the plan, follow this format:
{
  "ConstructionPhases": [...],
  "Resources & Materials": {...}
}
Only return JSON when asked to "generate plan".
"""

    # Compose messages
    messages = [SystemMessage(content=system_prompt)] + st.session_state.chat_history

    # Call the model
    response = client.chat(
        model="mistral-medium",
        messages=messages,
    )
    assistant_reply = response.choices[0].message.content.strip()
    st.session_state.chat_history.append(SystemMessage(content=assistant_reply))

# Display conversation
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, UserMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Generate Construction Plan button
if st.button("🚧 Generate Construction Plan"):
    user_conversation = "\n".join(
        msg.content for msg in st.session_state.chat_history if isinstance(msg, UserMessage)
    )
    summary_prompt = f"""
Here is the full conversation. Extract all relevant project info and return a structured JSON plan.

Conversation:
{user_conversation}

Follow this structure exactly:
{{
  "ConstructionPhases": [...],
  "Resources & Materials": {{...}}
}}
Only return JSON, no explanation.
"""
    messages = [
        SystemMessage(content="You summarize the conversation and generate the final JSON plan."),
        UserMessage(content=summary_prompt),
    ]
    response = client.chat(model="mistral-medium", messages=messages)
    final_json = response.choices[0].message.content.strip()
    st.session_state.final_plan = final_json

# Show final plan
if st.session_state.final_plan:
    st.subheader("📦 Final Construction Plan")
    st.code(st.session_state.final_plan, language="json")
