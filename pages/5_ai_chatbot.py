import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import json
import os

# Load your Mistral API key securely
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = MistralClient(api_key=MISTRAL_API_KEY)

st.set_page_config(page_title="AI Chatbot Assistant", layout="wide")
st.title("🛠️ AI Assistant for NYC School Construction")

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

user_input = st.chat_input("Ask me anything about your school construction project...")

if user_input:
    st.session_state.chat_history.append(ChatMessage(role="user", content=user_input))

    # Keep prompt smart and dynamic
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

    chat_messages = [ChatMessage(role="system", content=system_prompt)] + st.session_state.chat_history
    response = client.chat(model="mistral-medium", messages=chat_messages)
    assistant_reply = response.choices[0].message.content.strip()
    st.session_state.chat_history.append(ChatMessage(role="assistant", content=assistant_reply))

# Display chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg.role):
        st.markdown(msg.content)

# Add Generate Button
if st.button("🚧 Generate Construction Plan"):
    summary_prompt = f"""
Here is the full conversation. Extract all relevant project info and return a structured JSON plan.

Conversation:
{[msg.content for msg in st.session_state.chat_history if msg.role == "user"]}

Follow this structure exactly:
{{
  "ConstructionPhases": [...],
  "Resources & Materials": {{...}}
}}
Only return JSON, no explanation.
"""
    response = client.chat(model="mistral-medium", messages=[
        ChatMessage(role="system", content="You summarize the conversation and generate the final JSON plan."),
        ChatMessage(role="user", content=summary_prompt),
    ])
    final_json = response.choices[0].message.content.strip()
    st.session_state.final_plan = final_json

if st.session_state.final_plan:
    st.subheader("📦 Final Construction Plan")
    st.code(st.session_state.final_plan, language="json")
