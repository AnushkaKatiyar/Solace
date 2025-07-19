import streamlit as st
from mistralai import Mistral, UserMessage, SystemMessage
import json

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

# Find next unanswered question key and text
def get_next_question():
    for key, question in questions:
        if st.session_state.collected_info[key] is None:
            return key, question
    return None, None

# On user input, save answer to current question and append messages
user_input = st.chat_input("Type your answer here...")

if user_input:
    # Append user message
    st.session_state.chat_history.append(UserMessage(content=user_input))
    
    # Find which question to assign this answer to
    current_key, current_question = get_next_question()
    if current_key is not None:
        st.session_state.collected_info[current_key] = user_input
    
    # Compose system prompt reminding assistant of its role and data so far

    # Find next unanswered question key and text
    def get_next_question():
        for key, question in questions:
            if st.session_state.collected_info[key] is None:
                return key, question
        return None, None

    next_key, next_question = get_next_question()

    system_prompt = f"""
    You are an expert NYC school construction planner assistant.
    The user has provided some information: {json.dumps(st.session_state.collected_info, indent=2)}.

    Please only ask the next missing question, which is:
    "{next_question}"

    Wait for the user's answer before asking anything else.
    If no questions remain, say 'Thank you, all information collected. You can now ask me to generate the plan.'
    """
    messages = [SystemMessage(content=system_prompt)] + st.session_state.chat_history
    
    response = client.chat.complete(
        model="mistral-medium",
        messages=messages,
    )
    assistant_reply = response.choices[0].message.content.strip()
    
    # Append assistant response
    st.session_state.chat_history.append(SystemMessage(content=assistant_reply))

# Display conversation history
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, UserMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Show next question automatically if any
next_key, next_question = get_next_question()
if next_question:
    with st.chat_message("assistant"):
        st.markdown(next_question)

# When all questions answered, show button to generate plan
if next_key is None:
    if st.button("🚧 Generate Construction Plan"):
        summary_prompt = f"""
Using the collected info, generate a detailed construction plan in JSON format with phases, subtasks, vendors, permissions, materials, and labor.

Collected info:
{json.dumps(st.session_state.collected_info, indent=2)}

Only output JSON with this structure:
{{
  "ConstructionPhases": [...],
  "Resources & Materials": {{...}}
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
