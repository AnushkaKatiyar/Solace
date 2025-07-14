import streamlit as st
import pandas as pd
import os
import datetime
from utils import log_user_activity

st.set_page_config(page_title="User Center", layout="wide")
st.title("👤 User Dashboard")

# === Log Page View ===
log_user_activity("Page View", "User Dashboard")

# === Tabs ===
tab1, tab2 = st.tabs(["📊 Analytics", "💬 Feedback"])

# === Tab 1: Analytics ===
with tab1:
    st.subheader("📈 User Activity Logs")
    log_path = "logs/activity_log.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path, header=None, names=["Timestamp", "Action", "Details"])
        st.dataframe(df.tail(30), use_container_width=True)

        st.bar_chart(df["Action"].value_counts())

        df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
        st.line_chart(df.groupby("Date").size())
    else:
        st.info("No logs yet.")

# === Tab 2: Feedback Form ===
with tab2:
    st.subheader("📝 Submit Feedback")
    with st.form("feedback_form"):
        feedback = st.text_area("Your feedback about Solace:")
        email = st.text_input("Email (optional)", "")
        submitted = st.form_submit_button("Submit")

        if submitted and feedback.strip():
            try:
                os.makedirs("feedback", exist_ok=True)
                with open("feedback/feedback.csv", "a") as f:
                    f.write(f"{datetime.datetime.now()},{email},{feedback.strip()}\n")
                log_user_activity("Feedback Submitted", email)
                st.success("✅ Thank you! Your feedback has been recorded.")
            except Exception as e:
                st.error(f"❌ Error saving feedback: {e}")
