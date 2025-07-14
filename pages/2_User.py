import streamlit as st
import pandas as pd
import os
import datetime
from utils import log_user_activity, save_feedback, send_feedback_email
import uuid

st.set_page_config(page_title="User Center", layout="wide")
st.title("👤 User Dashboard")

# Generate/get user_id for logging
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
user_id = st.session_state.user_id

# Log page view with user id
log_user_activity(user_id, "User Dashboard", "Page View")

tab1, tab2 = st.tabs(["📊 Analytics", "💬 Feedback"])

# Tab 1: Analytics - show recent user logs
with tab1:
    st.subheader("📈 User Activity Logs")
    log_path = "logs/activity_log.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path, names=["UserID", "Timestamp", "Page", "Action"])
        # Filter logs for this user only (optional)
        user_df = df[df["UserID"] == user_id]
        st.dataframe(user_df.tail(30), use_container_width=True)
        st.bar_chart(user_df["Action"].value_counts())
        user_df["Date"] = pd.to_datetime(user_df["Timestamp"]).dt.date
        st.line_chart(user_df.groupby("Date").size())
    else:
        st.info("No logs yet.")

# Tab 2: Feedback form
with tab2:
    st.subheader("📝 Submit Feedback")
    with st.form("feedback_form"):
        feedback = st.text_area("Your feedback about Solace:")
        email = st.text_input("Email (optional)", "")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if not feedback.strip():
                st.error("Please enter feedback before submitting.")
            else:
                save_feedback(user_id, email, feedback.strip())
                log_user_activity(user_id, "User Dashboard", "Feedback Submitted")

                st.success("✅ Thank you! Your feedback has been recorded.")

                # Send notification email if SMTP creds available
                if st.secrets.get("smtp_server") and st.secrets.get("smtp_email") and st.secrets.get("smtp_password"):
                    try:
                        send_feedback_email(
                            to_email=st.secrets["smtp_email"],
                            subject="New Feedback Received",
                            body=f"User ID: {user_id}\nEmail: {email}\nFeedback:\n{feedback.strip()}",
                            smtp_server=st.secrets["smtp_server"],
                            smtp_port=int(st.secrets.get("smtp_port", 465)),
                            sender_email=st.secrets["smtp_email"],
                            sender_password=st.secrets["smtp_password"],
                        )
                    except Exception as e:
                        st.warning(f"Feedback saved but failed to send notification email: {e}")
