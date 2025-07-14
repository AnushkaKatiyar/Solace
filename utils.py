import csv
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime

LOGS_CSV = "logs/user_activity.csv"
FEEDBACK_CSV = "feedback/feedback_data.csv"

# Ensure folders exist
os.makedirs(os.path.dirname(LOGS_CSV), exist_ok=True)
os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)

def log_user_activity(user_id: str, page_name: str, action: str) -> None:
    """Log user activity to CSV with timestamp."""
    timestamp = datetime.utcnow().isoformat()
    file_exists = os.path.isfile(LOGS_CSV)
    
    with open(LOGS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "user_id", "page_name", "action"])
        writer.writerow([timestamp, user_id, page_name, action])

def save_feedback(user_id: str, email: str, feedback_text: str) -> None:
    """Save user feedback to CSV with timestamp."""
    timestamp = datetime.utcnow().isoformat()
    file_exists = os.path.isfile(FEEDBACK_CSV)
    
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "user_id", "email", "feedback"])
        writer.writerow([timestamp, user_id, email, feedback_text])

def send_feedback_email(to_email: str, subject: str, body: str, 
                        smtp_server: str, smtp_port: int, 
                        sender_email: str, sender_password: str) -> None:
    """Send feedback notification email using SMTP."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)
