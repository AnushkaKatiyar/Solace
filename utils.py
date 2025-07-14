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

def log_user_activity(action: str, details: str = ""):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{timestamp},{action},{details}\n"
    try:
        with open("logs/activity_log.csv", "a") as f:
            f.write(log_line)
    except Exception as e:
        print(f"⚠️ Error writing log: {e}")

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
