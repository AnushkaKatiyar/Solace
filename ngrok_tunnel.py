import os
import toml
import subprocess
import time
from pyngrok import ngrok

# Load secrets manually from .streamlit/secrets.toml
secrets = toml.load(".streamlit/secrets.toml")
ngrok_token = secrets.get("ngrok_auth_token")

# Set ngrok token
ngrok.set_auth_token(ngrok_token)

# Connect to port 8501 where Streamlit runs
public_url = ngrok.connect(8501)
print(f"🌍 Public URL: {public_url}")

# Launch Streamlit app
print("🚀 Launching Streamlit app...")
subprocess.Popen(["streamlit", "run", "app.py"])

# Keep the tunnel open
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("🛑 Ngrok tunnel closed.")

