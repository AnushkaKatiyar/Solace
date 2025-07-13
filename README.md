# Step 1: Open your README.md file in VS Code or any editor and paste the following content:

cat > README.md <<EOF
# 🏗️ Solace – NYC School Construction Estimator

Solace is an intelligent, interactive Streamlit web application designed to estimate the **cost and schedule** of school construction projects in New York City. Powered by machine learning models and the Mistral LLM API, it delivers data-driven insights and detailed construction plans for all 8 project phases.

## ✨ Features

- 🔍 Estimate total cost and duration based on a project description and cost tier (Low, Mid, High)
- 📋 Get phase-wise breakdowns of:
  - Description
  - Subtasks
  - Required NYC permissions (SCA, DoE, FDNY)
  - Vendors and labor estimation
- 📊 Interactive charts and metrics for better visualization
- 💡 Dark mode toggle
- 🎥 Lottie animations for modern UI
- 📤 Downloadable results as Excel and PDF (coming soon)
- 🧠 Uses:
  - Mistral LLM API for phase plan generation
  - SentenceTransformer (BERT) for embedding descriptions
  - Trained ML models for cost and duration prediction

## 🚀 Tech Stack

- Python, Streamlit
- Scikit-learn, XGBoost, Pandas, NumPy
- Sentence Transformers (MiniLM)
- Mistral AI
- Streamlit Lottie
- Pickle for model serialization

## 🛠️ Setup Instructions

1. Clone the repo  
\`\`\`bash
git clone https://github.com/your-username/solace.git
cd solace
\`\`\`

2. Install dependencies  
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Add your Mistral API key in \`.streamlit/secrets.toml\`  
\`\`\`toml
[mistral]
mistral_api_key = "your_mistral_key_here"
\`\`\`

4. Run the app  
\`\`\`bash
streamlit run Home.py
\`\`\`

## 👩‍💻 Created By

**Anushka Katiyar**  
🎓 MS ECE (ML/DS) @ University of Southern California  
🔗 [LinkedIn](https://www.linkedin.com/in/anushka-katiyar12/) | [GitHub](https://github.com/AnushkaKatiyar)

---

## 📌 Disclaimer

This project is intended for educational/demo purposes only. Predictions are estimates and should not be used for real construction decisions without validation.
EOF

