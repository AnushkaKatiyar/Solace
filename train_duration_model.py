import pandas as pd
import numpy as np
import re
import os
import pickle
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from xgboost import XGBRegressor

# --- Config ---
DATA_PATH = "data/Capital_Project_Schedules_and_Budgets.csv"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Download stopwords ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Load Data ---
df = pd.read_csv(DATA_PATH)

# --- Feature Engineering ---
df["start_date"] = pd.to_datetime(df["Project Phase Actual Start Date"], errors="coerce", format="%m/%d/%Y")
actual_end = pd.to_datetime(df["Project Phase Actual End Date"], errors="coerce", format="%m/%d/%Y")
planned_end = pd.to_datetime(df["Project Phase Planned End Date"], errors="coerce", format="%m/%d/%Y")
df["end_date"] = actual_end.combine_first(planned_end)
df["project_status"] = df["Project Status Name"].str.strip().str.upper()
df.loc[df["project_status"] == "PNS", ["start_date", "end_date"]] = pd.NaT
df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days
df["timeline_status"] = np.where(
    df["project_status"] == "PNS", "Not Started",
    np.where(df["end_date"].isna(), "Incomplete", "Available")
)

# --- Text Cleaning ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['description_clean'] = df['Project Description'].apply(clean_text)
df['description_no_stopwords'] = df['description_clean'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# --- Fill missing ---
df['description_no_stopwords'] = df['description_no_stopwords'].fillna('')
df['Project Phase Name'] = df['Project Phase Name'].fillna('Unknown')

# --- Prepare modeling dataset ---
df_model = df[['duration_days', 'description_no_stopwords', 'Project Phase Name', 'project_status', 'timeline_status']].dropna(subset=['duration_days'])
df_model['duration_weeks'] = df_model['duration_days'] / 7.0

# --- Sentence Embedding Model ---
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_bert_embeddings(text_series):
    return bert_model.encode(text_series.tolist(), show_progress_bar=True)

# --- Encoding ---
cat_cols = ['Project Phase Name', 'project_status', 'timeline_status']
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(df_model[cat_cols])

def prepare_features(df_sub):
    bert_embeddings = get_bert_embeddings(df_sub['description_no_stopwords'])
    cat_feats = ohe.transform(df_sub[cat_cols])
    return np.hstack([bert_embeddings, cat_feats])

# --- Train-test split ---
X = prepare_features(df_model)
y = df_model['duration_weeks'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Hyperparameter Tuning ---
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42,
        'objective': 'reg:squarederror',
        'verbosity': 0
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# --- Final model ---
best_params = study.best_params
best_params.update({'random_state': 42, 'objective': 'reg:squarederror', 'verbosity': 0})
final_model = XGBRegressor(**best_params)
final_model.fit(X, y)

# --- Save model & encoder ---
with open(os.path.join(MODEL_DIR, 'duration_model.pkl'), 'wb') as f:
    pickle.dump(final_model, f)

with open(os.path.join(MODEL_DIR, 'ohe_duration.pkl'), 'wb') as f:
    pickle.dump(ohe, f)

print("✅ Duration model and encoder saved in 'models/' folder.")

# --- Evaluation ---
y_pred = final_model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"\n📊 Duration Model — RMSE: {rmse:.2f} weeks | R²: {r2:.3f}")
